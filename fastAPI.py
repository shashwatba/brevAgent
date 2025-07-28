from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

load_dotenv()

app = FastAPI(title="Learning Extension API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NVIDIA-compatible ChatOpenAI
# This works because NVIDIA's API is OpenAI-compatible
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model=os.getenv("NVIDIA_MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1"),
    temperature=0.7,
    max_tokens=2000
)

# Create data directory
DATA_DIR = "quiz_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

QUIZ_HISTORY_FILE = os.path.join(DATA_DIR, "quiz_history.json")

# Pydantic models
class LearningTopicsRequest(BaseModel):
    topics: List[str]

class KeywordsByTopicResponse(BaseModel):
    keywords_by_topic: Dict[str, List[str]]

class QuizRequest(BaseModel):
    topic: str
    keywords: List[str]
    difficulty: str = "medium"

class Question(BaseModel):
    question: str
    choice1: str
    choice2: str
    choice3: str
    choice4: str
    correct: str
    keyword: str
    difficulty: str = "medium"

class QuizResponse(BaseModel):
    topic: str
    questions: List[Question]
    generated_at: str
    quiz_id: str

class UserAnswer(BaseModel):
    question_index: int
    selected_answer: str
    is_correct: bool

class QuizSubmission(BaseModel):
    quiz_id: str
    topic: str
    keyword: str
    user_answers: List[UserAnswer]
    completed_at: str

# LangChain Tools
def generate_keywords_tool(topic: str) -> str:
    """Generate keywords for a learning topic"""
    prompt = f"""Generate a list of 10-15 important keywords or key terms that someone learning about "{topic}" should encounter and understand. 

These keywords should be:
- Core concepts, terms, or technologies related to {topic}
- Things that would commonly appear in articles, tutorials, or discussions about {topic}
- Fundamental building blocks of knowledge in this area

Return only the keywords as a simple comma-separated list, no explanations or numbering.

Example format: keyword1, keyword2, keyword3, etc."""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    return response.content.strip()

def generate_quiz_questions_tool(topic: str, keywords: str, difficulty: str) -> str:
    """Generate quiz questions for a topic and keywords"""
    difficulty_instruction = {
        "big": "The question should be hard/difficult and take some thought to solve. Give some context for the question.",
        "small": "The question should be simple and easy to answer.",
        "medium": "The question should be medium difficulty."
    }.get(difficulty, "The question should be medium difficulty.")
    
    prompt = f"""You are a kind and curious tutor that helps the user learn {topic}.
Create multiple choice quiz questions about "{topic}" focusing on these keywords: {keywords}

{difficulty_instruction}

Generate 3-5 multiple choice questions. Each question should have 4 answer choices.

Format your response as a JSON array like this:
[
  {{
    "question": "The quiz question text here?",
    "choice1": "First answer option text",
    "choice2": "Second answer option text", 
    "choice3": "Third answer option text",
    "choice4": "Fourth answer option text",
    "correct": "A",
    "keyword": "relevant_keyword",
    "difficulty": "{difficulty}"
  }}
]

Important:
- The question should test understanding of the keyword in context
- choice1 through choice4 should contain ONLY the answer text
- correct should be the letter (A, B, C, or D) of the correct answer
- Make the incorrect options plausible but clearly wrong
- Questions should be educational and test real understanding
- Return ONLY the JSON array, no other text"""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    return response.content.strip()

# Create structured tools for the agent
keyword_tool = StructuredTool.from_function(
    func=generate_keywords_tool,
    name="generate_keywords",
    description="Generate learning keywords for a topic. Input should be the topic name.",
    return_direct=False
)

quiz_tool = StructuredTool.from_function(
    func=generate_quiz_questions_tool,
    name="generate_quiz",
    description="Generate quiz questions for a topic with keywords. Inputs should be topic, keywords (comma-separated), and difficulty level.",
    return_direct=False
)

tools = [keyword_tool, quiz_tool]

# Create agent prompt
system_prompt = """You are an AI tutor assistant that helps create educational content.
You have access to tools to generate keywords and quiz questions.

When asked to generate keywords, use the generate_keywords tool with just the topic.
When asked to generate quiz questions, use the generate_quiz tool with the topic, keywords, and difficulty.

Always use the appropriate tool for the task and return the raw tool output."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    return_intermediate_steps=True
)

# Helper functions
def load_quiz_history() -> List[Dict]:
    try:
        if os.path.exists(QUIZ_HISTORY_FILE):
            with open(QUIZ_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading quiz history: {str(e)}")
        return []

def save_quiz_history(quizzes: List[Dict]):
    try:
        with open(QUIZ_HISTORY_FILE, 'w') as f:
            json.dump(quizzes, f, indent=2)
    except Exception as e:
        print(f"Error saving quiz history: {str(e)}")

def save_quiz_to_history(quiz_data: Dict):
    history = load_quiz_history()
    history.append(quiz_data)
    save_quiz_history(history)

def update_quiz_submission(quiz_id: str, submission: QuizSubmission):
    history = load_quiz_history()
    
    for quiz in history:
        if quiz.get('quiz_id') == quiz_id:
            quiz['user_answers'] = [answer.dict() for answer in submission.user_answers]
            quiz['completed_at'] = submission.completed_at
            quiz['score'] = sum(1 for answer in submission.user_answers if answer.is_correct)
            break
    
    save_quiz_history(history)

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Learning Extension API v2.0 with LangChain and NVIDIA!",
        "model": os.getenv("NVIDIA_MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1"),
        "powered_by": "NVIDIA AI + LangChain"
    }

@app.post("/generate-keywords", response_model=KeywordsByTopicResponse)
async def generate_keywords(request: LearningTopicsRequest):
    try:
        if not os.getenv("NVIDIA_API_KEY"):
            raise HTTPException(status_code=500, detail="NVIDIA API key not configured")
        
        keywords_by_topic = {}
        
        for topic in request.topics:
            if not topic.strip():
                continue
            
            # Use the agent to generate keywords
            result = agent_executor.invoke({
                "input": f"Use the generate_keywords tool to generate keywords for: {topic}",
                "chat_history": []
            })
            
            # Extract the keywords from the agent's output
            output = result.get("output", "")
            
            # Parse the keywords
            keywords = [k.strip() for k in output.split(',') if k.strip()]
            keywords = keywords[:15]  # Limit to 15 keywords
            
            if keywords:
                keywords_by_topic[topic.strip()] = keywords
        
        if not keywords_by_topic:
            raise HTTPException(status_code=400, detail="No valid topics provided")
        
        return KeywordsByTopicResponse(keywords_by_topic=keywords_by_topic)
    
    except Exception as e:
        print(f"Error in generate_keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating keywords: {str(e)}")

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    try:
        if not os.getenv("NVIDIA_API_KEY"):
            raise HTTPException(status_code=500, detail="NVIDIA API key not configured")
        
        if not request.topic.strip():
            raise HTTPException(status_code=400, detail="No topic provided")
        
        if not request.keywords:
            raise HTTPException(status_code=400, detail="No keywords provided")
        
        quiz_id = str(uuid.uuid4())
        keywords_str = ", ".join(request.keywords)
        
        # Use the agent to generate quiz questions
        result = agent_executor.invoke({
            "input": f"Use the generate_quiz tool to create quiz questions for topic '{request.topic}' with keywords '{keywords_str}' at difficulty level '{request.difficulty}'",
            "chat_history": []
        })
        
        # Extract the JSON from the agent's output
        output = result.get("output", "")
        
        try:
            # Find and parse JSON
            start_idx = output.find('[')
            end_idx = output.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = output[start_idx:end_idx]
                questions_data = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
            
            # Convert to Question objects
            questions = []
            for q_data in questions_data:
                keyword = q_data.get("keyword", request.keywords[0])
                if keyword not in request.keywords:
                    keyword = request.keywords[0]
                
                questions.append(Question(
                    question=q_data["question"],
                    choice1=q_data["choice1"],
                    choice2=q_data["choice2"],
                    choice3=q_data["choice3"],
                    choice4=q_data["choice4"],
                    correct=q_data["correct"],
                    keyword=keyword,
                    difficulty=q_data.get("difficulty", request.difficulty)
                ))
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing quiz JSON: {str(e)}")
            print(f"Raw output: {output}")
            # Create a fallback question
            questions = [Question(
                question=f"What is the most important concept in {request.topic}?",
                choice1=f"Understanding {request.keywords[0]}",
                choice2="Memorizing without understanding",
                choice3="Ignoring documentation",
                choice4="None of the above",
                correct="A",
                keyword=request.keywords[0],
                difficulty=request.difficulty
            )]
        
        if not questions:
            raise HTTPException(status_code=400, detail="No questions could be generated")
        
        generated_at = datetime.now().isoformat()
        quiz_data = {
            "quiz_id": quiz_id,
            "topic": request.topic,
            "keyword": request.keywords[0],
            "questions": [question.dict() for question in questions],
            "user_answers": [],
            "generated_at": generated_at,
            "completed_at": None,
            "score": None,
            "total_questions": len(questions)
        }
        
        save_quiz_to_history(quiz_data)
        
        return QuizResponse(
            topic=request.topic,
            questions=questions,
            generated_at=generated_at,
            quiz_id=quiz_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@app.post("/submit-quiz")
async def submit_quiz(submission: QuizSubmission):
    try:
        update_quiz_submission(submission.quiz_id, submission)
        
        return {
            "message": "Quiz submission saved successfully",
            "quiz_id": submission.quiz_id,
            "score": sum(1 for answer in submission.user_answers if answer.is_correct),
            "total": len(submission.user_answers)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting quiz: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_status = "configured" if os.getenv("NVIDIA_API_KEY") else "missing"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "nvidia_api": api_status,
        "model": os.getenv("NVIDIA_MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1"),
        "agent": "langchain"
    }

@app.get("/test-llm")
async def test_llm():
    """Test LLM connection"""
    try:
        if not os.getenv("NVIDIA_API_KEY"):
            return {"status": "error", "message": "NVIDIA API key not configured"}
        
        # Test the LLM directly
        messages = [HumanMessage(content="Say hello in one sentence.")]
        response = llm.invoke(messages)
        
        return {
            "status": "success",
            "message": "LLM is working",
            "response": response.content,
            "model": os.getenv("NVIDIA_MODEL_NAME")
        }
    except Exception as e:
        return {"status": "error", "message": f"LLM error: {str(e)}"}

@app.get("/test-agent")
async def test_agent():
    """Test the LangChain agent"""
    try:
        if not os.getenv("NVIDIA_API_KEY"):
            return {"status": "error", "message": "NVIDIA API key not configured"}
        
        result = agent_executor.invoke({
            "input": "Say hello and tell me what tools you have access to.",
            "chat_history": []
        })
        
        return {
            "status": "success",
            "message": "Agent is working",
            "response": result.get("output"),
            "tools": [tool.name for tool in tools]
        }
    except Exception as e:
        return {"status": "error", "message": f"Agent error: {str(e)}"}

# Include remaining endpoints from original (quiz history, stats, etc.)
@app.get("/quiz-history")
async def get_quiz_history():
    try:
        history_data = load_quiz_history()
        return {"quizzes": history_data, "total_count": len(history_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving quiz history: {str(e)}")

@app.get("/quiz-stats")
async def get_quiz_stats():
    try:
        history_data = load_quiz_history()
        
        total_quizzes = len(history_data)
        completed_quizzes = len([q for q in history_data if q.get("completed_at")])
        
        completed_with_scores = [q for q in history_data if q.get("score") is not None and q.get("total_questions", 0) > 0]
        avg_score_percentage = 0
        if completed_with_scores:
            total_percentage = sum((q["score"] / q["total_questions"]) * 100 for q in completed_with_scores)
            avg_score_percentage = total_percentage / len(completed_with_scores)
        
        topic_counts = {}
        for quiz in history_data:
            topic = quiz.get("topic", "Unknown")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_quizzes": total_quizzes,
            "completed_quizzes": completed_quizzes,
            "pending_quizzes": total_quizzes - completed_quizzes,
            "average_score_percentage": round(avg_score_percentage, 2),
            "topics": topic_counts
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving quiz stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)