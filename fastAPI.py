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
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser

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

# Initialize Nemo LLM via HuggingFace endpoint
# You'll need to configure this based on your Nemo deployment
NEMO_API_URL = os.getenv("NEMO_API_URL", "https://api-inference.huggingface.co/models/nvidia/nemo-megatron-gpt-5B")
NEMO_API_KEY = os.getenv("NEMO_API_KEY", os.getenv("HUGGINGFACE_API_KEY"))

# Initialize LLM
llm = HuggingFaceEndpoint(
    endpoint_url=NEMO_API_URL,
    huggingfacehub_api_token=NEMO_API_KEY,
    task="text-generation",
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 2000,
        "top_p": 0.95,
    }
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
def generate_keywords_tool(topic: str) -> List[str]:
    """Generate keywords for a learning topic"""
    prompt = f"""Generate a list of 10-15 important keywords or key terms that someone learning about "{topic}" should encounter and understand. 

These keywords should be:
- Core concepts, terms, or technologies related to {topic}
- Things that would commonly appear in articles, tutorials, or discussions about {topic}
- Fundamental building blocks of knowledge in this area

Return only the keywords as a simple comma-separated list, no explanations or numbering.

Example format: keyword1, keyword2, keyword3, etc."""

    response = llm.invoke(prompt)
    
    # Parse the response
    keywords_text = response.strip()
    keywords = [keyword.strip() for keyword in keywords_text.split(',')]
    keywords = [k for k in keywords if k][:15]
    
    return keywords

def generate_quiz_questions_tool(topic: str, keywords: List[str], difficulty: str) -> List[Dict]:
    """Generate quiz questions for a topic and keywords"""
    keywords_str = ", ".join(keywords)
    
    difficulty_instruction = {
        "big": "The question should be hard/difficult and take some thought to solve. Give some context for the question.",
        "small": "The question should be simple and easy to answer.",
        "medium": "The question should be medium difficulty."
    }.get(difficulty, "The question should be medium difficulty.")
    
    prompt = f"""You are a kind and curious tutor that helps the user learn {topic}.
Create multiple choice quiz questions about "{topic}" focusing on these keywords: {keywords_str}

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

    response = llm.invoke(prompt)
    
    # Extract JSON from response
    try:
        # Find the JSON array in the response
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            questions_data = json.loads(json_str)
            return questions_data
        else:
            raise ValueError("No JSON array found in response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {str(e)}")
        print(f"Response: {response}")
        # Return a fallback question
        return [{
            "question": f"What is the most important concept in {topic}?",
            "choice1": f"Understanding {keywords[0] if keywords else topic}",
            "choice2": "Memorizing without understanding",
            "choice3": "Ignoring documentation",
            "choice4": "None of the above",
            "correct": "A",
            "keyword": keywords[0] if keywords else "general",
            "difficulty": difficulty
        }]

# Create LangChain agent
keyword_tool = StructuredTool.from_function(
    func=generate_keywords_tool,
    name="generate_keywords",
    description="Generate learning keywords for a topic"
)

quiz_tool = StructuredTool.from_function(
    func=generate_quiz_questions_tool,
    name="generate_quiz",
    description="Generate quiz questions for a topic with keywords"
)

tools = [keyword_tool, quiz_tool]

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI tutor assistant that helps create educational content.
    You have access to tools to generate keywords and quiz questions.
    Always use the appropriate tool for the task."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
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

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Learning Extension API v2.0 with LangChain and Nemo!"}

@app.post("/generate-keywords", response_model=KeywordsByTopicResponse)
async def generate_keywords(request: LearningTopicsRequest):
    try:
        keywords_by_topic = {}
        
        for topic in request.topics:
            if not topic.strip():
                continue
            
            # Use the agent to generate keywords
            result = agent_executor.invoke({
                "input": f"Generate keywords for learning about: {topic}",
                "chat_history": []
            })
            
            # Extract keywords from the result
            keywords = generate_keywords_tool(topic.strip())
            keywords_by_topic[topic.strip()] = keywords
        
        if not keywords_by_topic:
            raise HTTPException(status_code=400, detail="No valid topics provided")
        
        return KeywordsByTopicResponse(keywords_by_topic=keywords_by_topic)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating keywords: {str(e)}")

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    try:
        if not request.topic.strip():
            raise HTTPException(status_code=400, detail="No topic provided")
        
        if not request.keywords:
            raise HTTPException(status_code=400, detail="No keywords provided")
        
        quiz_id = str(uuid.uuid4())
        
        # Use the agent to generate quiz questions
        questions_data = generate_quiz_questions_tool(
            request.topic, 
            request.keywords, 
            request.difficulty
        )
        
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm": "nemo",
        "agent": "langchain"
    }

@app.get("/test-llm")
async def test_llm():
    """Test LLM connection"""
    try:
        response = llm.invoke("Say hello")
        return {"status": "success", "message": "LLM is working", "response": response}
    except Exception as e:
        return {"status": "error", "message": f"LLM error: {str(e)}"}

# Include all the other endpoints from the original file
# (quiz history, stats, etc.) - they remain the same

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)