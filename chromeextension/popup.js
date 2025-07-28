// popup.js - Handles the extension popup interface

// Use your Brev deployment URL
const API_BASE_URL = process.env.NODE_ENV === 'development' 
    ? 'http://localhost:8000'
    : 'https://camel-tutor-backend.brev.dev'; // Replace with your actual Brev URL

document.addEventListener('DOMContentLoaded', async () => {
    const difficultySlider = document.getElementById('difficulty-slider');
    const difficultyLabel = document.getElementById('difficulty-label');

    difficultySlider.addEventListener('input', () => {
        const value = difficultySlider.value;
        if (value === '1') {
            difficultyLabel.textContent = 'Small';
        } else if (value === '2') {
            difficultyLabel.textContent = 'Medium';
        } else {
            difficultyLabel.textContent = 'Large';
        }
    });
    
    const topicInput = document.getElementById('topic-input');
    const startButton = document.getElementById('start-tracking');
    const formatSelector = document.getElementById('quiz-format');
    const statusDiv = document.getElementById('status');
    const topicsListDiv = document.getElementById('topics-list');
    const trackingInfoDiv = document.getElementById('tracking-info');
    const clearButton = document.getElementById('clear-data');
    
    // Check API health on load
    checkAPIHealth();
    
    // Load and display current tracking data
    await displayCurrentTracking();
    
    // Handle start tracking button
    startButton.addEventListener('click', async () => {
        const topic = topicInput.value.trim();
        
        if (!topic) {
            showStatus('Please enter a topic to learn', 'error');
            return;
        }
        
        startButton.disabled = true;
        showStatus('Generating keywords...', 'success');
        
        try {
            const selectedFormat = formatSelector.value;
            await chrome.storage.local.set({ quizFormat: selectedFormat });
            
            const response = await fetch(`${API_BASE_URL}/generate-keywords`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ topics: [topic] })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to generate keywords');
            }
            
            const data = await response.json();
            const keywords = data.keywords_by_topic[topic];
            
            // Store the topic and keywords
            const storage = await chrome.storage.local.get(['learningTopics']);
            const topics = storage.learningTopics || {};
            
            topics[topic] = {
                keywords: keywords,
                keywordCounts: {},
                createdAt: new Date().toISOString(),
                threshold: 5
            };
            
            // Initialize keyword counts
            keywords.forEach(keyword => {
                topics[topic].keywordCounts[keyword] = 0;
            });
            
            await chrome.storage.local.set({ learningTopics: topics });
            
            showStatus(`Started tracking ${keywords.length} keywords for "${topic}"`, 'success');
            topicInput.value = '';
            
            // Refresh the display
            await displayCurrentTracking();
            
            // Notify background script to start tracking
            chrome.runtime.sendMessage({ 
                action: 'startTracking', 
                topic: topic,
                keywords: keywords 
            });
            
        } catch (error) {
            console.error('Error:', error);
            showStatus(error.message || 'Error generating keywords. Check your connection.', 'error');
        } finally {
            startButton.disabled = false;
        }
    });
    
    // Handle clear data button
    clearButton.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear all tracking data?')) {
            await chrome.storage.local.clear();
            await displayCurrentTracking();
            showStatus('All data cleared', 'success');
        }
    });
    
    // Check API health
    async function checkAPIHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            if (!response.ok) {
                showStatus('API connection issue. Check your settings.', 'error');
            }
        } catch (error) {
            showStatus('Cannot connect to API. Using local mode.', 'error');
        }
    }
    
    // Display current tracking topics
    async function displayCurrentTracking() {
        const storage = await chrome.storage.local.get(['learningTopics']);
        const topics = storage.learningTopics || {};
        
        topicsListDiv.innerHTML = '';
        
        if (Object.keys(topics).length === 0) {
            trackingInfoDiv.innerHTML = 'No topics being tracked. Add a topic to start learning!';
            trackingInfoDiv.className = 'tracking-info';
            clearButton.style.display = 'none';
            return;
        }
        
        clearButton.style.display = 'block';
        trackingInfoDiv.innerHTML = '<span class="tracking-active">✓ Actively tracking keywords on all pages</span>';
        trackingInfoDiv.className = 'tracking-info';
        
        // Display each topic
        Object.entries(topics).forEach(([topicName, topicData]) => {
            const topicDiv = document.createElement('div');
            topicDiv.className = 'topic-item';
            
            const totalCount = Object.values(topicData.keywordCounts).reduce((sum, count) => sum + count, 0);
            
            topicDiv.innerHTML = `
                <span class="topic-name">${topicName}</span>
                <span class="keyword-count">${totalCount} matches found</span>
            `;
            
            topicsListDiv.appendChild(topicDiv);
        });
    }
    
    // Show status message
    function showStatus(message, type) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
        
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
    }
});