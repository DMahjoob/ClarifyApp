# Clarify App - Live Classroom Q&A System

An intelligent real-time classroom question-and-answer system that helps professors monitor student questions, receive AI-powered summaries, and generate quizzes based on course material.

## Overview

This system provides two interfaces:
- **Student Interface** (`index.html`): Students can ask questions, receive slide recommendations, and get AI-generated answers based on course material
- **Professor Dashboard** (`professor.html`): Real-time view of incoming questions with periodic AI summaries and automatically generated quizzes

## Features

### For Students
- **Voice Input**: Speech-to-text for asking questions
- **Smart Slide Recommendations**: RAG-powered system finds relevant slides from course material
- **Intelligent TA Responses**: LLM-generated answers grounded in actual course content
- **Quiz Generation**: Generate practice quizzes on demand with difficulty selection

### For Professors
- **Live Question Feed**: Real-time display of student questions via WebSockets
- **AI Summaries**: Automatic summarization of questions every 30 seconds (when 3+ new questions)
- **Auto-Generated Quizzes**: Quizzes created from summaries (MCQ, True/False, Fill-in-the-blank)
- **Topic Categorization**: Questions grouped by course topics with urgency flagging

## Tech Stack

- **Backend**: FastAPI (Python)
- **LLM**: Groq API (llama-3.1-8b-instant)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Real-time**: WebSockets
- **Frontend**: Vanilla HTML/CSS/JavaScript with Marked.js for markdown rendering
- **NLP**: NLTK for text preprocessing

## Installation

### Prerequisites
- Python 3.8+
- Groq API key

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
PORT=8000  # Optional, defaults to 8000
```

4. **Prepare course data**
The system expects course material in JSONL format in the `data/` directory. See the data format section below.

## Usage

### Running the Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Accessing the Interfaces

- **Student Interface**: `http://localhost:8000/` or `http://localhost:8000/student.html`
- **Professor Dashboard**: `http://localhost:8000/professor.html`

## Data Format

Course material should be in JSONL format (one JSON object per line) with the following structure:

```json
{
  "deck_name": "CS356_Unit03_Floats",
  "slide_number": 1,
  "chunk_index": 0,
  "title": "Unit 3 — Floating Point",
  "summary": "Brief summary of slide content",
  "main_text": "Full text content of the slide",
  "notes_text": "Additional notes or context",
  "keywords": ["keyword1", "keyword2"],
  "images": [],
  "layout": {"num_text_boxes": 1, "num_images": 0, "dominant_visual_type": "text-heavy"},
  "metadata": {
    "course": "CS356",
    "unit": 3,
    "topic": "Overview / IEEE 754",
    "importance_score": 6
  }
}
```

Place your JSONL files in the `data/` directory and update the main course data file path in `main.py`.

## Course Context Configuration

To customize for your course, modify the context file (e.g., `cs356_context.py`):

```python
SYSTEM_PROMPT = """You are a TA for [Your Course Name]. 
Summarize student questions by:

1. **Categorize by topic**: [List your course topics]
2. **Identify patterns**: Group similar questions
3. **Flag urgency**: Mark keywords like "crash", "deadline", "exam"
4. **Be concise**: Use bullet points, technical terms

**Topics & Keywords:**
- [Topic 1]: keyword1, keyword2, ...
- [Topic 2]: keyword3, keyword4, ...
...
"""
```

## API Endpoints

### Student Endpoints

**POST `/api/ask`**
Submit a question and receive slide recommendations + AI answer
```json
{
  "user": "student_name",
  "text": "question text"
}
```

**POST `/api/generate-quiz`**
Generate a quiz based on a topic
```json
{
  "user": "student_name",
  "text": "topic or question",
  "difficulty": "easy|medium|hard"
}
```

### Professor Endpoints

**WebSocket `/ws`**
Real-time connection for receiving:
- New questions (`event: "new_question"`)
- AI summaries (`event: "summary"`)
- Generated quizzes (`event: "quiz"`)

**GET `/health`**
Health check endpoint

## Key Features Explained

### RAG-Powered Slide Recommendations

The system uses semantic search to find relevant slides:
1. Preprocesses student questions (removes stopwords, normalizes text)
2. Embeds question using SentenceTransformer
3. Computes cosine similarity with pre-embedded slide content
4. Returns top 5 most relevant slides with scores

### AI-Generated Answers

Responses are grounded in course material:
1. Retrieves relevant slides using RAG
2. Constructs prompt with slide context
3. LLM generates answer using only provided material
4. Refuses off-topic questions

### Automatic Summarization

Every 30 seconds (when 3+ new questions):
1. Collects recent questions
2. Categorizes by topic using course-specific context
3. Identifies urgent questions (keywords: "segfault", "crash", "deadline")
4. Broadcasts summary to all connected professors

### Quiz Generation

Two modes:
1. **From summaries** (automatic): Generates quiz after each summary
2. **On-demand** (student-triggered): Custom quiz based on specific topic/question

## Performance Optimizations

- **Embedding Cache**: Slide embeddings are pre-computed and cached to `slide_embeddings.npy`
- **WebSocket Cleanup**: Automatic disconnection handling
- **Text Preprocessing**: Efficient stopword removal and text normalization
- **Rate Limiting**: Built-in support for rate limiting web fetch requests

## Customization

### Changing the Course

1. Update `SYSTEM_PROMPT` in your context file (e.g., `cs356_context.py`)
2. Replace course data JSONL files in `data/` directory
3. Import your context in `main.py`:
   ```python
   from your_course_context import SYSTEM_PROMPT
   ```
4. Delete `slide_embeddings.npy` to regenerate embeddings

### Adjusting Summarization

Modify timing and threshold in `main.py`:
```python
async def start_summarizer():
    async def loop():
        await asyncio.sleep(30)  # Change interval (seconds)
        if len(questions) >= 3:  # Change threshold
            # ...
```

### Styling

- Student interface: Modify styles in `static/index.html`
- Professor dashboard: Modify styles in `static/professor.html`

### Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## Troubleshooting

**No slide embeddings found**
- On first run, the system generates embeddings (may take 1-2 minutes)
- Check console for "✅ Embeddings generated and cached"

**WebSocket connection failed**
- Ensure server is running
- Check browser console for connection errors
- Verify firewall/proxy settings

**Voice input not working**
- Only works in HTTPS or localhost
- Check browser permissions for microphone access
- Supported in Chrome, Edge, Safari (not Firefox)

**LLM responses are slow**
- Groq API has rate limits
- Check your API key quota
- Consider caching common queries
