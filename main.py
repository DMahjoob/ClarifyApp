import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np

# Import context, can replace with any class context
# from cs356_context import SYSTEM_PROMPT

# ========== Setup ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Groq client initialized")
else:
    groq_client = None
    print("GROQ_API_KEY not found")

questions = []
clients: list[WebSocket] = []

class Question(BaseModel):
    text: str
    user: str

# ========== API Endpoints ==========
# Render needs to check that the connection is valid

# Precompute slide embeddings
@app.on_event("startup")
async def prepare_slide_embeddings():
    df = pd.read_json("data/CS356_data.jsonl", lines=True)
    app.state.slides_df = df

    EMBEDDING_CACHE = "slide_embeddings.npy"

    # If cache exists -> load it
    if os.path.exists(EMBEDDING_CACHE):
        print("⚡ Loading cached slide embeddings...")
        app.state.slide_embeddings = np.load(EMBEDDING_CACHE)
        print("Embeddings loaded from cache")
        return

    # Otherwise generate once
    print("Generating slide embeddings (first run only)...")
    slide_texts = (
        df["title"] + " " +
        df["summary"] + " " +
        df["main_text"] + " " +
        df["keywords"].apply(lambda kws: " ".join(kws) if isinstance(kws, list) else "") + " " +
        df["deck_name"] + " " +
        df["slide_number"].astype(str) + " "
    ).tolist()

    embeddings = model.encode(slide_texts, normalize_embeddings=True)

    # Save embeddings to disk
    np.save(EMBEDDING_CACHE, embeddings)
    app.state.slide_embeddings = embeddings
    print("✅ Embeddings generated and cached to slide_embeddings.npy")

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "connected_clients": len(clients)}

# Asking questions 
@app.post("/api/ask")
async def ask_question(q: Question):
    print(f"New question from {q.user}: {q.text}")
    questions.append(q.dict())

    # Broadcast question for professor view
    disconnected = []
    for client in clients:
        try:
            await client.send_json({"event": "new_question", "data": q.dict()})
        except Exception:
            disconnected.append(client)

    # handling user connectinos
    for client in disconnected:
        if client in clients:
            clients.remove(client)

    print(f"📊 Total: {len(questions)} questions, {len(clients)} clients")

    # Generate recommendation ONLY for this user
    recommendation = recommend_slide(q.text)

    return {
        "status": "received",
        "slide_recommendation": recommendation
    }


# Text Cleaning Helper Function
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# 
# Recommend slides based on the query using simple keyword matching
def recommend_slide(query: str):
    # Clean and preprocess input using Regex
    # Match keywords to slides in the datafram
    df = app.state.slides_df
    slide_embeddings = app.state.slide_embeddings

    # Clean question
    processed_query = clean_text(query)

    # Encode question
    query_embedding = model.encode([processed_query], normalize_embeddings=True)[0]

    # Cosine similarity
    similarities = np.dot(slide_embeddings, query_embedding)

    # Get top 3 indices
    top_indices = np.argsort(similarities)[-3:][::-1]

    recommendations = []

    for idx in top_indices:
        slide = df.iloc[idx]
        recommendations.append({
            "deck_name": slide["deck_name"],
            "slide_number": int(slide["slide_number"]),
            "title": slide["title"],
            "score": float(similarities[idx]),
            "summary": slide.get("summary", ""),
            "keywords": slide.get("keywords", [])
        })

    return recommendations


# Websockets for real time data transfer to professor.html
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    print(f"🔌 Client connected. Total: {len(clients)}")
    
    try:
        # Send existing questions
        for q in questions:
            await ws.send_json({"event": "new_question", "data": q})
        
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ws in clients:
            clients.remove(ws)

# ========== Summarization ==========
async def summarize_questions():
    """Summarize questions with the imported context"""
    if not groq_client or not questions:
        return None

    # Build question list
    question_text = ""
    for q in questions[-5:]:
        question_text += f"- [{q['user']}] {q['text']}\n"

    try:        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", # can be replaced with any model, llama instant is fast
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, # imported from the context file
                {"role": "user", "content": f"Summarize these student questions:\n\n{question_text}"}
            ],
            max_tokens=600,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        print(f"Summary generated")
        return summary
        
    except Exception as e:
        print(f"Groq error: {e}")
        return None

# ========== Quiz Generation ==========
async def generate_quiz_from_summary(summary: str):
    """
    Generate quiz-style questions (MCQ, True/False, Fill-in-the-blank)
    based on the summary text.
    """
    if not groq_client or not summary:
        return None
# variety of questions provided: mcq, true_false, fill_blank
    quiz_prompt = f"""
You are a helpful and clear teaching assistant.

From the summarized content below, generate a short quiz with:
- 3 Multiple Choice Questions (each with 4 options A-D, and clearly mark the correct answer)
- 1 True/False question (mark correct answer)
- 1 Fill-in-the-blank question (provide the answer)

Keep questions directly grounded in the passage.
Return JSON ONLY in this format:

{{
  "mcq": [
    {{
      "question": "...?",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "B"
    }}
  ],
  "true_false": [
    {{
      "question": "...?",
      "answer": true
    }}
  ],
  "fill_blank": [
    {{
      "question": "_____ is ...",
      "answer": "..."
    }}
  ]
}}

Summary:
\"\"\"{summary}\"\"\"
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You generate quizzes only from provided content."},
                {"role": "user", "content": quiz_prompt}
            ],
            max_tokens=800,
            temperature=0.4 # slight creativity for quiz variety
        )

        quiz_json = response.choices[0].message.content
        print("Quiz generated")
        return quiz_json

    except Exception as e:
        print(f"Groq quiz generation error: {e}")
        return None


@app.on_event("startup")
async def start_summarizer():
    """Background summarization loop"""
    last_summarized_count = 0
    
    async def loop():
        nonlocal last_summarized_count
        print("🚀 Summarizer started (30s intervals)")
        while True:
            await asyncio.sleep(30)

            # Check if there have been more than 3 questinons and also if the number of current 
            # questions is less than we summarized
            if len(questions) > last_summarized_count and len(questions) >= 3:
                summary = await summarize_questions()
                if summary:
                    disconnected = []
                    for client in clients:
                        try:
                            await client.send_json({"event": "summary", "data": summary})
                        except Exception:
                            disconnected.append(client)
                    
                    # Generate quiz from summary
                    quiz = await generate_quiz_from_summary(summary)
                    if quiz:
                        for client in clients:
                            try:
                                await client.send_json({"event": "quiz", "data": quiz})
                            except Exception:
                                disconnected.append(client)
                    
                    for client in disconnected:
                        if client in clients:
                            clients.remove(client)
                    
                    last_summarized_count = len(questions)
                    print(f"Summary sent. Tracking {last_summarized_count} questions.")
    
    asyncio.create_task(loop())

# ========== Static HTML Pages ==========
@app.get("/")
async def serve_root():
    """Serve student page at root"""
    return FileResponse("static/index.html")

@app.get("/student.html")
async def serve_student():
    """Serve student page"""
    return FileResponse("static/index.html")

@app.get("/professor.html")
async def serve_professor():
    """Serve professor dashboard"""
    return FileResponse("static/professor.html")

# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
