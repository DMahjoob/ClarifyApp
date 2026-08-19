import os
import sys

# Python 3.13 + NLTK: NLTK's inisec.py blocks importing regex from CWD.
# Temporarily remove CWD from sys.path so regex loads from site-packages.
_cwd_entries = [p for p in sys.path if p in ("", os.getcwd())]
for p in _cwd_entries:
    sys.path.remove(p)

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import re
# import nltk
# from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import psycopg2
from psycopg2.extras import RealDictCursor

# Restore CWD to sys.path so local module imports work
for p in _cwd_entries:
    sys.path.insert(0, p)

# Import context for each supported class
from cs356_context import SYSTEM_PROMPT as CS356_PROMPT
from cs102_context import SYSTEM_PROMPT as CS102_PROMPT
from cs103_context import SYSTEM_PROMPT as CS103_PROMPT

# Class registry: maps class_id -> display name, data file, system prompt
CLASS_REGISTRY = {
    "cs356": {
        "name": "CS356 - Computer Systems",
        "data_file": "data/CS356_data.jsonl",
        "system_prompt": CS356_PROMPT,
        "embedding_cache": "slide_embeddings_cs356.npy",
    },
    "cs102": {
        "name": "CS102 - Intro to Programming",
        "data_file": "data/CS102_data.jsonl",
        "system_prompt": CS102_PROMPT,
        "embedding_cache": "slide_embeddings_cs102.npy",
    },
    "cs103": {
        "name": "CS103 - Intro to Programming in C++",
        "data_file": "data/CS103_data.jsonl",
        "system_prompt": CS103_PROMPT,
        "embedding_cache": "slide_embeddings_cs103.npy",
    },
}

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

# Connect to Supabase
def get_db():
    return psycopg2.connect(os.getenv("DB_URL"))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

PROF_PASSWORD = os.getenv("PROF_PASSWORD")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer model loaded.")

# nltk.download('stopwords')
# STOPWORDS = set(stopwords.words('english'))
STOPWORDS = {"i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"}

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
    user: str = ""
    class_id: str = "cs356"

class QuizRequest(BaseModel):
    user: str = ""
    text: str
    difficulty: str  # "easy", "medium", "hard"
    question_types: list[str] = ["mcq", "true_false", "short_answer"]
    class_id: str = "cs356"

class QuizResultSubmission(BaseModel):
    class_id: str
    topic: str
    question_text: str
    selected_answer: str
    correct_answer: str
    is_correct: bool


# ========== API Endpoints ==========
# Render needs to check that the connection is valid

# Precompute slide embeddings for all registered classes
@app.on_event("startup")
async def ensure_quiz_results_table():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS "QuizResults" (
                        id SERIAL PRIMARY KEY,
                        class_id VARCHAR(20) NOT NULL,
                        topic VARCHAR(500) NOT NULL,
                        question_text TEXT NOT NULL,
                        selected_answer TEXT NOT NULL,
                        correct_answer TEXT NOT NULL,
                        is_correct BOOLEAN NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
        print("QuizResults table ready")
    except Exception as e:
        print(f"Could not create QuizResults table: {e}")

@app.on_event("startup")
async def prepare_slide_embeddings():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_all_embeddings_sync)

def load_embeddings_for_class(class_id: str, config: dict):
    data_file = config["data_file"]
    cache_file = config["embedding_cache"]

    if not os.path.exists(data_file):
        print(f"Data file not found for {class_id}: {data_file}, skipping.")
        return None, None

    df = pd.read_json(data_file, lines=True)

    if os.path.exists(cache_file):
        print(f"Loading cached embeddings for {class_id}...")
        embeddings = np.load(cache_file)
        return df, embeddings

    print(f"Generating embeddings for {class_id}...")
    slide_texts = (
        df["title"] + " " + df["summary"] + " " +
        df["summary"] + " " + df["summary"] + " " +
        df["main_text"] + " " +
        df["keywords"].apply(lambda kws: " ".join(kws) if isinstance(kws, list) else "") + " " +
        df["deck_name"] + " " +
        df["slide_number"].astype(str)
    ).tolist()

    embeddings = embedding_model.encode(slide_texts, normalize_embeddings=True)
    np.save(cache_file, embeddings)
    return df, embeddings

def load_all_embeddings_sync():
    app.state.class_data = {}
    for class_id, config in CLASS_REGISTRY.items():
        df, embeddings = load_embeddings_for_class(class_id, config)
        if df is not None:
            app.state.class_data[class_id] = {
                "slides_df": df,
                "slide_embeddings": embeddings,
            }
            print(f"Embeddings ready for {class_id} ({len(df)} slides).")
        else:
            print(f"Skipped {class_id} (no data).")
    print(f"Loaded {len(app.state.class_data)} class(es).")

# List available classes
@app.get("/api/classes")
async def list_classes():
    return [
        {"id": cid, "name": cfg["name"]}
        for cid, cfg in CLASS_REGISTRY.items()
        if cid in app.state.class_data
    ]

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "connected_clients": len(clients)}

# Asking questions 
@app.post("/api/ask")
async def ask_question(q: Question):
    print(f"New question from {q.user}: {q.text}")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO "QuestionBank" (username, text, class_id) VALUES (%s, %s, %s)',
                (q.user, q.text, q.class_id)
            )

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

    if q.class_id not in app.state.class_data:
        return {"status": "error", "detail": f"Class '{q.class_id}' not found"}

    # Generate recommendation & summary only for this user
    # Run in executor to avoid blocking the event loop (which blocks the summarizer + websockets)
    loop = asyncio.get_event_loop()
    recommendation, response = await loop.run_in_executor(
        None, recommend_slide_and_answer, q.text, q.class_id
    )

    return {
        "status": "received",
        "slide_recommendation": recommendation,
        "rag_response": response
    }

# Awaiting user to request a quiz
@app.post("/api/generate-quiz")
async def generate_quiz(req: QuizRequest):
    try:
        print(f"Quiz request from {req.user} | Difficulty: {req.difficulty}")

        if req.difficulty not in {"easy", "medium", "hard"}:
            return {
                "status": "error",
                "detail": "Invalid difficulty"
            }

        if req.class_id not in app.state.class_data:
            return {"status": "error", "detail": f"Class '{req.class_id}' not found"}

        loop = asyncio.get_event_loop()
        raw_quiz = await loop.run_in_executor(
            None, generate_quiz_from_question, req.text, req.difficulty, req.question_types, req.class_id
        )

        try:
            quiz_json = json.loads(raw_quiz)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "detail": "Failed to parse quiz JSON from LLM"
            }

        # Flatten quiz into frontend-friendly format
        questions = []

        # MCQs
        for q in quiz_json.get("mcq", []):
            options_dict = q["options"]
            questions.append({
                "question": q["question"],
                "options": list(options_dict.values()),
                "answer": q["answer"]
            })

        # True / False
        tf = quiz_json.get("true_false")
        if tf:
            questions.append({
                "question": tf["question"],
                "options": ["True", "False"],
                "answer": str(tf["answer"])
            })

        # Short Answer
        sa = quiz_json.get("short_answer")
        if sa:
            questions.append({
                "question": sa["question"],
                "answer": sa["answer"]
            })

        return {
            "status": "success",
            "questions": questions,
            "topic": req.text,
            "class_id": req.class_id
        }

    except Exception as e:
        print(f"❌ Quiz generation error: {e}")
        return {
            "status": "error",
            "detail": "Internal quiz generation error"
        }




@app.post("/api/quiz-result")
async def submit_quiz_result(result: QuizResultSubmission):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO "QuizResults"
                       (class_id, topic, question_text, selected_answer, correct_answer, is_correct)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (result.class_id, result.topic, result.question_text,
                     result.selected_answer, result.correct_answer, result.is_correct)
                )
        return {"status": "recorded"}
    except Exception as e:
        print(f"Quiz result save error: {e}")
        return {"status": "error", "detail": "Could not save quiz result"}


@app.get("/api/quiz-stats")
async def quiz_stats(class_id: str = None):
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if class_id:
                    cur.execute("""
                        SELECT class_id, topic,
                               COUNT(*) AS total,
                               SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct,
                               ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*)) AS pct
                        FROM "QuizResults"
                        WHERE class_id = %s AND created_at >= NOW() - INTERVAL '7 days'
                        GROUP BY class_id, topic
                        ORDER BY topic
                    """, (class_id,))
                else:
                    cur.execute("""
                        SELECT class_id, topic,
                               COUNT(*) AS total,
                               SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct,
                               ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*)) AS pct
                        FROM "QuizResults"
                        WHERE created_at >= NOW() - INTERVAL '7 days'
                        GROUP BY class_id, topic
                        ORDER BY class_id, topic
                    """)
                rows = cur.fetchall()
        return {"stats": rows}
    except Exception as e:
        print(f"Quiz stats error: {e}")
        return {"stats": []}


# Text Cleaning Helper Function
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


import numpy as np

# Recommend slide and produce answer
def recommend_slide_and_answer(query: str, class_id: str):
    """
    RAG-powered TA assistant.
    - Retrieves relevant slides for the given class
    - Answers question using class-specific context
    """

    class_data = app.state.class_data[class_id]
    df = class_data["slides_df"]
    slide_embeddings = class_data["slide_embeddings"]
    system_prompt = CLASS_REGISTRY[class_id]["system_prompt"]

    # ========== Preprocess Question ==========
    processed_query = clean_text(query)

    # Embed student question
    query_embedding = embedding_model.encode(
        [processed_query],
        normalize_embeddings=True
    )[0]

    # ========== Cosine Similarity ==========
    similarities = np.dot(slide_embeddings, query_embedding)

    # Get top 3 relevant slides
    top_indices = np.argsort(similarities)[-5:][::-1]

    recommendations = []
    retrieved_slides_str = ""


    for i in top_indices:
        slide = df.iloc[i]
        rec = {
            "deck_name": slide["deck_name"],
            "slide_number": int(slide["slide_number"]),
            "title": slide["title"],
            "score": float(similarities[i]),
            "summary": slide.get("summary", ""),
            "keywords": slide.get("keywords", [])
        }
        recommendations.append(rec)

        retrieved_slides_str += f"""
            Deck: {rec['deck_name']}
            Slide: {rec['slide_number']}
            Title: {rec['title']}
            Summary: {rec['summary']}
            Keywords: {rec['keywords']}
        """

    # ========== RAG Prompt ==========
    prompt = f"""
        You are a helpful Teaching Assistant. Your #1 priority is to answer the student's question. Always provide a clear, complete answer. Never refuse to answer or say the slides don't cover the topic.

        RULE 1 — Always Answer
        - Above all else, answer the question. Do NOT say "the slides do not cover this" or "this topic is not available." Just answer it.
        - Use the provided slides when relevant. For slide-sourced claims, reference the specific slide (deck name and slide number).
        - When the slides don't fully cover the topic, seamlessly use your general knowledge to give a complete answer. Do not call attention to the gap — just answer naturally.

        RULE 2 — Accuracy
        - Be factual and accurate. Do not speculate or fabricate information.

        RULE 3 — Debugging Emphasis
        - If the question involves crashes, segfaults, stack smashing, or debugging, include concrete debugging steps.

        COURSE SLIDES:
        {retrieved_slides_str}

        Student question:
        {query}
    """

    # ========== Groq LLM Call ==========
    try:
        response_rag = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3,
        )

        answer = response_rag.choices[0].message.content.strip()
        print(f"RAG answer generated for {class_id}")

    except Exception as e:
        print(f"❌ Groq error: {e}")
        return None

    # ========== Return ==========
    return recommendations, answer

# Generate quiz from user input
def generate_quiz_from_question(query: str, difficulty: str, question_types: list[str] = None, class_id: str = "cs356"):
    """
    Generate a quiz from a single student question/topic.
    Difficulty controls depth and subtlety.
    """
    if question_types is None:
        question_types = ["mcq", "true_false", "short_answer"]

    class_data = app.state.class_data[class_id]
    df = class_data["slides_df"]
    slide_embeddings = class_data["slide_embeddings"]
    system_prompt = CLASS_REGISTRY[class_id]["system_prompt"]

    processed_query = clean_text(query)
    query_embedding = embedding_model.encode([processed_query], normalize_embeddings=True)[0]

    similarities = np.dot(slide_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-5:][::-1]

    retrieved_slides = ""
    for i in top_indices:
        slide = df.iloc[i]
        retrieved_slides += f"""
        Deck: {slide['deck_name']}
        Slide: {slide['slide_number']}
        Title: {slide['title']}
        Summary: {slide.get('summary', '')}
        Keywords: {slide.get('keywords', [])}
        """

    difficulty_guidance = {
        "easy": "Focus on definitions, direct recall, and surface-level understanding.",
        "medium": "Include conceptual understanding and light application.",
        "hard": "Include tricky edge cases, reasoning, pitfalls, or debugging-style questions."
    }

    # Build generation instructions and JSON format based on selected types
    generate_lines = []
    json_parts = []
    json_parts.append(f'"difficulty": "{difficulty}"')

    if "mcq" in question_types:
        generate_lines.append("- 4 Multiple Choice Questions (A–D, mark correct answer)")
        json_parts.append('"mcq": [{"question": "...?", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "A"}]')
    if "true_false" in question_types:
        generate_lines.append("- 1 True/False question")
        json_parts.append('"true_false": {"question": "...?", "answer": true}')
    if "short_answer" in question_types:
        generate_lines.append("- 1 Short-answer question")
        json_parts.append('"short_answer": {"question": "...?", "answer": "..."}')

    generate_section = "\n    ".join(generate_lines)
    json_format = "{\n    " + ",\n    ".join(json_parts) + "\n    }"

    class_name = CLASS_REGISTRY[class_id]["name"]
    quiz_prompt = f"""
    You are a {class_name} teaching assistant.

    Generate a quiz based ONLY on the provided slide content.

    Difficulty: {difficulty.upper()}
    Guidance: {difficulty_guidance.get(difficulty, "")}

    Generate:
    {generate_section}

    Return JSON ONLY in this exact format:

    {json_format}

    Slides:
    {retrieved_slides}

    Topic:
    {query}
    """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": quiz_prompt}
        ],
        max_tokens=900,
        temperature=0.4
    )

    return response.choices[0].message.content

# Websockets for real time data transfer to professor.html
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    print(f"🔌 Client connected. Total: {len(clients)}")
    
    try:
        # Send existing questions
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT username, text, class_id FROM "QuestionBank" ORDER BY timestamp ASC')
                past = cur.fetchall()
        for q in past:
            await ws.send_json({"event": "new_question", "data": {"user": q["username"], "text": q["text"], "class_id": q.get("class_id", "")}})
        
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
    if not groq_client:
        return None

    # Build question list
    question_text = ""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT username, text FROM "QuestionBank" ORDER BY timestamp DESC LIMIT 5')
            recent = cur.fetchall()

    question_text = ""
    for q in recent:
        question_text += f"- [{q['username']}] {q['text']}\n"


    # Use the first available class prompt as default for summarization
    default_prompt = next(iter(CLASS_REGISTRY.values()))["system_prompt"]

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": default_prompt},
                {"role": "user", "content": f"Summarize these student questions:\n\n{question_text}"}
            ],
            max_tokens=600,
            temperature=0.3,
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
- 5 Multiple Choice Questions (each with 4 options A-D, and clearly mark the correct answer)
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
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You generate quizzes only from provided content."},
                {"role": "user", "content": quiz_prompt}
            ],
            max_tokens=800,
            temperature=0.4,
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

            # Check if there have been more than 3 questions and also if the number of current 
            # questions is less than we summarized
            try:
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute('SELECT COUNT(*) FROM "QuestionBank"')
                        total = cur.fetchone()[0]
            except Exception as e:
                print(f"DB error in summarizer: {e}")
                continue

            if total > last_summarized_count and total >= 3:
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
                    
                    last_summarized_count = total
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
async def serve_professor(request: Request):
    token = request.cookies.get("prof_auth")
    if token != PROF_PASSWORD:
        return RedirectResponse(url="/login")
    return FileResponse("static/professor.html")

@app.get("/login")
async def login_page():
    return HTMLResponse("""
        <html><body style="font-family:sans-serif;display:flex;justify-content:center;margin-top:100px">
        <form method="post" action="/login">
            <h2>Professor Login</h2>
            <input type="password" name="password" placeholder="Enter password" style="padding:8px;margin-right:8px"/>
            <button type="submit" style="padding:8px">Login</button>
        </form></body></html>
    """)

@app.post("/login")
async def do_login(request: Request):
    form = await request.form()
    if form.get("password") == PROF_PASSWORD:
        response = RedirectResponse(url="/professor.html", status_code=303)
        response.set_cookie("prof_auth", PROF_PASSWORD, httponly=True)
        return response
    return HTMLResponse("""
        <html><body style="font-family:sans-serif;display:flex;justify-content:center;margin-top:100px">
        <form method="post" action="/login">
            <h2>Professor Login</h2>
            <input type="password" name="password" placeholder="Enter password" style="padding:8px;margin-right:8px"/>
            <button type="submit" style="padding:8px">Login</button>
            <p style="color:red">Incorrect password</p>
        </form></body></html>
    """)

# Favicon
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
