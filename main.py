import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# Import CS356 context
from cs356_context import SYSTEM_PROMPT

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
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "connected_clients": len(clients)}

@app.post("/api/ask")
async def ask_question(q: Question):
    print(f"New question from {q.user}: {q.text}")
    questions.append(q.dict())
    
    disconnected = []
    for client in clients:
        try:
            await client.send_json({"event": "new_question", "data": q.dict()})
        except Exception as e:
            print(f"Error sending to client: {e}")
            disconnected.append(client)
    
    for client in disconnected:
        if client in clients:
            clients.remove(client)
    
    print(f"📊 Total: {len(questions)} questions, {len(clients)} clients")
    return {"status": "received"}

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
    """Summarize questions with CS356 context"""
    if not groq_client or not questions:
        return None

    # Build question list
    question_text = ""
    for q in questions[-5:]:
        question_text += f"- [{q['user']}] {q['text']}\n"

    try:        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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

@app.on_event("startup")
async def start_summarizer():
    """Background summarization loop"""
    last_summarized_count = 0
    
    async def loop():
        nonlocal last_summarized_count
        print("🚀 Summarizer started (30s intervals)")
        while True:
            await asyncio.sleep(30)
            
            if len(questions) > last_summarized_count and len(questions) >= 3:
                summary = await summarize_questions()
                if summary:
                    disconnected = []
                    for client in clients:
                        try:
                            await client.send_json({"event": "summary", "data": summary})
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
