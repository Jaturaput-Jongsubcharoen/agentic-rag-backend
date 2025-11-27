from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.chat import ChatRequest, ChatResponse, handle_chat

app = FastAPI(title="Agentic RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(  
    request: ChatRequest):
    """
    POST /api/chat
    {
      "message": "...",
      "history": [ { "role": "...", "content": "..." }, ... ]
    }
    """
    response = handle_chat(request)
    return response
