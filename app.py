from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Enable CORS (so React can call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Agentic RAG backend is running!"}

@app.post("/api/chat")
def chat(payload: dict):
    return {
        "response": "Chat endpoint not implemented yet",
        "received": payload
    }
