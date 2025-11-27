from typing import List, Dict, Any
from pydantic import BaseModel

from .agent import build_rag_chain, retrieve_context, convert_history_to_lc


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryMessage] = []


class ChatResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    history: List[HistoryMessage]
    meta: Dict[str, Any] = {}


def handle_chat(request: ChatRequest) -> ChatResponse:
    """
    Main RAG + agent logic.
    Used by FastAPI endpoint in app.py.
    """
    # 1) Retrieve
    context_text, docs = retrieve_context(request.message)

    # 2) Agent with memory
    chain = build_rag_chain()
    lc_history = convert_history_to_lc(
        [h.dict() for h in request.history]
    )
    answer = chain.invoke({
        "question": request.message,
        "context": context_text,
        "chat_history": lc_history,
    })

    # 3) Build new history
    new_history = request.history + [
        HistoryMessage(role="user", content=request.message),
        HistoryMessage(role="assistant", content=answer),
    ]

    context_payload = [
        {
            "content_preview": d.page_content[:300],
            "metadata": d.metadata,
        }
        for d in docs
    ]

    meta = {
        "used_memory": len(request.history) > 0,
        "num_history_messages": len(request.history),
        "num_context_chunks": len(docs),
    }

    return ChatResponse(
        answer=answer,
        context=context_payload,
        history=new_history,
        meta=meta,
    )
