from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from .vectorstore import get_retriever

def get_llm():
    # You can swap to gemini-1.5-pro if allowed
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
    )


def get_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an assistant for our course project. "
            "Use ONLY the provided context. "
            "If you are not sure, say you are not sure."
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "User question: {question}\n\n"
            "Relevant context:\n{context}\n\n"
            "Answer clearly and concisely."
        )
    ])

def build_rag_chain():
    llm = get_llm()
    prompt = get_prompt()
    chain = prompt | llm | StrOutputParser()
    return chain


def retrieve_context(query: str):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join(d.page_content for d in docs)
    return context_text, docs

def convert_history_to_lc(history):
    """
    history: list of {"role": "user"|"assistant", "content": "..."}
    -> list of LC message objects
    """
    lc_messages = []
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages