
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from src.retrieval_and_generation import ask_pdf

app = FastAPI(
    title="MedicineChatbot API",
    version="1.0.0"
)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    history: list[ChatMessage] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    context: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer, context = ask_pdf(
            question=request.question,
            top_k=request.top_k,
            history=[
                message.model_dump()
                for message in request.history
            ]
        )

        return ChatResponse(
            answer=answer,
            context=context
        )

    except Exception as e:
        print(f"RAG error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate response"
        )
