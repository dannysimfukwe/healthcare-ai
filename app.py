from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

app = FastAPI(title="Healthcare AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    context: str = "general"

class ChatResponse(BaseModel):
    response: str
    language: str
    confidence: float

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

@app.get("/")
async def root():
    return {"status": "ok", "service": "Healthcare AI", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "rag_mode": os.getenv("RAG_MODE", "false")}

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(request: ChatRequest):
    ollama_response = await call_ollama(request.message, request.context)
    return ChatResponse(
        response=ollama_response,
        language=request.language,
        confidence=0.85
    )

@app.post("/clinical-notes")
async def clinical_notes(text: str):
    structured = await generate_clinical_notes(text)
    return {"notes": structured}

@app.post("/translate")
async def translate(text: str, from_lang: str = "en", to_lang: str = "ny"):
    translated = await translate_text(text, from_lang, to_lang)
    return {"translated": translated, "from": from_lang, "to": to_lang}

async def call_ollama(prompt: str, context: str) -> str:
    system_prompt = f"You are a healthcare assistant. Context: {context}. Provide accurate, helpful medical information."
    
    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json={
                    "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI service unavailable. Please ensure Ollama is running. Error: {str(e)}"
    
    return "AI service is not available. Please try again later."

async def generate_clinical_notes(text: str) -> dict:
    prompt = f"Convert this dictation into structured clinical notes: {text}"
    notes = await call_ollama(prompt, "clinical documentation")
    
    return {
        "chief_complaint": "Extracted from conversation",
        "history": notes,
        "assessment": "AI-generated summary",
        "plan": "Pending physician review"
    }

async def translate_text(text: str, from_lang: str, to_lang: str) -> str:
    lang_names = {"en": "English", "ny": "Chichewa", "tob": "Tumbuka"}
    prompt = f"Translate the following from {lang_names.get(from_lang, from_lang)} to {lang_names.get(to_lang, to_lang)}: {text}"
    return await call_ollama(prompt, "translation")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)