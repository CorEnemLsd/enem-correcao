"""FastAPI application for ENEM Essay Correction API."""
import os
import time
import base64
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from app.schemas import (
    TranscribeRequest, TranscribeResponse,
    CorrectRequest, CorrectResponse,
    HealthResponse
)
from app.correction.groq_llm import get_groq_llm
from app.correction.rag_pipeline import get_rag_pipeline


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting ENEM Essay Correction API...")
    
    # Initialize RAG pipeline
    rag = get_rag_pipeline()
    stats = rag.get_stats()
    print(f"📚 RAG initialized: {stats['guidelines_count']} guidelines, {stats['fewshots_count']} few-shots")
    
    # Verify Groq connection
    try:
        llm = get_groq_llm()
        print(f"🤖 Groq LLM ready: {llm.text_model} (vision: {llm.vision_model})")
    except Exception as e:
        print(f"⚠️ Groq LLM warning: {e}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


app = FastAPI(
    title="ENEM Essay Correction API",
    description="API para correção automática de redações do ENEM usando IA (Groq + RAG)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    rag = get_rag_pipeline()
    stats = rag.get_stats()
    
    groq_ok = False
    try:
        get_groq_llm()
        groq_ok = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if groq_ok else "degraded",
        version="1.0.0",
        models_loaded={
            "rag_guidelines": stats['guidelines_count'] > 0,
            "rag_fewshots": stats['fewshots_count'] > 0,
            "groq_llm": groq_ok,
            "groq_vision": groq_ok
        }
    )


@app.post("/transcrever", response_model=TranscribeResponse)
async def transcrever_redacao(
    file: UploadFile = File(..., description="Imagem da redação (JPG, PNG, PDF)"),
    use_groq_vision: bool = Form(True, description="Usar Groq Vision para OCR")
):
    """
    Transcreve imagem de redação para texto usando OCR.
    Suporta: JPG, PNG, PDF (primeira página).
    """
    start_time = time.time()
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Tipo de arquivo não suportado. Use: {', '.join(allowed_types)}")
    
    # Read file
    content = await file.read()
    
    # Convert to base64
    if file.content_type == "application/pdf":
        return TranscribeResponse(
            texto="",
            confidence=0.0,
            method="pdf_not_supported_yet",
            error="PDF support requires pdf2image + poppler. Use JPG/PNG for now."
        )
    
    image_base64 = base64.b64encode(content).decode('utf-8')
    
    # OCR via Groq Vision
    llm = get_groq_llm()
    result = llm.transcribe_image(image_base64)
    
    if "error" in result and result["error"]:
        raise HTTPException(500, f"OCR failed: {result['error']}")
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return TranscribeResponse(
        texto=result["texto"],
        confidence=result.get("confidence", 0.95),
        method=result.get("method", "groq_vision")
    )


@app.post("/corrigir", response_model=CorrectResponse)
async def corrigir_redacao(request: CorrectRequest):
    """
    Corrige uma redação do ENEM retornando notas nas 5 competências + total.
    """
    start_time = time.time()
    
    try:
        llm = get_groq_llm()
        correction = llm.correct_essay(
            essay_text=request.texto,
            tema=request.tema,
            textos_apoio=request.textos_apoio
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return CorrectResponse(
            success=True,
            data=correction,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        return CorrectResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time
        )


@app.post("/corrigir-imagem", response_model=CorrectResponse)
async def corrigir_redacao_imagem(
    file: UploadFile = File(..., description="Imagem da redação"),
    tema: Optional[str] = Form(None, description="Tema da redação"),
    textos_apoio: Optional[str] = Form(None, description="Textos de apoio (JSON array)")
):
    """
    Pipeline completo: Transcreve imagem + Corrige redação em uma chamada.
    """
    start_time = time.time()
    
    try:
        # Step 1: Transcribe
        content = await file.read()
        image_base64 = base64.b64encode(content).decode('utf-8')
        
        llm = get_groq_llm()
        ocr_result = llm.transcribe_image(image_base64)
        
        if "error" in ocr_result and ocr_result["error"]:
            return CorrectResponse(
                success=False,
                error=f"OCR failed: {ocr_result['error']}",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        essay_text = ocr_result["texto"]
        
        if not essay_text or len(essay_text.strip()) < 50:
            return CorrectResponse(
                success=False,
                error="Texto transcrito muito curto ou vazio. Verifique a imagem.",
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Step 2: Correct
        textos_apoio_list = None
        if textos_apoio:
            import json
            try:
                textos_apoio_list = json.loads(textos_apoio)
            except:
                pass
        
        correction = llm.correct_essay(
            essay_text=essay_text,
            tema=tema,
            textos_apoio=textos_apoio_list
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return CorrectResponse(
            success=True,
            data=correction,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        return CorrectResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time
        )


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    rag = get_rag_pipeline()
    stats = rag.get_stats()
    
    # Get actual models from env
    text_model = os.getenv("GROQ_TEXT_MODEL", "qwen/qwen3.8-27b")
    fast_model = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    vision_model = os.getenv("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")
    
    return {
        "rag": stats,
        "models": {
            "text_model": text_model,
            "vision_model": vision_model,
            "fast_model": fast_model
        },
        "endpoints": [
            "GET /health",
            "POST /transcrever",
            "POST /corrigir",
            "POST /corrigir-imagem",
            "GET /stats"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
