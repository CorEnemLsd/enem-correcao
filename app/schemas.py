"""Pydantic schemas for ENEM essay correction API."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class CompetenciaEnum(str, Enum):
    COMPETENCIA_1 = "competencia_1"
    COMPETENCIA_2 = "competencia_2"
    COMPETENCIA_3 = "competencia_3"
    COMPETENCIA_4 = "competencia_4"
    COMPETENCIA_5 = "competencia_5"


class CompetenciaScore(BaseModel):
    """Score for a single ENEM competency."""
    competencia: str = Field(..., description="Nome da competência")
    nota: int = Field(..., ge=0, le=200, description="Nota de 0 a 200")
    nivel: int = Field(..., ge=0, le=5, description="Nível de 0 a 5")
    justificativa: str = Field(..., description="Justificativa detalhada da nota")


class CorrectionResponse(BaseModel):
    """Full correction response."""
    competencias: List[CompetenciaScore]
    nota_total: int = Field(..., ge=0, le=1000)
    resumo_geral: str = Field(..., description="Resumo geral da correção")
    pontos_fortes: List[str] = Field(default_factory=list)
    pontos_melhoria: List[str] = Field(default_factory=list)
    proposta_intervencao_analise: Optional[str] = Field(None, description="Análise da competência 5")


class TranscribeRequest(BaseModel):
    """Request for image transcription."""
    image_base64: str = Field(..., description="Imagem em base64")
    use_groq_vision: bool = Field(default=True, description="Usar Groq Vision (True) ou Tesseract local (False)")


class TranscribeResponse(BaseModel):
    """Response for image transcription."""
    texto: str
    confidence: Optional[float] = None
    method: str


class CorrectRequest(BaseModel):
    """Request for essay correction."""
    texto: str = Field(..., min_length=50, description="Texto da redação")
    tema: Optional[str] = Field(None, description="Tema da redação (opcional)")
    textos_apoio: Optional[List[str]] = Field(None, description="Textos de apoio da prova")


class CorrectResponse(BaseModel):
    """API response for correction."""
    success: bool
    data: Optional[CorrectionResponse] = None
    error: Optional[str] = None
    processing_time_ms: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
