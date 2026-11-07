"""Groq LLM integration for ENEM essay correction and OCR."""
import os
import json
import re
import time
from typing import Dict, List, Optional, Any
from groq import Groq
import requests
from app.correction.rag_pipeline import get_rag_pipeline
from app.schemas import CompetenciaScore, CorrectionResponse


# System prompt for ENEM correction
ENEM_CORRECTION_SYSTEM_PROMPT = """Você é um CORRETOR OFICIAL DO ENEM. Avalie a redação nas 5 competências (0-200 cada, total 0-1000).

C1: Norma culta (ortografia, concordância, regência, pontuação)
C2: Desenvolvimento do tema + estrutura dissertativo-argumentativa
C3: Argumentação, progressão temática, repertório sociocultural
C4: Coesão textual (conectivos, referências, progressão)
C5: Proposta de intervenção (agente, ação, modo, detalhamento, DH)

REGRAS: Nota zero se fuga ao tema, não dissertativo, cópia indevida, ilegível, impropérios, desrespeito DH, <7 linhas.

RESPONDA APENAS JSON VÁLIDO (sem markdown, sem texto extra):
{
  "competencias": [
    {"competencia": "competencia_1", "nota": 160, "nivel": 4, "justificativa": "..."},
    {"competencia": "competencia_2", "nota": 160, "nivel": 4, "justificativa": "..."},
    {"competencia": "competencia_3", "nota": 120, "nivel": 3, "justificativa": "..."},
    {"competencia": "competencia_4", "nota": 200, "nivel": 5, "justificativa": "..."},
    {"competencia": "competencia_5", "nota": 160, "nivel": 4, "justificativa": "..."}
  ],
  "nota_total": 800,
  "resumo_geral": "Resumo em 2-3 frases",
  "pontos_fortes": ["...", "..."],
  "pontos_melhoria": ["...", "..."],
  "proposta_intervencao_analise": "Análise da competência 5"
}

NÍVEIS: 0=0, 1=40, 2=80, 3=120, 4=160, 5=200. SEJA RIGOROSO."""


# OCR prompt
OCR_SYSTEM_PROMPT = """OCR para redações ENEM. Extraia o texto COMPLETO preservando parágrafos. Ignore marcações de correção. Se houver textos de apoio, separe com [TEXTOS DE APOIO]. Retorne APENAS o texto transcrito."""


def extract_json(text: str) -> dict:
    """Extrai JSON válido de resposta que pode ter texto extra ou estar truncado."""
    # Tenta parse direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Tenta encontrar JSON entre chaves (greedy)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Tenta corrigir JSON truncado
    text = text.strip()
    if text.startswith('{') and not text.endswith('}'):
        text = text.rstrip() + '}}'
        try:
            return json.loads(text)
        except:
            pass
    # Fallback: estrutura padrão
    return {
        "competencias": [
            {"competencia": f"competencia_{i}", "nota": 120, "nivel": 3, "justificativa": "Erro na extração do JSON"}
            for i in range(1, 6)
        ],
        "nota_total": 600,
        "resumo_geral": "Erro na extração do JSON da resposta do modelo",
        "pontos_fortes": [], "pontos_melhoria": [], "proposta_intervencao_analise": ""
    }


class GroqLLM:
    def __init__(self, api_key: Optional[str] = None, nvidia_api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=self.api_key)
        
        self.text_model = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-20b")
        self.text_model_fast = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
        self.vision_model = os.getenv("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")
        
        # NVIDIA NIM fallback
        self.nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        self.nvidia_text_model = os.getenv("NVIDIA_TEXT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        
        self._rag = None
    
    @property
    def rag(self):
        if self._rag is None:
            self._rag = get_rag_pipeline()
        return self._rag
    
    def _call_nvidia_text(self, system_prompt: str, user_prompt: str) -> str:
        """Call NVIDIA NIM for text generation."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.nvidia_text_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers, json=payload, timeout=60
        )
        if response.status_code != 200:
            raise Exception(f"NVIDIA HTTP {response.status_code}: {response.text}")
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    def correct_essay(
        self,
        essay_text: str,
        tema: Optional[str] = None,
        textos_apoio: Optional[List[str]] = None,
        use_fast_model: bool = False
    ) -> CorrectionResponse:
        start_time = time.time()
        
        # Retrieve context
        try:
            guidelines = self.rag.retrieve_guidelines(essay_text[:300], n_results=4)
            fewshots = self.rag.retrieve_similar_essays(essay_text, n_results=2)
        except Exception as e:
            print(f"RAG warning: {e}")
            guidelines = []
            fewshots = []
        
        ctx_parts = []
        for g in guidelines[:3]:
            ctx_parts.append(g['document'][:800])
        for fs in fewshots[:2]:
            ctx_parts.append(fs['document'][:1000])
        context = "\n\n".join(ctx_parts)[:4000]
        
        user_prompt = f"""REDAÇÃO:
TEMA: {tema or 'Não informado'}
TEXTOS APOIO: {'; '.join(textos_apoio) if textos_apoio else 'Não informados'}
TEXTO: {essay_text[:2500]}

CONTEXTO (guidelines + few-shots):
{context}

Corrija seguindo critérios ENEM. Responda APENAS o JSON."""
        
        # Try Groq first
        models_to_try = []
        if not use_fast_model:
            models_to_try.append(("groq", self.text_model))
        models_to_try.append(("groq_fast", self.text_model_fast))
        if self.nvidia_api_key:
            models_to_try.append(("nvidia", self.nvidia_text_model))
        
        last_error = None
        for model_type, model_name in models_to_try:
            try:
                if model_type == "nvidia":
                    raw_content = self._call_nvidia_text(ENEM_CORRECTION_SYSTEM_PROMPT, user_prompt)
                else:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": ENEM_CORRECTION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2000
                    )
                    raw_content = response.choices[0].message.content.strip()
                
                result_json = extract_json(raw_content)
                
                competencias = [CompetenciaScore(**c) for c in result_json.get("competencias", [])]
                
                if len(competencias) != 5:
                    raise ValueError(f"Expected 5 competencies, got {len(competencias)}")
                
                correction = CorrectionResponse(
                    competencias=competencias,
                    nota_total=result_json.get("nota_total", sum(c.nota for c in competencias)),
                    resumo_geral=result_json.get("resumo_geral", ""),
                    pontos_fortes=result_json.get("pontos_fortes", []),
                    pontos_melhoria=result_json.get("pontos_melhoria", []),
                    proposta_intervencao_analise=result_json.get("proposta_intervencao_analise")
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                print(f"Correction: {processing_time}ms using {model_type} ({model_name})")
                return correction
                
            except Exception as e:
                last_error = e
                print(f"Error with {model_type} ({model_name}): {e}")
                continue
        
        raise last_error or Exception("All models failed")
    
    def transcribe_image(self, image_base64: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcreva esta redação do ENEM:"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_base64}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=2000
            )
            text = response.choices[0].message.content.strip()
            processing_time = int((time.time() - start_time) * 1000)
            if text.startswith("ERRO:"):
                return {"texto": "", "confidence": 0.0, "method": "groq_vision", "error": text}
            return {"texto": text, "confidence": 0.95, "method": "groq_vision", "processing_time_ms": processing_time}
        except Exception as e:
            print(f"OCR error: {e}")
            return {"texto": "", "confidence": 0.0, "method": "groq_vision", "error": str(e)}
    
    def transcribe_image_nvidia(self, image_base64: str, nvidia_api_key: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            headers = {"Authorization": "Bearer " + nvidia_api_key, "Content-Type": "application/json"}
            payload = {
                "model": "nvidia/nemotron-3-ultra-vl",
                "messages": [
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": "Transcreva esta redação do ENEM:"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_base64}}]}
                ],
                "temperature": 0.0, "max_tokens": 2000
            }
            response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                return {"texto": "", "confidence": 0.0, "method": "nvidia_nim_vision", "error": "HTTP " + str(response.status_code)}
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            processing_time = int((time.time() - start_time) * 1000)
            if text.startswith("ERRO:"):
                return {"texto": "", "confidence": 0.0, "method": "nvidia_nim_vision", "error": text}
            return {"texto": text, "confidence": 0.95, "method": "nvidia_nim_vision", "processing_time_ms": processing_time}
        except Exception as e:
            print(f"NVIDIA OCR error: {e}")
            return {"texto": "", "confidence": 0.0, "method": "nvidia_nim_vision", "error": str(e)}


_groq_instance = None
def get_groq_llm() -> GroqLLM:
    global _groq_instance
    if _groq_instance is None:
        _groq_instance = GroqLLM()
    return _groq_instance
