"""RAG Pipeline for ENEM essay correction - ChromaDB + Few-shot retrieval."""
import json
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path


class ENEMRAGPipeline:
    """RAG pipeline with ENEM guidelines and few-shot examples."""
    
    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collections
        self.guidelines_collection = self._get_or_create_collection("enem_guidelines")
        self.fewshots_collection = self._get_or_create_collection("enem_fewshots")
        
        self._initialized = False
    
    def _get_or_create_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def initialize(self, guidelines_dir: str = "data/matriz", fewshots_path: str = "data/few_shots/few_shots.json"):
        """Initialize collections with guidelines and few-shots."""
        if self._initialized:
            return
        
        # Load guidelines from matriz PDFs
        self._load_guidelines(guidelines_dir)
        
        # Load few-shot examples
        self._load_fewshots(fewshots_path)
        
        self._initialized = True
        print(f"RAG initialized: {self.guidelines_collection.count()} guidelines, {self.fewshots_collection.count()} few-shots")
    
    def _load_guidelines(self, guidelines_dir: str):
        """Load ENEM guidelines from text files."""
        guidelines_path = Path(guidelines_dir)
        if not guidelines_path.exists():
            return
        
        # Competency descriptions (from cartilha 2024)
        competencias = {
            "competencia_1": {
                "titulo": "Competência 1 - Domínio da modalidade escrita formal",
                "descricao": "Demonstrar domínio da modalidade escrita formal da língua portuguesa. Avalia ortografia, acentuação, uso de hífen, emprego de maiúsculas/minúsculas, separação silábica, regência verbal e nominal, concordância verbal e nominal, pontuação. Nota 0-200.",
                "criterios": [
                    "Zero desvios graves → 200 pontos",
                    "Desvios leves (até 2) → 160 pontos",
                    "Desvios moderados (3-4) → 120 pontos",
                    "Desvios frequentes (5-6) → 80 pontos",
                    "Muitos desvios graves → 40 pontos",
                    "Texto ilegível/fora do padrão → 0 pontos"
                ]
            },
            "competencia_2": {
                "titulo": "Competência 2 - Compreender a proposta e desenvolver o tema",
                "descricao": "Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema dentro dos limites estruturais do texto dissertativo-argumentativo em prosa. Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista. Nota 0-200.",
                "criterios": [
                    "Desenvolve tema plenamente, estrutura dissertativo-argumentativa perfeita → 200",
                    "Desenvolve bem, estrutura adequada → 160",
                    "Desenvolve parcialmente, alguns problemas estruturais → 120",
                    "Desenvolve pouco, estrutura frágil → 80",
                    "Não desenvolve, estrutura inadequada → 40",
                    "Fuga total ao tema / não é dissertativo-argumentativo → 0"
                ]
            },
            "competencia_3": {
                "titulo": "Competência 3 - Selecionar, relacionar, organizar e interpretar informações",
                "descricao": "Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista. Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação. Nota 0-200.",
                "criterios": [
                    "Argumentação consistente, progressão temática clara, repertório diversificado → 200",
                    "Argumentação boa, progressão adequada → 160",
                    "Argumentação parcial, progressão com falhas → 120",
                    "Argumentação frágil, progressão confusa → 80",
                    "Argumentação muito frágil / repetitiva → 40",
                    "Sem argumentação / cópia dos textos de apoio → 0"
                ]
            },
            "competencia_4": {
                "titulo": "Competência 4 - Conhecimento dos mecanismos linguísticos para argumentação",
                "descricao": "Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação. Coesão textual: conectivos, referências, substituições, elipses, progressão temática. Nota 0-200.",
                "criterios": [
                    "Texto coeso, conectivos variados e adequados, progressão temática fluida → 200",
                    "Texto coeso, bons conectivos → 160",
                    "Alguns problemas de coesão, conectivos repetitivos → 120",
                    "Problemas frequentes de coesão → 80",
                    "Muitos problemas, texto fragmentado → 40",
                    "Texto sem coesão / incompreensível → 0"
                ]
            },
            "competencia_5": {
                "titulo": "Competência 5 - Elaborar proposta de intervenção",
                "descricao": "Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos. Deve conter: agente, ação, modo/efeito, detalhamento. Nota 0-200.",
                "criterios": [
                    "Proposta completa (agente, ação, modo, detalhamento), respeita DH → 200",
                    "Proposta completa, pequeno problema → 160",
                    "Proposta incompleta (falta 1 elemento) → 120",
                    "Proposta vaga / genérica → 80",
                    "Proposta muito frágil / desrespeita DH → 40",
                    "Sem proposta / copia texto de apoio → 0"
                ]
            }
        }
        
        documents = []
        metadatas = []
        ids = []
        
        for comp_id, comp_data in competencias.items():
            # Main competency document
            doc = f"{comp_data['titulo']}\n\n{comp_data['descricao']}\n\nCritérios de pontuação:\n" + "\n".join(f"- {c}" for c in comp_data['criterios'])
            documents.append(doc)
            metadatas.append({
                "type": "competency_guideline",
                "competency": comp_id,
                "source": "cartilha_enem_2024"
            })
            ids.append(f"guideline_{comp_id}")
            
            # Individual criterion documents for finer retrieval
            for i, criterion in enumerate(comp_data['criterios']):
                documents.append(f"{comp_data['titulo']}: {criterion}")
                metadatas.append({
                    "type": "competency_criterion",
                    "competency": comp_id,
                    "criterion_index": i,
                    "source": "cartilha_enem_2024"
                })
                ids.append(f"guideline_{comp_id}_criterion_{i}")
        
        # Add general ENEM info
        documents.append("""
ENEM Redação - Regras Gerais:
- Texto dissertativo-argumentativo em prosa
- Mínimo 7 linhas, máximo 30 linhas
- Nota 0 a 1000 (soma de 5 competências × 200)
- Dois avaliadores independentes
- Discrepância > 100 pontos total ou > 80 em alguma competência → terceiro avaliador
- Nota zero: fuga ao tema, não dissertativo-argumentativo, cópia indevida, texto ilegível, impropérios, desrespeito aos direitos humanos
""")
        metadatas.append({"type": "general_rules", "source": "cartilha_enem_2024"})
        ids.append("guideline_general_rules")
        
        if documents:
            self.guidelines_collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    def _load_fewshots(self, fewshots_path: str):
        """Load few-shot examples from JSON."""
        path = Path(fewshots_path)
        if not path.exists():
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            fewshots = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, fs in enumerate(fewshots):
            # Create a rich document with essay + grades + analysis
            grades = fs['grades']
            doc = f"""EXEMPLO DE REDAÇÃO - Faixa: {fs['score_band']} - Total: {grades['total']}/1000

TEMA/PROPOSTA: {fs.get('prompt', 'Não informado')[:500]}

TEXTOS DE APOIO: {fs.get('supporting_text', 'Não informado')[:300]}

REDAÇÃO:
{fs['essay_text']}

NOTAS POR COMPETÊNCIA:
- Competência 1 (Norma culta): {grades['competencia_1']}/200
- Competência 2 (Desenvolvimento tema): {grades['competencia_2']}/200
- Competência 3 (Argumentação): {grades['competencia_3']}/200
- Competência 4 (Coesão): {grades['competencia_4']}/200
- Competência 5 (Proposta intervenção): {grades['competencia_5']}/200
- TOTAL: {grades['total']}/1000

COMENTÁRIO GERAL: {fs.get('comment', '')}
COMENTÁRIO ESPECÍFICO: {fs.get('specific_comment', '')}
"""
            documents.append(doc)
            metadatas.append({
                "type": "fewshot_example",
                "score_band": fs['score_band'],
                "total_score": grades['total'],
                "c1": grades['competencia_1'],
                "c2": grades['competencia_2'],
                "c3": grades['competencia_3'],
                "c4": grades['competencia_4'],
                "c5": grades['competencia_5'],
                "year": fs.get('year', 0),
                "source": "kamel_usp_sourceB" if "kamel" in str(fs.get('source', '')) else "davicab23" if "Davicab" in fs.get('score_band', '') else "gradesThousand"
            })
            ids.append(f"fewshot_{i}")
        
        if documents:
            self.fewshots_collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    def retrieve_guidelines(self, query: str, competency: Optional[str] = None, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant guidelines."""
        where = {"type": "competency_guideline"} if competency else {"type": {"$in": ["competency_guideline", "competency_criterion", "general_rules"]}}
        if competency:
            where["competency"] = competency
        
        results = self.guidelines_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        return [
            {"document": doc, "metadata": meta, "id": id_}
            for doc, meta, id_ in zip(results['documents'][0], results['metadatas'][0], results['ids'][0])
        ]
    
    def retrieve_fewshots(self, query: str, score_band: Optional[str] = None, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant few-shot examples."""
        where = {}
        if score_band:
            where["score_band"] = score_band
        
        results = self.fewshots_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where if where else None
        )
        
        return [
            {"document": doc, "metadata": meta, "id": id_}
            for doc, meta, id_ in zip(results['documents'][0], results['metadatas'][0], results['ids'][0])
        ]
    
    def retrieve_similar_essays(self, essay_text: str, n_results: int = 3) -> List[Dict]:
        """Find few-shots similar to the input essay."""
        return self.retrieve_fewshots(essay_text[:1000], n_results=n_results)
    
    def get_stats(self) -> Dict:
        return {
            "guidelines_count": self.guidelines_collection.count(),
            "fewshots_count": self.fewshots_collection.count()
        }


# Singleton instance
_rag_instance: Optional[ENEMRAGPipeline] = None


def get_rag_pipeline() -> ENEMRAGPipeline:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ENEMRAGPipeline()
        _rag_instance.initialize()
    return _rag_instance
