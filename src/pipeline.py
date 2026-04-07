"""
KG-RAG Pipeline Orchestrator
Ties all stages together: load → build KG → index → retrieve → extract → verify
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .document_loader import TextChunk, load_documents
from .knowledge_graph import KnowledgeGraph, build_knowledge_graph
from .retriever import (
    BM25Index,
    HybridRetriever,
    VectorIndex,
    build_bm25_index,
    build_vector_index,
)
from .extractor import ExtractionResult, extract_and_verify


@dataclass
class PipelineConfig:
    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    # Documents
    input_dir: str = "data/sample_docs"
    chunk_size: int = 512
    chunk_overlap: int = 64
    # KG
    max_triples_per_chunk: int = 20
    expansion_hops: int = 2
    # Retrieval
    top_k_bm25: int = 10
    top_k_vector: int = 10
    top_k_final: int = 10
    rrf_k: int = 60
    top_k_kg_extra: int = 5
    # Verification
    ngram_size: int = 3
    support_threshold: float = 0.3

    @property
    def llm_config(self) -> dict[str, Any]:
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
        }


@dataclass
class PipelineState:
    chunks: list[TextChunk] = field(default_factory=list)
    kg: KnowledgeGraph | None = None
    bm25_index: BM25Index | None = None
    vector_index: VectorIndex | None = None
    retriever: HybridRetriever | None = None


class KGRAGPipeline:
    """End-to-end KG-RAG pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.state = PipelineState()

    # ------------------------------------------------------------------
    # Stage 1: Load & chunk documents
    # ------------------------------------------------------------------
    def load_documents(self) -> None:
        print(f"Loading documents from {self.config.input_dir} ...")
        self.state.chunks = load_documents(
            self.config.input_dir,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        print(f"  → {len(self.state.chunks)} chunks from documents")

    # ------------------------------------------------------------------
    # Stage 2-3: Build Knowledge Graph
    # ------------------------------------------------------------------
    def build_knowledge_graph(self) -> None:
        print("Building knowledge graph ...")
        self.state.kg = build_knowledge_graph(
            self.state.chunks,
            llm_config=self.config.llm_config,
            max_triples_per_chunk=self.config.max_triples_per_chunk,
        )

    # ------------------------------------------------------------------
    # Stage 4: Build retrieval indices
    # ------------------------------------------------------------------
    def build_indices(self) -> None:
        print("Building BM25 index ...")
        self.state.bm25_index = build_bm25_index(self.state.chunks)

        print(f"Building vector index ({self.config.embedding_model}) ...")
        self.state.vector_index = build_vector_index(
            self.state.chunks,
            model_name=self.config.embedding_model,
        )

        assert self.state.kg is not None
        self.state.retriever = HybridRetriever(
            chunks=self.state.chunks,
            bm25_index=self.state.bm25_index,
            vector_index=self.state.vector_index,
            kg=self.state.kg,
            top_k_bm25=self.config.top_k_bm25,
            top_k_vector=self.config.top_k_vector,
            top_k_final=self.config.top_k_final,
            rrf_k=self.config.rrf_k,
            expansion_hops=self.config.expansion_hops,
            top_k_kg_extra=self.config.top_k_kg_extra,
        )

    # ------------------------------------------------------------------
    # Full indexing pipeline
    # ------------------------------------------------------------------
    def index(self) -> None:
        """Run the full indexing pipeline: load → KG → indices."""
        self.load_documents()
        self.build_knowledge_graph()
        self.build_indices()
        print("Indexing complete.\n")

    # ------------------------------------------------------------------
    # Query: retrieve + extract + verify
    # ------------------------------------------------------------------
    def query(self, question: str) -> list[ExtractionResult]:
        """Run a query through the pipeline and return verified extraction results."""
        assert self.state.retriever is not None, "Call .index() first"

        print(f"Query: {question}")
        evidence = self.state.retriever.retrieve(question)
        print(f"  → Retrieved {len(evidence)} evidence chunks")

        results = extract_and_verify(
            query=question,
            evidence_chunks=evidence,
            llm_config=self.config.llm_config,
            ngram_size=self.config.ngram_size,
            support_threshold=self.config.support_threshold,
        )
        return results

    @staticmethod
    def format_results(results: list[ExtractionResult]) -> str:
        """Format extraction results as a readable table."""
        lines = [f"{'Field':<20} {'Status':<14} Value", "-" * 70]
        for r in results:
            val = r.value if r.value is not None else "—"
            lines.append(f"{r.field:<20} {r.status:<14} {val}")
        return "\n".join(lines)

    @staticmethod
    def results_to_json(results: list[ExtractionResult]) -> str:
        """Serialize results to JSON."""
        data = [
            {"field": r.field, "value": r.value, "status": r.status}
            for r in results
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)
