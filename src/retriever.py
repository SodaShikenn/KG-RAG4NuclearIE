"""
Step 4: KG-Enhanced Hybrid Retrieval
- BM25 keyword-based retrieval
- Vector (dense) embedding retrieval
- Reciprocal Rank Fusion (RRF)
- Entity linking + multi-hop KG expansion for extra chunks
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .document_loader import TextChunk
from .knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Tokeniser helper
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Vector index (FAISS)
# ---------------------------------------------------------------------------

@dataclass
class VectorIndex:
    embeddings: np.ndarray  # (N, dim)
    model: SentenceTransformer

    def query(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_similarity) pairs."""
        q_emb = self.model.encode([text], normalize_embeddings=True)
        scores = self.embeddings @ q_emb.T  # (N, 1)
        scores = scores.squeeze(-1)
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


def build_vector_index(
    chunks: list[TextChunk],
    model_name: str = "all-MiniLM-L6-v2",
) -> VectorIndex:
    """Encode all chunks and build a vector index."""
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return VectorIndex(embeddings=np.array(embeddings), model=model)


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    bm25: BM25Okapi
    _corpus_tokens: list[list[str]]

    def query(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        tokens = _tokenize(text)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


def build_bm25_index(chunks: list[TextChunk]) -> BM25Index:
    corpus_tokens = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    return BM25Index(bm25=bm25, _corpus_tokens=corpus_tokens)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using RRF. Returns sorted (id, score)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# KG-Enhanced Retriever
# ---------------------------------------------------------------------------

@dataclass
class HybridRetriever:
    chunks: list[TextChunk]
    bm25_index: BM25Index
    vector_index: VectorIndex
    kg: KnowledgeGraph
    # Config
    top_k_bm25: int = 10
    top_k_vector: int = 10
    top_k_final: int = 10
    rrf_k: int = 60
    expansion_hops: int = 2
    top_k_kg_extra: int = 5

    def retrieve(self, query: str) -> list[TextChunk]:
        """
        Full retrieval pipeline:
        1. BM25 + Vector hybrid search -> seed chunks (via RRF)
        2. Entity linking: map query keywords to KG nodes
        3. Multi-hop KG expansion -> extra chunks
        4. Combine and deduplicate
        """
        # Step 1: Hybrid search
        bm25_results = self.bm25_index.query(query, self.top_k_bm25)
        vector_results = self.vector_index.query(query, self.top_k_vector)
        fused = reciprocal_rank_fusion([bm25_results, vector_results], k=self.rrf_k)
        seed_ids = [doc_id for doc_id, _ in fused[: self.top_k_final]]

        # Step 2: Entity linking — match query tokens to KG nodes
        query_tokens = set(_tokenize(query))
        matched_nodes: set[str] = set()
        for node in self.kg.graph.nodes:
            node_tokens = set(_tokenize(node))
            if query_tokens & node_tokens:
                matched_nodes.add(node)

        # Step 3: KG expansion
        expanded_nodes: set[str] = set()
        for node in matched_nodes:
            expanded_nodes.update(self.kg.get_neighbors(node, hops=self.expansion_hops))
        expanded_nodes.update(matched_nodes)

        extra_chunk_ids = self.kg.get_linked_chunk_ids(expanded_nodes)
        # Remove already-selected seed chunks
        extra_chunk_ids -= set(seed_ids)
        # Take top-k extra (by lowest chunk_id as a simple heuristic)
        extra_ids_sorted = sorted(extra_chunk_ids)[: self.top_k_kg_extra]

        # Step 4: Combine
        final_ids = seed_ids + extra_ids_sorted
        chunk_map = {c.chunk_id: c for c in self.chunks}
        return [chunk_map[i] for i in final_ids if i in chunk_map]
