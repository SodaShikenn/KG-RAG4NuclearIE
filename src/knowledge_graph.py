"""
Step 2 & 3: Knowledge Graph Construction
- Triple extraction from text chunks via LLM
- Graph construction, merging, and deduplication using NetworkX
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .document_loader import TextChunk
from .llm_client import call_llm_json


# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------

TRIPLE_EXTRACTION_SYSTEM = """\
You are an information extraction assistant specializing in nuclear engineering documents.
Extract factual triples (subject, predicate, object) from the given text.
Return a JSON array of objects with keys: "subject", "predicate", "object".
Only extract factual, clearly stated relationships. Do not infer or hallucinate."""

TRIPLE_EXTRACTION_PROMPT = """\
Extract all factual triples from the following text.
Return ONLY a JSON array. Example:
[
  {{"subject": "DFR", "predicate": "operated_at", "object": "60 MW(th)"}},
  {{"subject": "leak", "predicate": "caused_by", "object": "NaK coolant leakage"}}
]

Text:
{text}"""


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    source_chunk_id: int = -1


def extract_triples_from_chunk(
    chunk: TextChunk,
    llm_config: dict[str, Any],
    max_triples: int = 20,
) -> list[Triple]:
    """Use LLM to extract (subject, predicate, object) triples from a chunk."""
    prompt = TRIPLE_EXTRACTION_PROMPT.format(text=chunk.text)
    try:
        raw = call_llm_json(
            prompt=prompt,
            system=TRIPLE_EXTRACTION_SYSTEM,
            **llm_config,
        )
    except Exception:
        return []

    triples: list[Triple] = []
    if not isinstance(raw, list):
        return triples
    for item in raw[:max_triples]:
        if isinstance(item, dict) and all(k in item for k in ("subject", "predicate", "object")):
            triples.append(Triple(
                subject=_normalize(item["subject"]),
                predicate=_normalize(item["predicate"]),
                object=_normalize(item["object"]),
                source_chunk_id=chunk.chunk_id,
            ))
    return triples


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeGraph:
    """Wrapper around a NetworkX MultiDiGraph with chunk-linking metadata."""
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    # Mapping: node label -> set of chunk_ids that mention this node
    node_to_chunks: dict[str, set[int]] = field(default_factory=dict)

    def add_triple(self, triple: Triple) -> None:
        g = self.graph
        s, p, o = triple.subject, triple.predicate, triple.object

        # Add nodes
        for node in (s, o):
            if not g.has_node(node):
                g.add_node(node)
            self.node_to_chunks.setdefault(node, set()).add(triple.source_chunk_id)

        # Add edge (allow multiple edges between same pair)
        g.add_edge(s, o, predicate=p, source_chunk_id=triple.source_chunk_id)

    def get_neighbors(self, node: str, hops: int = 1) -> set[str]:
        """Return all nodes within *hops* edges of *node* (undirected)."""
        visited: set[str] = set()
        frontier: set[str] = {node}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for n in frontier:
                if n in visited:
                    continue
                visited.add(n)
                next_frontier.update(self.graph.successors(n))
                next_frontier.update(self.graph.predecessors(n))
            frontier = next_frontier - visited
        visited.update(frontier)
        visited.discard(node)
        return visited

    def get_linked_chunk_ids(self, nodes: set[str]) -> set[int]:
        """Return all chunk_ids linked to the given nodes."""
        chunk_ids: set[int] = set()
        for n in nodes:
            chunk_ids.update(self.node_to_chunks.get(n, set()))
        return chunk_ids

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()


def build_knowledge_graph(
    chunks: list[TextChunk],
    llm_config: dict[str, Any],
    max_triples_per_chunk: int = 20,
    verbose: bool = True,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from document chunks via LLM triple extraction."""
    from tqdm import tqdm

    kg = KnowledgeGraph()
    iterator = tqdm(chunks, desc="Extracting triples") if verbose else chunks

    for chunk in iterator:
        triples = extract_triples_from_chunk(chunk, llm_config, max_triples_per_chunk)
        for t in triples:
            kg.add_triple(t)

    if verbose:
        print(f"KG built: {kg.num_nodes} nodes, {kg.num_edges} edges")
    return kg
