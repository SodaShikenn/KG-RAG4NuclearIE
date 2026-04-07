"""
CLI entry point for the KG-RAG pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

from .pipeline import KGRAGPipeline, PipelineConfig


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    llm = raw.get("llm", {})
    emb = raw.get("embedding", {})
    doc = raw.get("documents", {})
    kg = raw.get("knowledge_graph", {})
    ret = raw.get("retrieval", {})
    ver = raw.get("verification", {})

    return PipelineConfig(
        llm_provider=llm.get("provider", "openai"),
        llm_model=llm.get("model", "gpt-4o"),
        llm_temperature=llm.get("temperature", 0.0),
        llm_max_tokens=llm.get("max_tokens", 4096),
        embedding_model=emb.get("model", "all-MiniLM-L6-v2"),
        input_dir=doc.get("input_dir", "data/sample_docs"),
        chunk_size=doc.get("chunk_size", 512),
        chunk_overlap=doc.get("chunk_overlap", 64),
        max_triples_per_chunk=kg.get("max_triples_per_chunk", 20),
        expansion_hops=kg.get("expansion_hops", 2),
        top_k_bm25=ret.get("top_k_bm25", 10),
        top_k_vector=ret.get("top_k_vector", 10),
        top_k_final=ret.get("top_k_final", 10),
        rrf_k=ret.get("rrf_k", 60),
        top_k_kg_extra=ret.get("top_k_kg_extra", 5),
        ngram_size=ver.get("ngram_size", 3),
        support_threshold=ver.get("support_threshold", 0.3),
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="KG-RAG: Knowledge Graph-Enhanced RAG for Nuclear Document Extraction"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Query to run (if omitted, enters interactive mode)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for JSON results",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = KGRAGPipeline(config)

    # Index documents
    pipeline.index()

    if args.query:
        # Single query mode
        results = pipeline.query(args.query)
        print("\n" + pipeline.format_results(results))
        if args.output:
            with open(args.output, "w") as f:
                f.write(pipeline.results_to_json(results))
            print(f"\nResults saved to {args.output}")
    else:
        # Interactive mode
        print("Entering interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            results = pipeline.query(query)
            print("\n" + pipeline.format_results(results) + "\n")


if __name__ == "__main__":
    main()
