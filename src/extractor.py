"""
Step 5: Grounded Extraction & Verification
- 8-field structured extraction via LLM
- Claim verification via N-gram matching against evidence chunks
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .document_loader import TextChunk
from .llm_client import call_llm_json


# ---------------------------------------------------------------------------
# 8-field extraction schema
# ---------------------------------------------------------------------------

EXTRACTION_FIELDS = [
    "facility_name",
    "event_date",
    "event_location",
    "event_description",
    "cause",
    "detection_method",
    "plant_status",
    "response",
]

EXTRACTION_SYSTEM = """\
You are a nuclear engineering information extraction assistant.
Given evidence text chunks, extract structured information about a nuclear facility incident.
Return a JSON object with exactly these keys:
  facility_name, event_date, event_location, event_description,
  cause, detection_method, plant_status, response
If information for a field is not found, set its value to null.
Only use information explicitly stated in the evidence. Do not hallucinate."""

EXTRACTION_PROMPT = """\
Based ONLY on the following evidence chunks, extract structured incident information.

Evidence:
{evidence}

Query: {query}

Return a JSON object with keys: facility_name, event_date, event_location,
event_description, cause, detection_method, plant_status, response."""


@dataclass
class ExtractionResult:
    field: str
    value: str | None
    status: str  # "Supported", "Unsupported", "Contradicted"


class VerificationStatus(str, Enum):
    SUPPORTED = "Supported"
    UNSUPPORTED = "Unsupported"
    CONTRADICTED = "Contradicted"


# ---------------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------------

def extract_fields(
    query: str,
    evidence_chunks: list[TextChunk],
    llm_config: dict[str, Any],
) -> dict[str, str | None]:
    """Use LLM to extract 8-field structured information from evidence."""
    evidence_text = "\n\n---\n\n".join(
        f"[{c.doc_id} p.{c.page}] {c.text}" for c in evidence_chunks
    )
    prompt = EXTRACTION_PROMPT.format(evidence=evidence_text, query=query)
    try:
        result = call_llm_json(
            prompt=prompt,
            system=EXTRACTION_SYSTEM,
            **llm_config,
        )
    except Exception:
        return {f: None for f in EXTRACTION_FIELDS}

    if not isinstance(result, dict):
        return {f: None for f in EXTRACTION_FIELDS}

    return {f: result.get(f) for f in EXTRACTION_FIELDS}


# ---------------------------------------------------------------------------
# N-gram claim verification
# ---------------------------------------------------------------------------

def _get_ngrams(text: str, n: int) -> Counter:
    """Extract character-level n-grams from text."""
    tokens = re.findall(r"\w+", text.lower())
    ngrams: list[str] = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(" ".join(tokens[i : i + n]))
    return Counter(ngrams)


def verify_claim(
    claim: str,
    evidence_texts: list[str],
    ngram_size: int = 3,
    support_threshold: float = 0.3,
) -> VerificationStatus:
    """
    Verify a single claim against evidence using n-gram overlap.
    Returns Supported if overlap ratio >= threshold, else Unsupported.
    """
    if not claim or not claim.strip():
        return VerificationStatus.UNSUPPORTED

    claim_ngrams = _get_ngrams(claim, ngram_size)
    if not claim_ngrams:
        return VerificationStatus.UNSUPPORTED

    # Merge all evidence n-grams
    evidence_ngrams: Counter = Counter()
    for text in evidence_texts:
        evidence_ngrams.update(_get_ngrams(text, ngram_size))

    # Compute overlap
    overlap = sum((claim_ngrams & evidence_ngrams).values())
    total = sum(claim_ngrams.values())

    ratio = overlap / total if total > 0 else 0.0

    if ratio >= support_threshold:
        return VerificationStatus.SUPPORTED
    return VerificationStatus.UNSUPPORTED


def extract_and_verify(
    query: str,
    evidence_chunks: list[TextChunk],
    llm_config: dict[str, Any],
    ngram_size: int = 3,
    support_threshold: float = 0.3,
) -> list[ExtractionResult]:
    """
    Full extraction + verification pipeline:
    1. LLM extracts 8-field JSON from evidence
    2. Each field value is verified against evidence via n-gram matching
    3. Only 'Supported' claims are retained
    """
    raw = extract_fields(query, evidence_chunks, llm_config)
    evidence_texts = [c.text for c in evidence_chunks]

    results: list[ExtractionResult] = []
    for field_name in EXTRACTION_FIELDS:
        value = raw.get(field_name)
        if value is None:
            results.append(ExtractionResult(field=field_name, value=None, status="Unsupported"))
            continue
        status = verify_claim(str(value), evidence_texts, ngram_size, support_threshold)
        results.append(ExtractionResult(
            field=field_name,
            value=value if status == VerificationStatus.SUPPORTED else None,
            status=status.value,
        ))

    return results
