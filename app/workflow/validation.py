"""Evidence Validation node (Milestone 2B.1; spec Sections 2.6-2.7).

The minimum safe validation required for the Hybrid Q&A vertical slice:

1. Reject evidence items with empty or whitespace-only supporting_text.
2. Deduplicate repeated chunk_id results (keep the first occurrence, which
   carries the best fused rank) without losing useful provenance — dropped
   duplicate ids are recorded in the trace.
3. Enforce the 24,000-character context budget (MAX_EVIDENCE_CONTEXT_CHARS,
   mirroring MAX_CONTEXT_CHARS in lakebase_retriever.py) on the cumulative
   supporting_text length.  Items beyond the budget are dropped and
   recorded in the trace.  The authoritative header-aware packing remains
   _assemble_context in the retriever.
4. If no usable evidence remains, mark the state as abstaining with a clear
   reason so answer generation is skipped downstream.

Deliberately NOT implemented here: semantic or RRF relevance thresholds.
  Their calibration requires a separate evaluation task (Milestone 4) —
  hardcoding arbitrary thresholds is forbidden by the behavior spec
  (Section 2.6, rule 1).

If a node upstream already set an error (e.g. hybrid retrieval failed),
the validation passes through without abstaining so the error reaches
final_answer.
"""

from __future__ import annotations

from typing import Optional

from .state import MAX_EVIDENCE_CONTEXT_CHARS
from .traces import make_trace


def make_evidence_validation_node(max_context_chars: Optional[int] = None):
    """Factory: create the evidence-validation LangGraph node.

    Args:
        max_context_chars: override the evidence context budget (used by
                           unit tests); defaults to MAX_EVIDENCE_CONTEXT_CHARS.
    """
    limit = MAX_EVIDENCE_CONTEXT_CHARS if max_context_chars is None else int(max_context_chars)

    def evidence_validation_node(state: dict) -> dict:
        # Upstream error (e.g. retrieval failure): pass through.  The error
        # must reach final_answer; abstaining here would mask it.
        if state.get("error"):
            return {
                "trace": [
                    make_trace(
                        "evidence_validation",
                        action="skipped",
                        reason="upstream error — passing through to final_answer",
                        has_error=True,
                    )
                ]
            }

        evidence = state.get("evidence", []) or []
        received = len(evidence)

        # Rule 1: non-empty supporting text
        usable = [e for e in evidence if (getattr(e, "supporting_text", "") or "").strip()]
        dropped_empty = received - len(usable)

        # Rule 2: deduplicate by chunk_id, keep first (best fused rank)
        seen: set = set()
        deduped = []
        dropped_dupes: list = []
        for item in usable:
            if item.chunk_id in seen:
                dropped_dupes.append(item.chunk_id)
                continue
            seen.add(item.chunk_id)
            deduped.append(item)

        # Rule 3: cumulative context budget on supporting_text
        kept = []
        overflow: list = []
        total_chars = 0
        for item in deduped:
            n = len(item.supporting_text or "")
            if total_chars + n > limit:
                overflow.append(item.chunk_id)
                continue
            kept.append(item)
            total_chars += n

        # Rule 4: abstain with a clear reason when nothing usable remains
        abstain = not kept
        abstain_reason = None
        if abstain:
            # Fix 011 (2026-10-30, Stage A): when an upstream node (evidence
            # refinement) already recorded WHY no evidence remains (e.g. all
            # retrieved chunks were filtered as irrelevant to the question),
            # keep that more specific reason — the generic 'No evidence
            # retrieved.' below would mask the true decision chain.
            abstain_reason = state.get("abstain_reason")
            if abstain_reason is None:
                if received == 0:
                    abstain_reason = "No evidence retrieved."
                elif not usable:
                    abstain_reason = (
                        "All retrieved evidence had empty or whitespace-only supporting text."
                    )
                elif overflow and not kept:
                    abstain_reason = (
                        f"All usable evidence exceeded the {limit}-character context budget."
                    )
                else:
                    abstain_reason = (
                        "No usable evidence remained after validation "
                        "(empty text, duplicates, or context budget)."
                    )

        details = {
            "received": received,
            "kept": len(kept),
            "dropped_empty_text": dropped_empty,
            "dropped_duplicate_chunk_ids": dropped_dupes,
            "dropped_context_overflow_chunk_ids": overflow,
            "total_supporting_chars": total_chars,
            "context_limit": limit,
        }
        return {
            "evidence": kept,
            "abstain": abstain,
            "abstain_reason": abstain_reason,
            "trace": [
                make_trace(
                    "evidence_validation",
                    action="validated" if kept else "rejected",
                    reason=(
                        f"{len(kept)} usable evidence items"
                        if kept
                        else (abstain_reason or "no usable evidence")
                    ),
                    **details,
                )
            ],
        }

    return evidence_validation_node
