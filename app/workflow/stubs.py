"""Stub capability nodes for the LangGraph foundation (Milestone 2A).

Each stub is a deterministic graph node that returns a placeholder result
with trace info.  No LLM, database, Lakebase, or PDF access.

In Milestone 3 these stubs will be replaced with real implementations.
"""

from __future__ import annotations

from .state import Intent
from .traces import make_trace


def corpus_overview_stub(state: dict) -> dict:
    """Stub for Capability A (Corpus Overview)."""
    intent: Intent = state.get("intent")
    return {
        "metadata_result": {
            "stub": True,
            "capability": "corpus_overview",
            "scope": intent.scope if intent else "all",
            "message": "Corpus overview stub - Milestone 3 will connect SQL queries.",
        },
        "trace": [
            make_trace("corpus_overview_stub", action="executed", route="corpus_overview",
                       reason="stub node executed")
        ],
    }


def hybrid_qa_stub(state: dict) -> dict:
    """Stub for Capability B (Hybrid Q&A)."""
    return {
        "evidence": [],
        "answer": None,
        "trace": [
            make_trace("hybrid_qa_stub", action="executed", route="hybrid_qa",
                       reason="stub - will call search_hybrid_v2 in Milestone 3")
        ],
    }


def clause_discovery_stub(state: dict) -> dict:
    """Stub for Capability C (Clause Discovery)."""
    intent: Intent = state.get("intent")
    return {
        "evidence": [],
        "metadata_result": {
            "stub": True,
            "capability": "clause_discovery",
            "clause_pattern": intent.clause_pattern if intent else None,
            "coverage": "0% (stub - no documents searched)",
            "message": "Clause discovery stub - Milestone 3 will run per-document retrieval.",
        },
        "trace": [
            make_trace("clause_discovery_stub", action="executed", route="clause_discovery",
                       reason="stub - per-document retrieval in Milestone 3")
        ],
    }


def structured_lookup_stub(state: dict) -> dict:
    """Stub for Capability D (Structured Lookup)."""
    intent: Intent = state.get("intent")
    return {
        "lookup_result": {
            "stub": True,
            "capability": "structured_lookup",
            "target_field": intent.target_field if intent else None,
            "target_doc_names": intent.target_doc_names if intent else [],
            "message": "Structured lookup stub - Milestone 3 will query extracted_fields.",
        },
        "trace": [
            make_trace("structured_lookup_stub", action="executed", route="structured_lookup",
                       reason="stub - will query extracted_fields in Milestone 3")
        ],
    }


def comparison_stub(state: dict) -> dict:
    """Stub for Capability E (Cross-Document Comparison)."""
    intent: Intent = state.get("intent")
    return {
        "comparison_result": {
            "stub": True,
            "capability": "comparison",
            "comparison_aspect": intent.comparison_aspect if intent else None,
            "target_doc_names": intent.target_doc_names if intent else [],
            "message": "Comparison stub - Milestone 3 will retrieve and align provisions.",
        },
        "evidence": [],
        "trace": [
            make_trace("comparison_stub", action="executed", route="comparison",
                       reason="stub - per-document retrieval in Milestone 3")
        ],
    }


def multi_step_stub(state: dict) -> dict:
    """Stub for Capability G (Multi-Step Planning)."""
    plan = state.get("plan", [])
    trace_entries = []
    for step in plan:
        node_name = step.node if hasattr(step, "node") else step.get("node", "unknown")
        sid = getattr(step, "step_id", step.get("step_id", "?")) if isinstance(step, dict) else getattr(step, "step_id", "?")
        trace_entries.append(
            make_trace("multi_step_stub", action=f"step_{sid}: {node_name}",
                       route="multi_step", reason="stub execution of plan step")
        )
    return {
        "answer": None,
        "trace": trace_entries or [
            make_trace("multi_step_stub", action="executed", route="multi_step", reason="empty plan")
        ],
    }


def evidence_validation_stub(state: dict) -> dict:
    """Stub for Evidence Validation (spec Section 2.6)."""
    evidence = state.get("evidence", [])
    return {
        "abstain": len(evidence) == 0,
        "abstain_reason": "No evidence retrieved (stub validation)." if not evidence else None,
        "trace": [
            make_trace("evidence_validation", action="validated",
                       reason=f"{len(evidence)} evidence items (stub)", abstain=len(evidence) == 0)
        ],
    }


def final_answer_node(state: dict) -> dict:
    """Final Answer Generation node (spec Section 2.11)."""
    if state.get("abstain"):
        answer = "The available documents do not contain enough information to answer this question."
    elif state.get("error"):
        answer = f"Error: {state['error']}"
    elif state.get("answer"):
        answer = state["answer"]
    elif state.get("metadata_result"):
        meta = state["metadata_result"]
        if meta.get("stub"):
            cap = meta.get("capability", "unknown")
            label = cap.replace("_", " ").title()
            answer = f"[{label}] {meta.get('message', 'Stub result.')}"
        else:
            answer = str(meta)
    elif state.get("lookup_result"):
        lookup = state["lookup_result"]
        answer = f"[Structured Lookup] {lookup.get('message', 'Stub result.')}" if lookup.get("stub") else str(lookup)
    elif state.get("comparison_result"):
        comp = state["comparison_result"]
        answer = f"[Comparison] {comp.get('message', 'Stub result.')}" if comp.get("stub") else str(comp)
    else:
        answer = "No answer generated."
    return {
        "answer": answer,
        "trace": [
            make_trace("final_answer", action="generated", reason="final answer formatted",
                       abstain=state.get("abstain", False), has_error=bool(state.get("error")))
        ],
    }


def clarification_node(state: dict) -> dict:
    """Clarification route (spec Section 2.9)."""
    intent: Intent = state.get("intent")
    message = intent.clarification_needed if intent and intent.clarification_needed else "Could you rephrase your question?"
    return {
        "answer": message,
        "trace": [
            make_trace("clarification", action="clarification_requested", reason=message[:200])
        ],
    }
