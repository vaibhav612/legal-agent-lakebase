"""Capability Guard node (spec Section 1.9 / Capability H).

Rejects unsupported tasks and oversized requests gracefully, giving a
concise reason and offering a supported alternative.
"""

from __future__ import annotations

from .state import Intent, MAX_DOC_LISTING, MAX_SOURCE_DOCS, MAX_COMPARISON_DOCS
from .traces import make_trace


def is_oversized_request(intent: Intent) -> bool:
    """Check whether the request exceeds defined limits."""
    # More than MAX_DOC_LISTING documents in a metadata listing
    if (
        intent.label == "corpus_overview"
        and intent.scope == "specific_docs"
        and len(intent.target_doc_names) > MAX_DOC_LISTING
    ):
        return True
    # More than MAX_COMPARISON_DOCS documents in a comparison execution
    # (separate from MAX_SOURCE_DOCS which is the source-popup display limit)
    if (
        intent.label == "comparison"
        and len(intent.target_doc_names) > MAX_COMPARISON_DOCS
    ):
        return True
    return False


def is_corpus_dump(query: str) -> bool:
    """Check whether the query asks to dump the entire corpus contents."""
    q = query.lower().strip()
    dump_patterns = [
        "show me all text from all documents",
        "show all text from all",
        "dump all documents",
        "give me all the text",
        "display all document content",
        "all text from every document",
    ]
    return any(p in q for p in dump_patterns)


def guard_node(state: dict) -> dict:
    """Capability Guard node: reject unsupported/oversized requests.

    Returns a rejection message with reason and supported alternatives.
    """
    intent: Intent = state.get("intent")
    user_query: str = state.get("user_query", "")

    reason = "The request is not supported."
    alternatives = (
        "I can search for specific clauses, look up extracted fields, "
        "or summarize the corpus composition."
    )

    if intent and intent.clarification_needed:
        reason = intent.clarification_needed
        answer = f"{reason}\n\n{alternatives}"
    elif is_corpus_dump(user_query):
        reason = (
            "I cannot display all text from all documents. "
            "This would dump the entire corpus."
        )
        answer = f"{reason}\n\n{alternatives}"
    elif intent and is_oversized_request(intent):
        reason = (
            f"This request exceeds the limit of {MAX_DOC_LISTING} document names "
            f"or {MAX_COMPARISON_DOCS} documents in a comparison."
        )
        answer = f"{reason}\n\n{alternatives}"
    else:
        reason = "No available capability can serve this request."
        answer = f"{reason}\n\n{alternatives}"

    return {
        "answer": answer,
        "error": None,
        "abstain": False,
        "trace": [
            make_trace(
                "capability_guard",
                action="rejected",
                reason=reason,
                intent_label=intent.label if intent else "unknown",
            )
        ],
    }
