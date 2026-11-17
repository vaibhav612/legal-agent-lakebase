"""Hybrid Legal Q&A capability node (Milestone 2B.1).

Wraps the existing hybrid retrieval implementation (LakebaseRetriever.
search_hybrid_v2, reached via the retriever's public search() entry point
with RETRIEVER_MODE=hybrid_v2) as a LangGraph node.  No BM25, semantic
search, or RRF logic is duplicated here.

Node contract (spec Section 2.5, Hybrid Q&A Node B):
  Input:  user_query, intent (resolved doc scope, if any)
  Output: evidence (list[EvidenceItem]) preserved in graph state so later
          nodes (evidence_validation, final_answer) can validate it and
          build source references.

The retriever is injected, so unit tests can pass a fake without any
Lakebase connection, and production can pass the real LakebaseRetriever
singleton from lakebase_retriever.get_lakebase_retriever().
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .state import EvidenceItem, Intent
from .traces import make_trace
from .clause_discovery import (
    SqlExecutorLike,
    _CANONICAL_DOCS_SQL,
    _resolve_doc_names,
)


DEFAULT_NUM_RESULTS: int = 5


class RetrieverLike(Protocol):
    """Structural type of the injected retriever (satisfied by LakebaseRetriever)."""

    def search(self, query: str, num_results: int = 5,
               doc_ids: Optional[list[str]] = None) -> list[dict]:
        ...


def search_result_to_evidence(result: dict) -> EvidenceItem:
    """Map one search_hybrid_v2 result row to the shared EvidenceItem schema.

    Preserves: doc_id, document name (doc_name / file_name), chunk_id,
    section_path, page_start, page_end, supporting text (chunk_text), and
    the best available retrieval score (rrf_score, else bm25_score, else
    semantic distance).  Channel-specific ranks and scores are preserved
    in the hybrid_qa trace entry (see _score_summary).
    """
    doc_id = str(result.get("doc_id") or "") or "unknown_doc"
    doc_name = (
        result.get("doc_name")
        or result.get("file_name")
        or result.get("doc_id")
        or "Unknown"
    )
    score = result.get("rrf_score")
    if score is None:
        score = result.get("bm25_score")
    if score is None:
        score = result.get("distance")
    return EvidenceItem(
        doc_id=doc_id,
        doc_name=str(doc_name),
        chunk_id=str(result.get("chunk_id") or "") or "unknown_chunk",
        section_path=result.get("section_path"),
        page_start=result.get("page_start"),
        page_end=result.get("page_end"),
        supporting_text=result.get("chunk_text") or "",
        retrieval_score=float(score) if score is not None else None,
        retrieval_method="hybrid",
        # Fix 011 (2026-10-30, Stage C): preserved table structure rides
        # along from the retriever result (UC chunks_v2 -> Lakebase
        # document_chunks_v2 table_html / has_table).
        table_html=result.get("table_html") or None,
        has_table=bool(result.get("has_table")),
        # Fix 011 (Stage A): this chunk is a top-k retrieval result (as
        # opposed to 'section_expansion' items added by refinement).
        evidence_role="retrieved",
    )


def _score_summary(results: list[dict]) -> list[dict]:
    """Compact per-chunk record of every retrieval score/rank available.

    EvidenceItem carries a single retrieval_score, so the full channel
    detail (rrf, semantic, bm25) is preserved here in the trace entry.
    """
    summary = []
    for r in results:
        summary.append({
            "chunk_id": r.get("chunk_id"),
            "rrf_score": r.get("rrf_score"),
            "rrf_rank": r.get("rrf_rank"),
            "semantic_rank": r.get("semantic_rank"),
            "bm25_rank": r.get("bm25_rank"),
            "bm25_score": r.get("bm25_score"),
            "distance": r.get("distance"),
        })
    return summary


def make_hybrid_qa_node(
    retriever: RetrieverLike,
    num_results: int = DEFAULT_NUM_RESULTS,
    sql_executor: Optional["SqlExecutorLike"] = None,
):
    """Factory: create a LangGraph hybrid-Q&A node bound to a retriever.

    Args:
        retriever: anything exposing search(query, num_results, doc_ids) —
                   the real LakebaseRetriever in production, a fake in tests.
        num_results: top-k passed to the retriever.
        sql_executor: optional executor for canonical-document enumeration,
                   used to resolve requested document names to doc_ids (D2 fix:
                   doc-specific questions must retrieve ONLY from the named
                   documents; without an executor the behavior is unchanged
                   corpus-wide top-k retrieval).
    """

    def hybrid_qa_node(state: dict) -> dict:
        query: str = state.get("user_query", "")
        intent: Optional[Intent] = state.get("intent")
        # Fix 011 (2026-10-30, Stage E): on a follow-up turn the planner may
        # carry a COMBINED search query (previous question + follow-up
        # phrasing) when it resolved the follow-up to the previous turn's
        # document scope.  The raw follow-up text alone ('what about the
        # limits?') is not a usable retrieval query — it matches nothing.
        if intent is not None and getattr(intent, "search_query", None):
            query = str(intent.search_query)
        doc_scope = list(intent.target_doc_names) if intent and intent.target_doc_names else []

        # ── D2 fix (2026-10-29): resolve the requested document scope to
        # doc_ids and restrict retrieval to those documents BEFORE fusion.
        # A requested scope that resolves to NOTHING is a clarification
        # (never a silent corpus-wide search — same rule as clause_discovery).
        doc_ids: Optional[list[str]] = None
        resolution_note: Optional[str] = None
        if doc_scope and sql_executor is not None:
            try:
                all_canonical = sql_executor.execute(_CANONICAL_DOCS_SQL, {})
            except Exception as exc:
                return {
                    "error": ("Hybrid Q&A document enumeration failed: " + str(exc))[:300],
                    "evidence": [],
                    "trace": [
                        make_trace(
                            "hybrid_qa",
                            action="failed",
                            route="hybrid_qa",
                            reason=str(exc)[:300],
                            step="enumerate_documents",
                        )
                    ],
                }
            resolved, ambiguous, unresolved = _resolve_doc_names(
                sql_executor, all_canonical, doc_scope,
            )
            if resolved:
                doc_ids = [str(d.get("doc_id") or "") for d in resolved]
                notes: list[str] = []
                if unresolved:
                    notes.append(
                        "Could not find documents matching: "
                        + ", ".join(u["requested_name"] for u in unresolved) + "."
                    )
                for a in ambiguous:
                    notes.append(
                        f"Name '{a['requested_name']}' is ambiguous — "
                        f"matches: {', '.join(a['matching_files'])}."
                    )
                resolution_note = " ".join(notes) or None
            else:
                msg_parts: list[str] = []
                if unresolved:
                    msg_parts.append(
                        "Could not find documents matching: "
                        + ", ".join(u["requested_name"] for u in unresolved) + "."
                    )
                for a in ambiguous:
                    msg_parts.append(
                        f"Name '{a['requested_name']}' is ambiguous — "
                        f"matches: {', '.join(a['matching_files'])}."
                    )
                msg = " ".join(msg_parts) or "Could not resolve the requested document names."
                result = {
                    "stub": False,
                    "capability": "hybrid_qa",
                    "doc_scope": doc_scope,
                    "limitations": [msg],
                    "answer_text": msg,
                }
                return {
                    "evidence": [],
                    "metadata_result": result,
                    "trace": [
                        make_trace(
                            "hybrid_qa",
                            action="clarification",
                            route="hybrid_qa",
                            reason=msg[:200],
                            doc_scope=doc_scope,
                            unresolved=[u["requested_name"] for u in unresolved],
                            ambiguous=[a["requested_name"] for a in ambiguous],
                        )
                    ],
                }

        try:
            if doc_ids is not None:
                results = retriever.search(query, num_results=num_results, doc_ids=doc_ids)
            else:
                results = retriever.search(query, num_results=num_results)
        except Exception as exc:  # safe failure — recorded, never falls through
            return {
                "error": f"Hybrid retrieval failed: {exc}",
                "evidence": [],
                "trace": [
                    make_trace(
                        "hybrid_qa",
                        action="failed",
                        route="hybrid_qa",
                        reason=str(exc)[:300],
                    )
                ],
            }

        evidence = [search_result_to_evidence(r) for r in results]
        details: dict[str, Any] = {
            "num_requested": num_results,
            "num_returned": len(results),
            "doc_scope": doc_scope,  # requested document names (user wording)
            "doc_ids": doc_ids,       # resolved document scope (D2), None = corpus-wide
            "resolution_note": resolution_note,
            "scores": _score_summary(results),
        }
        return {
            "evidence": evidence,
            "trace": [
                make_trace(
                    "hybrid_qa",
                    action="retrieved",
                    route="hybrid_qa",
                    reason=f"{len(evidence)} chunks retrieved",
                    **details,
                )
            ],
        }

    return hybrid_qa_node
