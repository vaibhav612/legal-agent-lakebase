"""Clause Discovery capability node (Capability C; spec Section 1.4).

Discovers clauses or provisions matching a pattern across the canonical
corpus (or a specific set of identified documents) using per-document hybrid
retrieval with coverage tracking.

Key design decisions (spec Section 1.4):
  1. The eligible corpus is the set of canonical documents from
     ``workspace.legal_assistant_v2.documents`` (``is_canonical = true``).
  2. For each eligible document the node runs the injected retriever with a
     document filter (``doc_ids=[doc_id]``) so that the filter is applied
     *inside* the retrieval query — before top-k — not as a post-fusion filter.
  3. Per-document top-k retrieval reduces the candidate set but does NOT
     constitute exhaustive discovery.  The reduction must be validated
     against a benchmark (Milestone 4).
  4. No relevance threshold is invented here.  All retrieved chunks are
     "candidate passages"; relevance calibration is deferred to Milestone 4.
  5. Processing coverage (documents searched / eligible) is tracked and
     reported separately from semantic recall.
  6. A retrieval miss is NEVER treated as proof that a clause does not exist.
  7. Document names are resolved against actual metadata.  Unresolved or
     ambiguous names produce a clarification or limitation, never a silent
     broadening to the full corpus.

Dependency injection follows the existing pattern:
  - *retriever*: anything exposing ``search(query, num_results, doc_ids)`` —
    the real ``LakebaseRetriever`` in production (with the new ``doc_ids``
    parameter), a fake in unit tests.
  - *sql_executor*: ``SqlExecutorLike`` (reused from structured_lookup) for
    canonical-document enumeration and document-name resolution against the
    Databricks SQL warehouse ``documents`` table.

Node contract (spec Section 2.5-C):
  Input:  user_query, intent (clause_pattern, target_doc_names, scope,
          coverage_requirement)
  Output: evidence (list[EvidenceItem]) — candidate passages preserved in
          graph state for evidence_validation → final_answer.
          metadata_result — coverage summary, per-document status, answer_text.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .state import EvidenceItem, Intent
from .traces import make_trace
from .structured_lookup import SqlExecutorLike


# ── Constants ───────────────────────────────────────────────────────────

DOCUMENTS_TABLE = "workspace.legal_assistant_v2.documents"

# Execution limits (internal constants, not user input)
MAX_DISCOVERY_DOCS: int = 56       # cap on eligible docs searched (canonical corpus size)
DEFAULT_PER_DOC_RESULTS: int = 5   # per-document top-k
MAX_TOTAL_EVIDENCE: int = 200      # total evidence items retained
MAX_NAME_PATTERN_CHARS: int = 100  # max chars of a document-name fragment

# Caveat attached when any documents have no candidate matches
NO_MATCH_CAVEAT = (
    "A retrieval miss is not proof that the clause does not exist in the "
    "document. The search may not have surfaced all relevant passages."
)

# Caveat about candidate vs confirmed
CANDIDATE_CAVEAT = (
    "Retrieved passages are candidate matches, not confirmed findings. "
    "Relevance calibration is pending (Milestone 4)."
)


# ── Retriever protocol (extended with doc_ids) ─────────────────────────

class DiscoveryRetrieverLike(Protocol):
    """Structural type of the injected retriever for clause discovery.

    Satisfied by LakebaseRetriever after the doc_ids parameter was added.
    The doc_ids parameter is optional (default None) so the existing
    RetrieverLike protocol is still satisfied.
    """

    def search(
        self,
        query: str,
        num_results: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        ...


# ── SQL queries (fixed templates; no user input interpolated) ───────────

_CANONICAL_DOCS_SQL = (
    "SELECT doc_id, file_name\n"
    "FROM " + DOCUMENTS_TABLE + "\n"
    "WHERE is_canonical = true\n"
    "ORDER BY file_name"
).strip()


# ── Helpers ─────────────────────────────────────────────────────────────

def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def search_result_to_evidence(result: dict) -> EvidenceItem:
    """Map one retrieval result row to an EvidenceItem (same field mapping
    as hybrid_qa.search_result_to_evidence, kept here for module
    self-containment).
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
    )


def _resolve_doc_names(
    executor: SqlExecutorLike,
    all_docs: list[dict],
    target_names: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Resolve user-provided document names against canonical document metadata.

    Returns ``(resolved, ambiguous, unresolved)`` where each entry in
    *resolved* is a ``{"doc_id", "file_name"}`` dict, *ambiguous* carries the
    name plus all matching file names, and *unresolved* carries the name.
    Matching is case-insensitive substring on file_name (the user typically
    provides a partial name like "Armstrong" for "Armstrong_Supply_Agreement.pdf").
    """
    resolved: list[dict] = []
    ambiguous: list[dict] = []
    unresolved: list[dict] = []

    for name in target_names:
        name_lower = name.lower().strip()[:MAX_NAME_PATTERN_CHARS]
        if not name_lower:
            unresolved.append({"requested_name": name})
            continue
        matches = [
            d for d in all_docs
            if name_lower in _to_str(d.get("file_name")).lower()
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            ambiguous.append({
                "requested_name": name,
                "matching_files": [_to_str(m.get("file_name")) for m in matches],
            })
        else:
            unresolved.append({"requested_name": name})

    # Deduplicate resolved docs by doc_id (user may name the same doc twice)
    seen_ids: set[str] = set()
    deduped_resolved: list[dict] = []
    for d in resolved:
        did = _to_str(d.get("doc_id"))
        if did not in seen_ids:
            seen_ids.add(did)
            deduped_resolved.append(d)

    return deduped_resolved, ambiguous, unresolved


def _build_answer_text(result: dict) -> str:
    """Human-readable answer summarising the clause-discovery run."""
    clause = result.get("clause_pattern", "(unspecified)")
    coverage = result.get("coverage", {})
    lines: list[str] = []

    lines.append(f'Clause discovery — pattern: "{clause}"')
    lines.append(f"- Eligible documents: {coverage.get('eligible_documents', 0)}")
    lines.append(f"- Documents searched: {coverage.get('documents_searched', 0)}")
    lines.append(f"- Documents skipped: {coverage.get('documents_skipped', 0)}")
    lines.append(f"- Documents failed: {coverage.get('documents_failed', 0)}")
    lines.append(f"- Documents with candidate matches: {coverage.get('documents_with_matches', 0)}")
    lines.append(f"- Documents with no candidate matches: {coverage.get('documents_no_matches', 0)}")
    pct = coverage.get("processing_coverage_pct", 0)
    lines.append(f"- Processing coverage: {pct}%")
    lines.append(f"- Total candidate passages: {result.get('total_candidate_passages', 0)}")

    # Per-document summary (show first few, then summary)
    per_doc = result.get("per_document_status", [])
    if per_doc:
        with_matches = [d for d in per_doc if d.get("status") == "searched_with_matches"]
        if with_matches:
            lines.append("")
            lines.append("Documents with candidate matches:")
            for d in with_matches[:10]:
                lines.append(f"  - {_to_str(d.get('doc_name'))}: {d.get('num_candidates', 0)} candidates")
            if len(with_matches) > 10:
                lines.append(f"  ... and {len(with_matches) - 10} more")

    no_matches = [d for d in per_doc if d.get("status") == "searched_no_matches"]
    if no_matches:
        lines.append("")
        lines.append(f"Documents searched with no candidate matches: {len(no_matches)}")

    failed = [d for d in per_doc if d.get("status") == "failed"]
    if failed:
        lines.append("")
        lines.append(f"Documents that failed during search: {len(failed)}")
        for d in failed[:5]:
            lines.append(f"  - {_to_str(d.get('doc_name'))}: {_to_str(d.get('error'))}")

    # Limitations
    limitations = result.get("limitations", [])
    unresolved = result.get("unresolved_names", [])
    ambiguous = result.get("ambiguous_names", [])
    if unresolved:
        lines.append("")
        lines.append(f"Unresolved document names: {', '.join(unresolved)}")
    if ambiguous:
        for a in ambiguous:
            lines.append(f"Ambiguous name '{a.get('requested_name')}': matches {', '.join(a.get('matching_files', []))}")
    if limitations:
        lines.append("")
        for lim in limitations:
            lines.append("Limitation: " + lim)

    # Caveats
    no_match_count = coverage.get("documents_no_matches", 0)
    if no_match_count > 0:
        lines.append("")
        lines.append("Note: " + NO_MATCH_CAVEAT)
    if result.get("total_candidate_passages", 0) > 0:
        lines.append("Note: " + CANDIDATE_CAVEAT)

    return "\n".join(lines)


# ── Node factory ────────────────────────────────────────────────────────

def make_clause_discovery_node(
    retriever: DiscoveryRetrieverLike,
    sql_executor: SqlExecutorLike,
    num_results: int = DEFAULT_PER_DOC_RESULTS,
    max_docs: int = MAX_DISCOVERY_DOCS,
    max_per_doc: int = DEFAULT_PER_DOC_RESULTS,
):
    """Factory: create a LangGraph clause-discovery node.

    Args:
        retriever: anything exposing search(query, num_results, doc_ids).
        sql_executor: SqlExecutorLike for canonical-doc enumeration and name
                      resolution (Databricks SQL warehouse).
        num_results: default per-document top-k (used when max_per_doc is
                     not overridden).
        max_docs: execution limit on eligible documents searched.
        max_per_doc: per-document top-k (overrides num_results for the
                     per-document search call).
    """
    per_doc_k = max_per_doc if max_per_doc else num_results

    def clause_discovery_node(state: dict) -> dict:
        intent: Optional[Intent] = state.get("intent")
        user_query = state.get("user_query", "")

        # Tolerant intent access (same pattern as corpus_overview)
        def _iget(name, default=None):
            if intent is None:
                return default
            if isinstance(intent, dict):
                return intent.get(name, default)
            return getattr(intent, name, default)

        clause_pattern = _iget("clause_pattern") or "unspecified"
        target_doc_names = list(_iget("target_doc_names") or [])
        scope = _iget("scope", "all")

        # Use the original user query for retrieval when the clause pattern
        # is unspecified (the classifier could not extract a specific pattern).
        search_query = user_query if clause_pattern == "unspecified" else clause_pattern

        # ── 1. Enumerate eligible documents ──
        try:
            all_canonical = sql_executor.execute(_CANONICAL_DOCS_SQL, {})
        except Exception as exc:
            return {
                "error": ("Clause discovery document enumeration failed: " + str(exc))[:300],
                "evidence": [],
                "trace": [make_trace(
                    "clause_discovery", action="failed", route="clause_discovery",
                    reason=str(exc)[:300], step="enumerate_documents",
                )],
            }

        if not all_canonical:
            result = {
                "stub": False,
                "capability": "clause_discovery",
                "clause_pattern": clause_pattern,
                "coverage": {
                    "eligible_documents": 0,
                    "documents_searched": 0,
                    "documents_skipped": 0,
                    "documents_failed": 0,
                    "documents_with_matches": 0,
                    "documents_no_matches": 0,
                    "processing_coverage_pct": 0.0,
                },
                "per_document_status": [],
                "total_candidate_passages": 0,
                "limitations": ["No canonical documents found in the corpus."],
                "answer_text": "No canonical documents found in the corpus.",
            }
            return {
                "evidence": [],
                "metadata_result": result,
                "trace": [make_trace(
                    "clause_discovery", action="completed", route="clause_discovery",
                    reason="no canonical documents", clause_pattern=clause_pattern,
                )],
            }

        # ── 2. Resolve document scope ──
        unresolved_names: list[str] = []
        ambiguous_names: list[dict] = []

        if target_doc_names and scope in ("specific_docs", "filtered"):
            resolved, ambiguous, unresolved = _resolve_doc_names(
                sql_executor, all_canonical, target_doc_names,
            )
            unresolved_names = [u["requested_name"] for u in unresolved]
            ambiguous_names = ambiguous

            if not resolved:
                # No names resolved at all — return clarification
                msg_parts = []
                if unresolved_names:
                    msg_parts.append(f"Could not find documents matching: {', '.join(unresolved_names)}.")
                if ambiguous_names:
                    for a in ambiguous_names:
                        msg_parts.append(
                            f"Name '{a['requested_name']}' is ambiguous — "
                            f"matches: {', '.join(a['matching_files'])}."
                        )
                msg = " ".join(msg_parts) or "Could not resolve the requested document names."
                result = {
                    "stub": False,
                    "capability": "clause_discovery",
                    "clause_pattern": clause_pattern,
                    "coverage": {
                        "eligible_documents": 0,
                        "documents_searched": 0,
                        "documents_skipped": 0,
                        "documents_failed": 0,
                        "documents_with_matches": 0,
                        "documents_no_matches": 0,
                        "processing_coverage_pct": 0.0,
                    },
                    "per_document_status": [],
                    "total_candidate_passages": 0,
                    "unresolved_names": unresolved_names,
                    "ambiguous_names": ambiguous_names,
                    "limitations": [msg],
                    "answer_text": msg,
                }
                return {
                    "evidence": [],
                    "metadata_result": result,
                    "trace": [make_trace(
                        "clause_discovery", action="clarification", route="clause_discovery",
                        reason=msg[:200], clause_pattern=clause_pattern,
                        unresolved=unresolved_names, ambiguous=ambiguous_names,
                    )],
                }
            eligible_docs = resolved
        else:
            # Full corpus: all canonical docs
            eligible_docs = [
                {"doc_id": _to_str(d.get("doc_id")), "file_name": _to_str(d.get("file_name"))}
                for d in all_canonical
            ]

        # ── 3. Apply execution limit ──
        total_eligible = len(eligible_docs)
        skipped_docs: list[dict] = []
        if total_eligible > max_docs:
            skipped_docs = eligible_docs[max_docs:]
            eligible_docs = eligible_docs[:max_docs]

        # ── 4. Per-document retrieval ──
        all_evidence: list[EvidenceItem] = []
        per_doc_status: list[dict] = []

        for doc in eligible_docs:
            doc_id = _to_str(doc.get("doc_id"))
            doc_name = _to_str(doc.get("file_name")) or doc_id
            try:
                results = retriever.search(
                    search_query,
                    num_results=per_doc_k,
                    doc_ids=[doc_id],
                )
                evidence = [search_result_to_evidence(r) for r in (results or [])]
                all_evidence.extend(evidence)
                per_doc_status.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "status": "searched_with_matches" if evidence else "searched_no_matches",
                    "num_candidates": len(evidence),
                })
            except Exception as exc:
                per_doc_status.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "status": "failed",
                    "error": str(exc)[:200],
                    "num_candidates": 0,
                })

        # ── 5. Apply total evidence cap ──
        evidence_truncated = False
        if len(all_evidence) > MAX_TOTAL_EVIDENCE:
            all_evidence = all_evidence[:MAX_TOTAL_EVIDENCE]
            evidence_truncated = True

        # ── 6. Compute coverage ──
        searched = sum(1 for s in per_doc_status if s["status"].startswith("searched"))
        failed = sum(1 for s in per_doc_status if s["status"] == "failed")
        skipped = len(skipped_docs)
        with_matches = sum(1 for s in per_doc_status if s["status"] == "searched_with_matches")
        no_matches = sum(1 for s in per_doc_status if s["status"] == "searched_no_matches")
        # Total eligible includes skipped (they were eligible but not searched)
        total_for_coverage = total_eligible
        coverage_pct = round(searched / total_for_coverage * 100, 1) if total_for_coverage > 0 else 0.0

        # ── 7. Build result ──
        limitations: list[str] = []
        if skipped > 0:
            limitations.append(
                f"{skipped} document(s) skipped (execution limit: {max_docs}). "
                f"Processing coverage is incomplete."
            )
        if evidence_truncated:
            limitations.append(
                f"Evidence truncated to {MAX_TOTAL_EVIDENCE} items. "
                f"Some candidate passages may not be shown."
            )
        if unresolved_names:
            limitations.append(
                f"Unresolved document names not searched: {', '.join(unresolved_names)}."
            )
        if ambiguous_names:
            for a in ambiguous_names:
                limitations.append(
                    f"Ambiguous name '{a['requested_name']}' — "
                    f"matches: {', '.join(a['matching_files'])}. Not searched."
                )

        result = {
            "stub": False,
            "capability": "clause_discovery",
            "clause_pattern": clause_pattern,
            "coverage": {
                "eligible_documents": total_for_coverage,
                "documents_searched": searched,
                "documents_skipped": skipped,
                "documents_failed": failed,
                "documents_with_matches": with_matches,
                "documents_no_matches": no_matches,
                "processing_coverage_pct": coverage_pct,
            },
            "per_document_status": per_doc_status,
            "total_candidate_passages": len(all_evidence),
            "unresolved_names": unresolved_names,
            "ambiguous_names": ambiguous_names,
            "limitations": limitations,
        }
        result["answer_text"] = _build_answer_text(result)

        # Stage A remediation (2026-10-29): a genuine retrieval failure must
        # not masquerade as "searched, no matches" or a no-evidence abstention
        # (spec 2.8).  Surface it as a state error so evidence_validation
        # passes it through and final_answer reports the failure distinctly;
        # the per-document 'failed' statuses remain in the trace and coverage.
        node_error = None
        if failed > 0:
            node_error = (
                f"Document search failed for {failed} of {total_for_coverage} "
                f"eligible document(s); clause-discovery results are incomplete. "
                f"Please try again later."
            )

        trace_details = {
            "clause_pattern": clause_pattern,
            "scope": scope,
            "eligible": total_for_coverage,
            "searched": searched,
            "skipped": skipped,
            "failed": failed,
            "with_matches": with_matches,
            "no_matches": no_matches,
            "coverage_pct": coverage_pct,
            "total_candidates": len(all_evidence),
        }

        return {
            "evidence": all_evidence,
            "metadata_result": result,
            "error": node_error,
            "trace": [make_trace(
                "clause_discovery",
                action="completed",
                route="clause_discovery",
                reason=(
                    f"{searched}/{total_for_coverage} docs searched, "
                    f"{with_matches} with matches, {len(all_evidence)} candidates"
                ),
                **trace_details,
            )],
        }

    return clause_discovery_node
