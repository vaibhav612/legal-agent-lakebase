"""Unit tests for Milestone 3 - Clause Discovery node in LangGraph.

Covers full-corpus enumeration, document-specific retrieval (with the
new doc_ids filter applied inside the retrieval query), correct
document/evidence association, section/page/chunk-ID/score preservation,
per-document matches and no-match results, coverage calculations, skipped
and failed documents, incomplete searches from execution limits, ambiguous
or unresolved document names, bounded evidence, no-match caveat (no proof
of clause absence), graph routing, error handling, and preservation of
existing hybrid QA, structured_lookup, corpus_overview, and comparison paths.

No LLM endpoint, Lakebase, SQL warehouse, or application deployment:
the graph is built with a FakeSqlExecutor and a FakeRetriever.
"""

from __future__ import annotations

import os
import sys

# Add the app folder to sys.path so 'workflow' is importable
_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

import pytest

from workflow.state import Intent
from workflow.classifier import RuleBasedIntentClassifier
from workflow.clause_discovery import (
    MAX_DISCOVERY_DOCS,
    MAX_TOTAL_EVIDENCE,
    NO_MATCH_CAVEAT,
    CANDIDATE_CAVEAT,
    make_clause_discovery_node,
)
from workflow.graph import build_graph
from workflow.stubs import clause_discovery_stub


DISCOVERY_QUERY = "Find all force majeure clauses across all contracts"
DOC_SPECIFIC_QUERY = "Find indemnification provisions in the Armstrong agreement"


# -- Fakes (no network, no warehouse, no LLM) ---------------------------

class FakeSqlExecutor:
    """Fake SqlExecutorLike: records calls and returns deterministic rows.

    Dispatches on SQL content: the canonical-docs query returns doc_rows;
    corpus_overview count/class queries return count_rows/class_rows.
    """

    def __init__(self, doc_rows=None, error=None, count_rows=None,
                 class_rows=None):
        self.doc_rows = list(doc_rows or [])
        self.count_rows = list(count_rows or [])
        self.class_rows = list(class_rows or [])
        self.error = error
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append({"sql": sql, "params": dict(params or {})})
        if self.error is not None:
            raise self.error
        # Corpus overview: count query
        if "total_documents" in sql and "document_class" not in sql:
            return list(self.count_rows)
        # Corpus overview: class distribution
        if "document_class" in sql and "GROUP BY" in sql:
            return list(self.class_rows)
        # Clause discovery: canonical doc enumeration
        if "doc_id" in sql and "file_name" in sql and "is_canonical" in sql:
            return list(self.doc_rows)
        # Field coverage (corpus overview)
        if "total_canonical_docs" in sql:
            return list(self.count_rows)
        return []


class FakeRetriever:
    """Fake retriever supporting the doc_ids parameter.

    When doc_ids is provided, only returns results whose doc_id is in the set.
    This mirrors the real LakebaseRetriever's document filtering.
    """

    def __init__(self, results=None, error_doc_ids=None):
        # results: list of dicts with doc_id, chunk_id, chunk_text, etc.
        self.results = list(results or [])
        self.error_doc_ids = set(error_doc_ids or [])  # doc_ids that raise on search
        self.calls = []

    def search(self, query, num_results=5, doc_ids=None):
        self.calls.append({
            "query": query,
            "num_results": num_results,
            "doc_ids": list(doc_ids) if doc_ids else None,
        })
        if doc_ids is not None:
            for did in doc_ids:
                if did in self.error_doc_ids:
                    raise RuntimeError(f"search failed for doc {did}")
            filtered = [r for r in self.results if r.get("doc_id") in set(doc_ids)]
            return filtered[:num_results]
        return self.results[:num_results]


def _make_chunk(doc_id, chunk_id, doc_name=None, section_path=None,
                page_start=None, page_end=None, text=None, score=None):
    """Build a single fake retrieval result dict."""
    return {
        "doc_id": doc_id,
        "doc_name": doc_name or f"{doc_id}_file.pdf",
        "chunk_id": chunk_id,
        "chunk_text": text or f"Clause text for {chunk_id}",
        "section_path": section_path,
        "page_start": page_start,
        "page_end": page_end,
        "rrf_score": score,
        "rrf_rank": 0,
        "bm25_rank": None,
        "semantic_rank": None,
        "bm25_score": None,
        "distance": None,
    }


def _make_doc_rows(n):
    """Build n canonical-doc rows for the FakeSqlExecutor."""
    return [
        {"doc_id": f"doc_{i}", "file_name": f"Contract_{i}.pdf"}
        for i in range(n)
    ]


def _make_intent(label="clause_discovery", scope="all", clause_pattern="force majeure",
                  target_doc_names=None, needs_evidence=True):
    return Intent(
        label=label,
        scope=scope,
        clause_pattern=clause_pattern,
        target_doc_names=target_doc_names or [],
        coverage_requirement="exhaustive",
        needs_evidence=needs_evidence,
    )


# ── 1. Full-corpus enumeration ──────────────────────────────────────────

class TestFullCorpus:

    def test_all_canonical_docs_enumerated_and_searched(self):
        """Every canonical doc is enumerated and searched at least once."""
        docs = _make_doc_rows(3)
        retriever = FakeRetriever(results=[
            _make_chunk("doc_0", "c0"),
            _make_chunk("doc_1", "c1"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor, max_per_doc=5)

        state = {
            "user_query": DISCOVERY_QUERY,
            "intent": _make_intent(clause_pattern="force majeure"),
        }
        result = node(state)

        assert len(executor.calls) == 1
        assert "is_canonical" in executor.calls[0]["sql"]
        assert len(retriever.calls) == 3
        # Each doc searched individually with doc_ids filter
        for call in retriever.calls:
            assert call["doc_ids"] is not None
            assert len(call["doc_ids"]) == 1

    def test_coverage_100_pct_when_all_searched(self):
        """All docs searched → 100% processing coverage."""
        docs = _make_doc_rows(5)
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        coverage = result["metadata_result"]["coverage"]
        assert coverage["eligible_documents"] == 5
        assert coverage["documents_searched"] == 5
        assert coverage["documents_skipped"] == 0
        assert coverage["processing_coverage_pct"] == 100.0

    def test_empty_corpus_returns_zero_coverage(self):
        """No canonical docs → zero coverage with a limitation."""
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=[])
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        coverage = result["metadata_result"]["coverage"]
        assert coverage["eligible_documents"] == 0
        assert coverage["documents_searched"] == 0
        assert coverage["processing_coverage_pct"] == 0.0
        assert "No canonical documents" in result["metadata_result"]["answer_text"]


# ── 2. Document-specific retrieval ─────────────────────────────────────

class TestDocumentSpecific:

    def test_doc_ids_filter_passed_to_retriever(self):
        """The retriever receives doc_ids=[single_id] for each doc."""
        docs = _make_doc_rows(3)
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        node({"user_query": DISCOVERY_QUERY,
              "intent": _make_intent()})

        for call in retriever.calls:
            assert call["doc_ids"] is not None
            assert len(call["doc_ids"]) == 1

    def test_specific_doc_resolution(self):
        """target_doc_names are resolved against file_name metadata."""
        docs = [
            {"doc_id": "doc_a", "file_name": "Armstrong_Agreement.pdf"},
            {"doc_id": "doc_b", "file_name": "Bellring_Contract.pdf"},
            {"doc_id": "doc_c", "file_name": "Cerence_Agreement.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({
            "user_query": DOC_SPECIFIC_QUERY,
            "intent": _make_intent(
                scope="specific_docs",
                target_doc_names=["Armstrong"],
                clause_pattern="indemnification",
            ),
        })

        # Only doc_a should be searched
        assert len(retriever.calls) == 1
        assert retriever.calls[0]["doc_ids"] == ["doc_a"]
        coverage = result["metadata_result"]["coverage"]
        assert coverage["eligible_documents"] == 1
        assert coverage["documents_searched"] == 1

    def test_unresolved_doc_name_returns_clarification(self):
        """Unresolved name → clarification, no search, no broadening."""
        docs = _make_doc_rows(3)
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({
            "user_query": "Find clauses in the Nonexistent agreement",
            "intent": _make_intent(
                scope="specific_docs",
                target_doc_names=["Nonexistent"],
            ),
        })

        assert len(retriever.calls) == 0
        assert result["evidence"] == []
        meta = result["metadata_result"]
        assert "Could not find" in meta["answer_text"]
        assert "Nonexistent" in meta["unresolved_names"]

    def test_ambiguous_doc_name_reported(self):
        """Ambiguous name → reported, not silently searched."""
        docs = [
            {"doc_id": "doc_a", "file_name": "Armstrong_Supply.pdf"},
            {"doc_id": "doc_b", "file_name": "Armstrong_Service.pdf"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({
            "user_query": "Find clauses in the Armstrong agreement",
            "intent": _make_intent(
                scope="specific_docs",
                target_doc_names=["Armstrong"],
            ),
        })

        meta = result["metadata_result"]
        assert len(meta["ambiguous_names"]) == 1
        assert meta["ambiguous_names"][0]["requested_name"] == "Armstrong"
        assert len(retriever.calls) == 0

    def test_partial_resolution_searches_resolved_reports_unresolved(self):
        """Some names resolved, some not → search resolved, report unresolved."""
        docs = [
            {"doc_id": "doc_a", "file_name": "Armstrong_Agreement.pdf"},
            {"doc_id": "doc_b", "file_name": "Bellring_Contract.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({
            "user_query": "Find clauses in Armstrong and Nonexistent",
            "intent": _make_intent(
                scope="specific_docs",
                target_doc_names=["Armstrong", "Nonexistent"],
            ),
        })

        meta = result["metadata_result"]
        assert len(retriever.calls) == 1  # only Armstrong searched
        assert "Nonexistent" in meta["unresolved_names"]
        assert any("Nonexistent" in lim for lim in meta["limitations"])


# ── 3. Evidence association and metadata preservation ───────────────────

class TestEvidenceMetadata:

    def test_evidence_carries_correct_doc_id_and_name(self):
        """Evidence items carry the doc_id and doc_name from the retriever."""
        docs = [{"doc_id": "doc_a", "file_name": "Alpha.pdf"}]
        chunk = _make_chunk("doc_a", "c0", doc_name="Alpha.pdf")
        retriever = FakeRetriever(results=[chunk])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        ev = result["evidence"]
        assert len(ev) == 1
        assert ev[0].doc_id == "doc_a"
        assert ev[0].doc_name == "Alpha.pdf"

    def test_section_page_chunk_id_score_preserved(self):
        """All metadata fields are preserved from retrieval to EvidenceItem."""
        docs = [{"doc_id": "doc_a", "file_name": "Alpha.pdf"}]
        chunk = _make_chunk(
            "doc_a", "chunk_42", doc_name="Alpha.pdf",
            section_path="Section 5.2", page_start=12, page_end=13,
            text="Force majeure clause text", score=0.035,
        )
        retriever = FakeRetriever(results=[chunk])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        ev = result["evidence"][0]
        assert ev.chunk_id == "chunk_42"
        assert ev.section_path == "Section 5.2"
        assert ev.page_start == 12
        assert ev.page_end == 13
        assert ev.supporting_text == "Force majeure clause text"
        assert ev.retrieval_score == 0.035
        assert ev.retrieval_method == "hybrid"


# ── 4. Per-document matches and no-match results ───────────────────────

class TestPerDocumentStatus:

    def test_with_matches_and_no_matches(self):
        """Docs with results → searched_with_matches; without → searched_no_matches."""
        docs = _make_doc_rows(3)
        retriever = FakeRetriever(results=[
            _make_chunk("doc_0", "c0"),
            _make_chunk("doc_2", "c2"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        per_doc = result["metadata_result"]["per_document_status"]
        statuses = {d["doc_id"]: d["status"] for d in per_doc}
        assert statuses["doc_0"] == "searched_with_matches"
        assert statuses["doc_1"] == "searched_no_matches"
        assert statuses["doc_2"] == "searched_with_matches"

    def test_coverage_with_matches_and_no_matches(self):
        """Coverage counts match per-document statuses."""
        docs = _make_doc_rows(4)
        retriever = FakeRetriever(results=[
            _make_chunk("doc_0", "c0"),
            _make_chunk("doc_2", "c2"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        coverage = result["metadata_result"]["coverage"]
        assert coverage["documents_with_matches"] == 2
        assert coverage["documents_no_matches"] == 2


# ── 5. Skipped and failed documents ────────────────────────────────────

class TestSkippedAndFailed:

    def test_execution_limit_causes_skips(self):
        """max_docs < eligible → some docs skipped, coverage < 100%."""
        docs = _make_doc_rows(5)
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor, max_docs=3)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        coverage = result["metadata_result"]["coverage"]
        assert coverage["eligible_documents"] == 5
        assert coverage["documents_searched"] == 3
        assert coverage["documents_skipped"] == 2
        assert coverage["processing_coverage_pct"] == 60.0
        assert any("skipped" in lim for lim in result["metadata_result"]["limitations"])

    def test_failed_document_recorded_not_propagated(self):
        """Retrieval failure for one doc → status=failed, others continue."""
        docs = _make_doc_rows(3)
        retriever = FakeRetriever(
            results=[_make_chunk("doc_0", "c0")],
            error_doc_ids={"doc_1"},
        )
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        per_doc = result["metadata_result"]["per_document_status"]
        statuses = {d["doc_id"]: d["status"] for d in per_doc}
        assert statuses["doc_1"] == "failed"
        assert statuses["doc_0"] == "searched_with_matches"
        assert statuses["doc_2"] == "searched_no_matches"
        coverage = result["metadata_result"]["coverage"]
        assert coverage["documents_failed"] == 1

    def test_failed_doc_not_described_as_no_match(self):
        """A failed doc has status=failed, NOT searched_no_matches."""
        docs = _make_doc_rows(1)
        retriever = FakeRetriever(error_doc_ids={"doc_0"})
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        per_doc = result["metadata_result"]["per_document_status"]
        assert per_doc[0]["status"] == "failed"
        assert "error" in per_doc[0]


# ── 6. Incomplete search from execution limits ─────────────────────────

class TestExecutionLimits:

    def test_max_docs_zero_searches_nothing(self):
        """max_docs=0 → all docs skipped, 0% coverage."""
        docs = _make_doc_rows(3)
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor, max_docs=0)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        coverage = result["metadata_result"]["coverage"]
        assert coverage["documents_searched"] == 0
        assert coverage["documents_skipped"] == 3
        assert coverage["processing_coverage_pct"] == 0.0

    def test_evidence_truncated_at_max(self):
        """Total evidence beyond MAX_TOTAL_EVIDENCE is truncated."""
        docs = _make_doc_rows(3)
        # 100 chunks per doc = 300 total, exceeds MAX_TOTAL_EVIDENCE=200
        chunks = [_make_chunk(f"doc_{i}", f"c_{i}_{j}")
                  for i in range(3) for j in range(100)]
        retriever = FakeRetriever(results=chunks)
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor, max_per_doc=100)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        assert len(result["evidence"]) <= MAX_TOTAL_EVIDENCE
        assert any("truncated" in lim.lower()
                   for lim in result["metadata_result"]["limitations"])


# ── 7. No-match caveat ─────────────────────────────────────────────────

class TestNoMatchCaveat:

    def test_no_match_caveat_present_when_no_matches(self):
        """NO_MATCH_CAVEAT appears when any doc has no candidate matches."""
        docs = _make_doc_rows(2)
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        answer = result["metadata_result"]["answer_text"]
        assert NO_MATCH_CAVEAT in answer

    def test_no_match_caveat_absent_when_all_match(self):
        """NO_MATCH_CAVEAT absent when every searched doc has matches."""
        docs = _make_doc_rows(2)
        retriever = FakeRetriever(results=[
            _make_chunk("doc_0", "c0"),
            _make_chunk("doc_1", "c1"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        answer = result["metadata_result"]["answer_text"]
        assert NO_MATCH_CAVEAT not in answer

    def test_candidate_caveat_present_when_candidates_exist(self):
        """CANDIDATE_CAVEAT appears when candidate passages are returned."""
        docs = _make_doc_rows(1)
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        answer = result["metadata_result"]["answer_text"]
        assert CANDIDATE_CAVEAT in answer

    def test_no_proof_of_absence_language(self):
        """The answer must not claim that a clause does not exist."""
        docs = _make_doc_rows(1)
        retriever = FakeRetriever()  # no results → no matches
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        answer = result["metadata_result"]["answer_text"]
        assert "does not exist" not in answer.lower().replace("not proof that the clause does not exist", "")


# ── 8. Error handling ─────────────────────────────────────────────────

class TestErrorHandling:

    def test_enumeration_error_recorded_not_propagated(self):
        """SQL executor failure during enumeration → error in state, no raise."""
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(error=RuntimeError("DB connection lost"))
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        assert "error" in result
        assert "enumeration failed" in result["error"].lower()
        assert result["evidence"] == []
        assert len(retriever.calls) == 0

    def test_trace_recorded_on_error(self):
        """Error trace entry has action=failed and the reason."""
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(error=RuntimeError("timeout"))
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        trace = result["trace"]
        assert trace[0]["action"] == "failed"
        assert "timeout" in trace[0]["reason"]

    def test_single_doc_failure_does_not_crash_node(self):
        """A failure on one doc does not prevent processing of others, and
        the retrieval failure is surfaced as a state error (Stage A
        remediation 2026-10-29: a genuine retrieval failure must not
        masquerade as a clean no-matches or no-evidence-abstention outcome)."""
        docs = _make_doc_rows(3)
        retriever = FakeRetriever(
            results=[_make_chunk("doc_0", "c0")],
            error_doc_ids={"doc_1"},
        )
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        result = node({"user_query": DISCOVERY_QUERY,
                       "intent": _make_intent()})

        # Node completed; other docs were still processed
        coverage = result["metadata_result"]["coverage"]
        assert coverage["documents_failed"] == 1
        assert coverage["documents_searched"] == 2  # doc_0 and doc_2 searched
        # The failure is surfaced distinctly, not masked as success
        assert "error" in result
        assert "failed for 1 of 3" in result["error"]
        assert "clause-discovery results are incomplete" in result["error"]


# ── 9. Graph routing ──────────────────────────────────────────────────

class TestGraphRouting:

    def test_real_node_used_when_both_deps_injected(self):
        """When retriever + sql_executor injected, real clause_discovery runs."""
        docs = _make_doc_rows(2)
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({"user_query": DISCOVERY_QUERY,
                               "conversation_history": []})

        meta = result.get("metadata_result", {})
        assert not meta.get("stub")
        assert meta.get("capability") == "clause_discovery"
        assert len(retriever.calls) == 2

    def test_stub_used_when_deps_missing(self):
        """Without retriever or sql_executor, the stub is used."""
        graph = build_graph(classifier=RuleBasedIntentClassifier())

        result = graph.invoke({"user_query": DISCOVERY_QUERY,
                               "conversation_history": []})

        meta = result.get("metadata_result", {})
        assert meta.get("stub") is True
        assert meta.get("capability") == "clause_discovery"

    def test_unresolvable_doc_name_user_sees_clarification(self):
        """D11 completion (2026-10-29): a clause-discovery question naming a
        document that does not exist must ask for clarification — no
        corpus-wide retrieval runs — and the clarification question reaches
        the user instead of the generic abstention message."""
        from workflow.answers import ABSTAIN_MESSAGE

        docs = [{"doc_id": "doc_0", "file_name": "Alpha_MSA.pdf"},
                {"doc_id": "doc_1", "file_name": "Beta_MSA.pdf"}]
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "Find force majeure clauses in the Zephyr agreement",
            "conversation_history": [],
        })

        assert len(retriever.calls) == 0, "no retrieval may run for an unresolvable name"
        answer = result.get("answer", "")
        assert "Could not find documents matching" in answer
        assert answer != ABSTAIN_MESSAGE

    def test_clause_discovery_trace_order(self):
        """Trace: query_planner → clause_discovery → evidence_validation → final_answer."""
        docs = _make_doc_rows(1)
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({"user_query": DISCOVERY_QUERY,
                               "conversation_history": []})

        nodes = [t["node"] for t in result.get("trace", [])]
        assert "query_planner" in nodes
        assert "clause_discovery" in nodes
        assert "evidence_validation" in nodes
        assert "final_answer" in nodes

    def test_answer_text_present(self):
        """Final answer uses the clause_discovery answer_text."""
        docs = _make_doc_rows(1)
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({"user_query": DISCOVERY_QUERY,
                               "conversation_history": []})

        answer = result.get("answer", "")
        assert "Clause discovery" in answer
        assert "force majeure" in answer.lower()

    def test_source_refs_built_from_evidence(self):
        """Source refs are built from evidence for clause discovery results."""
        docs = _make_doc_rows(1)
        chunk = _make_chunk("doc_0", "c0", doc_name="Alpha.pdf",
                           section_path="Sec 5", page_start=10, page_end=11)
        retriever = FakeRetriever(results=[chunk])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({"user_query": DISCOVERY_QUERY,
                               "conversation_history": []})

        refs = result.get("source_refs", [])
        assert len(refs) >= 1
        assert refs[0].doc_name == "Alpha.pdf"
        assert refs[0].section_path == "Sec 5"


# ── 10. Path preservation (regression) ─────────────────────────────────

class TestPathPreservation:

    def test_hybrid_qa_path_preserved(self):
        """Hybrid Q&A queries still route to hybrid_qa, not clause_discovery."""
        retriever = FakeRetriever(results=[_make_chunk("doc_0", "c0")])
        executor = FakeSqlExecutor(doc_rows=_make_doc_rows(1))
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "What are the termination conditions in this contract?",
            "conversation_history": [],
        })

        intent = result.get("intent")
        assert intent.label == "hybrid_qa"
        nodes = [t["node"] for t in result.get("trace", [])]
        assert "hybrid_qa" in nodes
        assert "clause_discovery" not in nodes

    def test_corpus_overview_path_preserved(self):
        """Corpus overview queries still route to corpus_overview."""
        executor = FakeSqlExecutor(doc_rows=_make_doc_rows(1))
        # Also provide count_rows for the corpus overview node
        executor.count_rows = [{"total_documents": 1, "canonical_documents": 1}]
        executor.class_rows = []
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "How many documents are in the corpus?",
            "conversation_history": [],
        })

        intent = result.get("intent")
        assert intent.label == "corpus_overview"
        nodes = [t["node"] for t in result.get("trace", [])]
        assert "corpus_overview" in nodes

    def test_structured_lookup_path_preserved(self):
        """Structured lookup queries still route to structured_lookup."""
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=_make_doc_rows(1))
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "What is the governing law of the Bellring agreement?",
            "conversation_history": [],
        })

        intent = result.get("intent")
        assert intent.label == "structured_lookup"

    def test_comparison_stub_path_preserved(self):
        """Comparison queries still route to comparison_stub."""
        graph = build_graph(classifier=RuleBasedIntentClassifier())

        result = graph.invoke({
            "user_query": "Compare termination clauses between Alpha and Beta agreements",
            "conversation_history": [],
        })

        intent = result.get("intent")
        assert intent.label == "comparison"
        nodes = [t["node"] for t in result.get("trace", [])]
        assert "comparison_stub" in nodes

    def test_guard_path_preserved(self):
        """Unsupported queries still route to guard_node."""
        graph = build_graph(classifier=RuleBasedIntentClassifier())

        result = graph.invoke({
            "user_query": "show me all text from all documents",
            "conversation_history": [],
        })

        intent = result.get("intent")
        assert intent.label == "unsupported"
        nodes = [t["node"] for t in result.get("trace", [])]
        # Unsupported with clarification_needed routes to clarification_node;
        # unsupported without clarification_needed routes to guard_node.
        assert "guard_node" in nodes or "clarification" in nodes, \
            f"Expected guard_node or clarification in trace, got nodes: {nodes}"

    def test_clause_discovery_without_executor_uses_stub(self):
        """Retriever but no executor → stub (clause_discovery needs both)."""
        retriever = FakeRetriever()
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
        )

        result = graph.invoke({"user_query": DISCOVERY_QUERY,
                               "conversation_history": []})

        meta = result.get("metadata_result", {})
        assert meta.get("stub") is True

    def test_clause_discovery_without_retriever_uses_stub(self):
        """Executor but no retriever → stub (clause_discovery needs both)."""
        executor = FakeSqlExecutor(doc_rows=_make_doc_rows(1))
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            sql_executor=executor,
        )

        result = graph.invoke({"user_query": DISCOVERY_QUERY,
                               "conversation_history": []})

        meta = result.get("metadata_result", {})
        assert meta.get("stub") is True


# ── 11. Bounded evidence and deduplication ────────────────────────────

class TestBoundedEvidence:

    def test_per_doc_top_k_respected(self):
        """The retriever is called with num_results=max_per_doc for each doc."""
        docs = _make_doc_rows(2)
        retriever = FakeRetriever(results=[
            _make_chunk("doc_0", f"c0_{i}") for i in range(10)
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor, max_per_doc=3)

        node({"user_query": DISCOVERY_QUERY,
              "intent": _make_intent()})

        for call in retriever.calls:
            assert call["num_results"] == 3

    def test_evidence_dedup_by_evidence_validation(self):
        """Duplicate chunk_ids across docs are deduplicated by evidence_validation."""
        from workflow.validation import make_evidence_validation_node
        from workflow.clause_discovery import search_result_to_evidence

        chunk = _make_chunk("doc_0", "dup_chunk", text="Same chunk")
        evidence = [search_result_to_evidence(chunk), search_result_to_evidence(chunk)]

        val_node = make_evidence_validation_node()
        result = val_node({"evidence": evidence})

        assert len(result["evidence"]) == 1


# ── 12. Unspecified clause pattern ────────────────────────────────────

class TestUnspecifiedPattern:

    def test_unspecified_pattern_uses_user_query(self):
        """When clause_pattern is 'unspecified', the user_query is the search query."""
        docs = _make_doc_rows(1)
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        node({
            "user_query": "Find all clauses across all contracts",
            "intent": _make_intent(clause_pattern="unspecified"),
        })

        assert retriever.calls[0]["query"] == "Find all clauses across all contracts"

    def test_specified_pattern_used_as_search_query(self):
        """When clause_pattern is specified, it is used as the search query."""
        docs = _make_doc_rows(1)
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_clause_discovery_node(retriever, executor)

        node({
            "user_query": DISCOVERY_QUERY,
            "intent": _make_intent(clause_pattern="force majeure"),
        })

        assert retriever.calls[0]["query"] == "force majeure"
