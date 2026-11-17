"""Unit tests for Milestone 3 - Contract Comparison node in LangGraph.

Covers two-document and multi-document comparisons, document resolution
and ambiguous names, MAX_COMPARISON_DOCS enforcement, independent
document-filtered retrieval, correct evidence attribution, comparison by
the requested aspect, evidence and citations preserved per agreement,
missing or uncertain evidence, partial retrieval failures and coverage
reporting, no-match not treated as proof of clause absence, grounded
comparison output, correct graph routing and error traces, and
preservation of existing capability and guard paths.

No LLM endpoint, Lakebase, SQL warehouse, or application deployment:
the graph is built with a FakeSqlExecutor and a FakeRetriever.
"""

from __future__ import annotations

import os
import sys

_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

import pytest

from workflow.state import Intent
from workflow.classifier import RuleBasedIntentClassifier
from workflow.comparison import (
    MAX_TOTAL_EVIDENCE,
    make_comparison_node,
)
from workflow.graph import build_graph


COMPARISON_QUERY = "Compare termination clauses between Armstrong and Bellring"
MULTI_DOC_QUERY = "Compare governing law across Armstrong, Bellring, and Cerence"


# -- Fakes (no network, no warehouse, no LLM) ---------------------------

class FakeSqlExecutor:
    """Fake SqlExecutorLike: records calls and returns deterministic rows.

    Dispatches on SQL content: canonical-docs query returns doc_rows;
    field-query returns field_rows.
    """

    def __init__(self, doc_rows=None, field_rows=None, error=None):
        self.doc_rows = list(doc_rows or [])
        self.field_rows = list(field_rows or [])
        self.error = error
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append({"sql": sql, "params": dict(params or {})})
        if self.error is not None:
            raise self.error
        if "doc_id" in sql and "file_name" in sql and "is_canonical" in sql:
            return list(self.doc_rows)
        if "value_raw" in sql and "value_status" in sql:
            return list(self.field_rows)
        return []


class FakeRetriever:
    """Fake retriever supporting the doc_ids parameter.

    When doc_ids is provided, only returns results whose doc_id is in the set.
    """

    def __init__(self, results=None, error_doc_ids=None):
        self.results = list(results or [])
        self.error_doc_ids = set(error_doc_ids or [])
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


class FakeAnswerGenerator:
    """Fake answer generator for grounded comparison narrative tests."""

    def __init__(self, response="Generated comparison narrative."):
        self.response = response
        self.calls = []

    def __call__(self, query, context_chunks):
        self.calls.append({"query": query, "chunks": context_chunks})
        return self.response


def _make_chunk(doc_id, chunk_id, doc_name=None, section_path=None,
                page_start=None, page_end=None, text=None, score=None):
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
    return [
        {"doc_id": f"doc_{i}", "file_name": f"Contract_{i}.pdf"}
        for i in range(n)
    ]


def _make_intent(label="comparison", aspect="termination clauses",
                  target_doc_names=None):
    return Intent(
        label=label,
        scope="specific_docs",
        target_doc_names=target_doc_names or ["Alpha", "Beta"],
        comparison_aspect=aspect,
        needs_evidence=True,
    )


# == 1. Two-document comparison ==

class TestTwoDocComparison:

    def test_two_docs_resolved_and_searched(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha_Agreement.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta_Contract.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a", doc_name="Alpha_Agreement.pdf"),
            _make_chunk("doc_b", "c_b", doc_name="Beta_Contract.pdf"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        assert len(retriever.calls) == 2
        comp = result["comparison_result"]
        assert comp["coverage"]["documents_compared"] == 2
        assert comp["coverage"]["documents_with_evidence"] == 2

    def test_aspect_used_as_search_query(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(
                aspect="indemnification",
                target_doc_names=["Alpha", "Beta"],
            ),
        })

        for call in retriever.calls:
            assert call["query"] == "indemnification"

    def test_doc_ids_filter_per_doc(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        assert retriever.calls[0]["doc_ids"] == ["doc_a"]
        assert retriever.calls[1]["doc_ids"] == ["doc_b"]

    def test_answer_text_present(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a"),
            _make_chunk("doc_b", "c_b"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        answer = result["comparison_result"]["answer_text"]
        assert "Comparison" in answer
        assert "Alpha.pdf" in answer
        assert "Beta.pdf" in answer


# == 2. Multi-document comparison ==

class TestMultiDocComparison:

    def test_three_docs_compared(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
            {"doc_id": "doc_c", "file_name": "Gamma.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a"),
            _make_chunk("doc_b", "c_b"),
            _make_chunk("doc_c", "c_c"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": MULTI_DOC_QUERY,
            "intent": _make_intent(
                aspect="governing law",
                target_doc_names=["Alpha", "Beta", "Gamma"],
            ),
        })

        comp = result["comparison_result"]
        assert comp["coverage"]["documents_compared"] == 3
        assert len(retriever.calls) == 3

    def test_per_doc_findings_list(self):
        docs = _make_doc_rows(3)
        retriever = FakeRetriever(results=[
            _make_chunk("doc_0", "c0"),
            _make_chunk("doc_2", "c2"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": MULTI_DOC_QUERY,
            "intent": _make_intent(
                target_doc_names=["Contract_0", "Contract_1", "Contract_2"],
            ),
        })

        findings = result["comparison_result"]["per_doc_findings"]
        assert len(findings) == 3
        statuses = {f["doc_id"]: f["status"] for f in findings}
        assert statuses["doc_0"] == "searched_with_evidence"
        assert statuses["doc_1"] == "searched_no_evidence"
        assert statuses["doc_2"] == "searched_with_evidence"


# == 3. Document resolution and ambiguous names ==

class TestDocumentResolution:

    def test_unresolved_names_reported(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare termination clauses between Nonexistent1 and Nonexistent2",
            "intent": _make_intent(target_doc_names=["Nonexistent1", "Nonexistent2"]),
        })

        comp = result["comparison_result"]
        assert len(retriever.calls) == 0
        assert "Could not find" in comp["answer_text"]
        assert "Nonexistent1" in comp["unresolved_names"]

    def test_ambiguous_names_reported(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha_Supply.pdf"},
            {"doc_id": "doc_b", "file_name": "Alpha_Service.pdf"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare termination clauses between Alpha and Alpha",
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        comp = result["comparison_result"]
        assert len(comp["ambiguous_names"]) >= 1
        assert len(retriever.calls) == 0

    def test_single_doc_returns_clarification(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare Alpha with Nonexistent",
            "intent": _make_intent(target_doc_names=["Alpha", "Nonexistent"]),
        })

        comp = result["comparison_result"]
        assert comp["coverage"]["documents_compared"] == 0
        assert "at least two" in comp["answer_text"].lower()

    def test_partial_resolution_searches_resolved(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c_a")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare Alpha, Beta, and Nonexistent",
            "intent": _make_intent(target_doc_names=["Alpha", "Beta", "Nonexistent"]),
        })

        comp = result["comparison_result"]
        assert len(retriever.calls) == 2
        assert "Nonexistent" in comp["unresolved_names"]
        assert any("Nonexistent" in lim for lim in comp["limitations"])


# == 4. MAX_COMPARISON_DOCS enforcement ==

class TestMaxDocs:

    def test_exceeds_limit_returns_scope_limitation(self):
        docs = _make_doc_rows(12)
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor, max_docs=10)

        result = node({
            "user_query": "Compare all 12 contracts",
            "intent": _make_intent(
                target_doc_names=[f"Contract_{i}" for i in range(12)],
            ),
        })

        comp = result["comparison_result"]
        assert len(retriever.calls) == 0
        assert comp["coverage"]["documents_compared"] == 0
        assert "limited to 10" in comp["answer_text"].lower()

    def test_at_limit_succeeds(self):
        docs = _make_doc_rows(10)
        retriever = FakeRetriever(results=[
            _make_chunk(f"doc_{i}", f"c_{i}") for i in range(10)
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor, max_docs=10)

        result = node({
            "user_query": "Compare all 10 contracts",
            "intent": _make_intent(
                target_doc_names=[f"Contract_{i}" for i in range(10)],
            ),
        })

        comp = result["comparison_result"]
        assert comp["coverage"]["documents_compared"] == 10


# == 5. Evidence attribution ==

class TestEvidenceAttribution:

    def test_evidence_carries_correct_doc_id(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a", doc_name="Alpha.pdf"),
            _make_chunk("doc_b", "c_b", doc_name="Beta.pdf"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        ev = result["evidence"]
        assert len(ev) == 2
        doc_ids = {e.doc_id for e in ev}
        assert doc_ids == {"doc_a", "doc_b"}

    def test_evidence_not_cross_attributed(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a", doc_name="Alpha.pdf", text="Alpha clause"),
            _make_chunk("doc_b", "c_b", doc_name="Beta.pdf", text="Beta clause"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        for e in result["evidence"]:
            if e.doc_id == "doc_a":
                assert "Alpha" in e.supporting_text
            elif e.doc_id == "doc_b":
                assert "Beta" in e.supporting_text

    def test_section_page_chunk_id_preserved(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        chunk = _make_chunk(
            "doc_a", "chunk_42", doc_name="Alpha.pdf",
            section_path="Section 5.2", page_start=12, page_end=13,
            text="Force majeure clause text", score=0.035,
        )
        retriever = FakeRetriever(results=[chunk])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        ev = [e for e in result["evidence"] if e.doc_id == "doc_a"][0]
        assert ev.chunk_id == "chunk_42"
        assert ev.section_path == "Section 5.2"
        assert ev.page_start == 12
        assert ev.page_end == 13
        assert ev.retrieval_score == 0.035


# == 6. Missing or uncertain evidence ==

class TestMissingEvidence:

    def test_no_evidence_status(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c_a")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        findings = result["comparison_result"]["per_doc_findings"]
        statuses = {f["doc_id"]: f["status"] for f in findings}
        assert statuses["doc_b"] == "searched_no_evidence"
        assert result["comparison_result"]["coverage"]["documents_no_evidence"] == 1

    def test_no_match_not_proof_of_absence(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c_a")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        answer = result["comparison_result"]["answer_text"]
        assert "not proof" in answer.lower()

    def test_failed_doc_not_no_evidence(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(
            results=[_make_chunk("doc_a", "c_a")],
            error_doc_ids={"doc_b"},
        )
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        findings = result["comparison_result"]["per_doc_findings"]
        statuses = {f["doc_id"]: f["status"] for f in findings}
        assert statuses["doc_b"] == "failed"
        assert statuses["doc_b"] != "searched_no_evidence"
        assert result["comparison_result"]["coverage"]["documents_failed"] == 1


# == 7. Partial retrieval failures ==

class TestPartialFailures:

    def test_one_doc_fails_comparison_continues(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
            {"doc_id": "doc_c", "file_name": "Gamma.pdf"},
        ]
        retriever = FakeRetriever(
            results=[_make_chunk("doc_a", "c_a"), _make_chunk("doc_c", "c_c")],
            error_doc_ids={"doc_b"},
        )
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": MULTI_DOC_QUERY,
            "intent": _make_intent(
                target_doc_names=["Alpha", "Beta", "Gamma"],
            ),
        })

        comp = result["comparison_result"]
        # Stage A remediation 2026-10-29: the partial retrieval failure is
        # surfaced as a state error so it cannot masquerade as a clean
        # comparison result; the completed documents are still compared.
        assert "error" in result
        assert "failed for 1 of 3" in result["error"]
        assert "comparison is incomplete" in result["error"]
        assert comp["coverage"]["documents_failed"] == 1
        assert comp["coverage"]["documents_with_evidence"] == 2
        assert any("failed" in lim for lim in comp["limitations"])

    def test_all_docs_fail(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(error_doc_ids={"doc_a", "doc_b"})
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        comp = result["comparison_result"]
        assert comp["coverage"]["documents_failed"] == 2
        assert comp["coverage"]["documents_with_evidence"] == 0
        assert len(result["evidence"]) == 0


# == 8. Structured field lookup ==

class TestStructuredFieldLookup:

    def test_governing_law_triggers_field_query(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        field_rows = [
            {"doc_id": "doc_a", "value_raw": "Delaware", "value_normalized": "Delaware",
             "value_status": "found_explicit", "extraction_method": "llm", "field_id": "f_a"},
            {"doc_id": "doc_b", "value_raw": "New York", "value_normalized": "New York",
             "value_status": "found_explicit", "extraction_method": "llm", "field_id": "f_b"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs, field_rows=field_rows)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare governing law between Alpha and Beta",
            "intent": _make_intent(
                aspect="governing law",
                target_doc_names=["Alpha", "Beta"],
            ),
        })

        comp = result["comparison_result"]
        assert comp["aspect_field"] == "governing_law"
        findings = comp["per_doc_findings"]
        for f in findings:
            assert f.get("structured_values")
        assert findings[0]["structured_values"][0]["value_raw"] == "Delaware"
        assert findings[1]["structured_values"][0]["value_raw"] == "New York"

    def test_non_field_aspect_no_structured_query(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare indemnification between Alpha and Beta",
            "intent": _make_intent(
                aspect="indemnification",
                target_doc_names=["Alpha", "Beta"],
            ),
        })

        comp = result["comparison_result"]
        assert comp["aspect_field"] is None
        for f in comp["per_doc_findings"]:
            assert "structured_values" not in f

    def test_missing_extraction_caveat_present(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        field_rows = [
            {"doc_id": "doc_a", "value_raw": "", "value_normalized": "",
             "value_status": "not_found", "extraction_method": "llm", "field_id": "f_a"},
            {"doc_id": "doc_b", "value_raw": "Delaware", "value_normalized": "Delaware",
             "value_status": "found_explicit", "extraction_method": "llm", "field_id": "f_b"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c_a")])
        executor = FakeSqlExecutor(doc_rows=docs, field_rows=field_rows)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare governing law between Alpha and Beta",
            "intent": _make_intent(
                aspect="governing law",
                target_doc_names=["Alpha", "Beta"],
            ),
        })

        answer = result["comparison_result"]["answer_text"]
        assert "not proof" in answer.lower()


# == 9. Grounded comparison output ==

class TestGroundedOutput:

    def test_answer_generator_called_with_evidence(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a", text="Alpha termination clause"),
            _make_chunk("doc_b", "c_b", text="Beta termination clause"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        gen = FakeAnswerGenerator("Generated comparison.")
        node = make_comparison_node(retriever, executor, answer_generator=gen)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        assert len(gen.calls) == 1
        assert "Compare" in gen.calls[0]["query"]
        assert "only claims supported" in gen.calls[0]["query"]
        assert len(gen.calls[0]["chunks"]) == 2
        assert result["comparison_result"]["answer_text"] == "Generated comparison."

    def test_generator_failure_falls_back_to_structured(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c_a")])
        executor = FakeSqlExecutor(doc_rows=docs)
        def failing_generator(q, c):
            raise RuntimeError("LLM failed")
        node = make_comparison_node(retriever, executor, answer_generator=failing_generator)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        answer = result["comparison_result"]["answer_text"]
        assert "Comparison" in answer

    def test_no_generator_uses_structured_text(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c_a")])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        answer = result["comparison_result"]["answer_text"]
        assert "Findings by agreement" in answer
        assert "Alpha.pdf" in answer


# == 10. Error handling ==

class TestErrorHandling:

    def test_enumeration_error_recorded(self):
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(error=RuntimeError("DB connection lost"))
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        assert "error" in result
        assert "enumeration failed" in result["error"].lower()
        assert len(retriever.calls) == 0

    def test_trace_on_error(self):
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(error=RuntimeError("timeout"))
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        trace = result["trace"]
        assert trace[0]["action"] == "failed"
        assert "timeout" in trace[0]["reason"]

    def test_empty_corpus(self):
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=[])
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        comp = result["comparison_result"]
        assert comp["coverage"]["documents_compared"] == 0
        assert "No canonical documents" in comp["answer_text"]


# == 11. Graph routing ==

class TestGraphRouting:

    def test_real_node_used_with_deps(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a"),
            _make_chunk("doc_b", "c_b"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "Compare termination clauses between Alpha and Beta",
            "conversation_history": [],
        })

        comp = result.get("comparison_result", {})
        assert not comp.get("stub")
        assert comp.get("capability") == "comparison"

    def test_stub_used_without_deps(self):
        graph = build_graph(classifier=RuleBasedIntentClassifier())

        result = graph.invoke({
            "user_query": "Compare termination clauses between Bellring and Cerence",
            "conversation_history": [],
        })

        comp = result.get("comparison_result", {})
        assert comp.get("stub") is True

    def test_comparison_trace_order(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[_make_chunk("doc_a", "c_a")])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "Compare termination clauses between Alpha and Beta",
            "conversation_history": [],
        })

        nodes = [t["node"] for t in result.get("trace", [])]
        assert "query_planner" in nodes
        assert "comparison" in nodes
        assert "evidence_validation" in nodes
        assert "final_answer" in nodes

    def test_answer_uses_comparison_result(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a"),
            _make_chunk("doc_b", "c_b"),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "Compare termination clauses between Alpha and Beta",
            "conversation_history": [],
        })

        answer = result.get("answer", "")
        assert "Comparison" in answer

    def test_source_refs_built_from_evidence(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        retriever = FakeRetriever(results=[
            _make_chunk("doc_a", "c_a", doc_name="Alpha.pdf",
                       section_path="Sec 5", page_start=10),
            _make_chunk("doc_b", "c_b", doc_name="Beta.pdf",
                       section_path="Sec 3", page_start=5),
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "Compare termination clauses between Alpha and Beta",
            "conversation_history": [],
        })

        refs = result.get("source_refs", [])
        assert len(refs) >= 1


# == 12. Path preservation (regression) ==

class TestPathPreservation:

    def test_hybrid_qa_path_preserved(self):
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
        assert "comparison" not in nodes

    def test_clause_discovery_path_preserved(self):
        retriever = FakeRetriever()
        executor = FakeSqlExecutor(doc_rows=_make_doc_rows(3))
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "Find all force majeure clauses across all contracts",
            "conversation_history": [],
        })

        intent = result.get("intent")
        assert intent.label == "clause_discovery"
        nodes = [t["node"] for t in result.get("trace", [])]
        assert "clause_discovery" in nodes

    def test_structured_lookup_path_preserved(self):
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

    def test_corpus_overview_path_preserved(self):
        executor = FakeSqlExecutor(
            doc_rows=_make_doc_rows(1),
            field_rows=[],
        )
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

    def test_guard_path_preserved(self):
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
        assert ("guard_node" in nodes or "clarification" in nodes), (
            f"Expected guard_node or clarification in trace, got nodes: {nodes}")

    def test_comparison_without_retriever_uses_stub(self):
        executor = FakeSqlExecutor(doc_rows=_make_doc_rows(2))
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            sql_executor=executor,
        )

        result = graph.invoke({
            "user_query": "Compare termination clauses between Alpha and Beta",
            "conversation_history": [],
        })

        comp = result.get("comparison_result", {})
        assert comp.get("stub") is True

    def test_comparison_without_executor_uses_stub(self):
        retriever = FakeRetriever()
        graph = build_graph(
            classifier=RuleBasedIntentClassifier(),
            retriever=retriever,
        )

        result = graph.invoke({
            "user_query": "Compare termination clauses between Alpha and Beta",
            "conversation_history": [],
        })

        comp = result.get("comparison_result", {})
        assert comp.get("stub") is True


# == 13. Evidence truncation ==

class TestEvidenceTruncation:

    def test_evidence_capped_at_max(self):
        docs = [
            {"doc_id": "doc_a", "file_name": "Alpha.pdf"},
            {"doc_id": "doc_b", "file_name": "Beta.pdf"},
        ]
        chunks = [_make_chunk("doc_a", f"c_a_{i}") for i in range(150)] + \
                  [_make_chunk("doc_b", f"c_b_{i}") for i in range(150)]
        retriever = FakeRetriever(results=chunks)
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(
            retriever, executor, max_per_doc=150,
        )

        result = node({
            "user_query": COMPARISON_QUERY,
            "intent": _make_intent(target_doc_names=["Alpha", "Beta"]),
        })

        assert len(result["evidence"]) <= MAX_TOTAL_EVIDENCE
        comp = result["comparison_result"]
        assert any("truncated" in lim.lower() for lim in comp["limitations"])

    def test_source_display_caveat_when_gt_3_docs(self):
        docs = _make_doc_rows(5)
        retriever = FakeRetriever(results=[
            _make_chunk(f"doc_{i}", f"c_{i}") for i in range(5)
        ])
        executor = FakeSqlExecutor(doc_rows=docs)
        node = make_comparison_node(retriever, executor)

        result = node({
            "user_query": "Compare termination clauses across 5 contracts",
            "intent": _make_intent(
                target_doc_names=[f"Contract_{i}" for i in range(5)],
            ),
        })

        answer = result["comparison_result"]["answer_text"]
        assert "capped at 3" in answer.lower()
