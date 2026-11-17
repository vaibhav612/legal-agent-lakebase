"""Unit tests for Milestone 3 - Corpus Overview node in LangGraph.

Covers corpus-level document counts, document-class distributions,
field-level extraction coverage (with latest-run deduplication),
not_found/uncertain/no-record distinctions, empty corpora, SQL safety
(allowlisted field names only, no user input in SQL structure), bounded
results, database-error handling, graph routing, and preservation of
existing hybrid QA and structured_lookup paths.

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

from workflow.state import ALLOWED_FIELDS, Intent
from workflow.classifier import RuleBasedIntentClassifier
from workflow.corpus_overview import (
    MAX_CATEGORY_ROWS,
    MISSING_CAVEAT,
    make_corpus_overview_node,
)
from workflow.graph import build_graph


CORPUS_QUERY = "How many documents are in the corpus?"
FIELD_COVERAGE_QUERY = "How many contracts have a termination for convenience clause?"
HYBRID_QUERY = "What are the termination conditions in the Cerence contract?"
LOOKUP_QUERY = "What is the governing law of the Bellring agreement?"


# -- Fakes (no network, no warehouse, no LLM) ---------------------------

class FakeSqlExecutor:
    """Fake SqlExecutorLike: records every (sql, params) call and returns
    deterministic rows.  Dispatches on SQL content so count queries, class
    distribution queries, and field-coverage queries can be scripted
    independently."""

    def __init__(self, count_rows=None, class_rows=None, coverage_rows=None,
                 error=None):
        self.count_rows = list(count_rows or [])
        self.class_rows = list(class_rows or [])
        self.coverage_rows = list(coverage_rows or [])
        self.error = error
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append({"sql": sql, "params": dict(params or {})})
        if self.error is not None:
            raise self.error
        if "total_documents" in sql and "document_class" not in sql:
            return list(self.count_rows)
        if "document_class" in sql and "GROUP BY" in sql:
            return list(self.class_rows)
        if "total_canonical_docs" in sql:
            return list(self.coverage_rows)
        return []


class FakeRetriever:
    """Minimal fake for the hybrid Q&A regression test."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def search(self, query, num_results=5):
        self.calls.append({"query": query, "num_results": num_results})
        return self.results[:num_results]


def _count_row(total=60, canonical=56):
    return {"total_documents": total, "canonical_documents": canonical}


def _class_row(cls="contract", count=40):
    return {"document_class": cls, "doc_count": count}


def _coverage_row(total=56, with_record=56, no_record=0,
                   found=52, not_found=2, inferred=2, conflicting=0, redacted=0):
    return {
        "total_canonical_docs": total,
        "docs_with_record": with_record,
        "docs_no_record": no_record,
        "docs_found_explicit": found,
        "docs_not_found": not_found,
        "docs_inferred": inferred,
        "docs_conflicting": conflicting,
        "docs_redacted": redacted,
    }


def _intent(**kwargs):
    base = dict(label="corpus_overview", scope="all", needs_evidence=False)
    base.update(kwargs)
    return Intent(**base)


def _run_node(executor, intent):
    node = make_corpus_overview_node(executor)
    return node({"user_query": "overview", "intent": intent})


def _make_graph(executor, retriever=None, generator=None):
    return build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        answer_generator=generator,
        sql_executor=executor,
    )


def _run(graph, query):
    return graph.invoke({"user_query": query, "conversation_history": []})


def _trace_nodes(state):
    return [e["node"] for e in state.get("trace", [])]


# -- Test group 1: corpus summary (document counts) ----------------------

def test_1a_corpus_summary_returns_counts():
    """Corpus summary returns total and canonical document counts."""
    executor = FakeSqlExecutor(
        count_rows=[_count_row(total=60, canonical=56)],
        class_rows=[_class_row("contract", 40), _class_row("amendment", 12),
                   _class_row("form_or_filing", 4)],
    )
    result = _run_node(executor, _intent())
    meta = result["metadata_result"]

    assert meta["stub"] is False
    assert meta["capability"] == "corpus_overview"
    assert meta["mode"] == "corpus_summary"
    counts = meta["doc_counts"]
    assert counts["total_documents"] == 60
    assert counts["canonical_documents"] == 56


def test_1b_corpus_summary_includes_class_distribution():
    """Corpus summary includes document-class distribution for canonical docs."""
    executor = FakeSqlExecutor(
        count_rows=[_count_row()],
        class_rows=[_class_row("contract", 40), _class_row("amendment", 12),
                   _class_row("form_or_filing", 4)],
    )
    result = _run_node(executor, _intent())
    dist = result["metadata_result"]["doc_class_distribution"]

    assert len(dist) == 3
    assert dist[0]["document_class"] == "contract"
    assert dist[0]["doc_count"] == 40
    assert dist[1]["document_class"] == "amendment"
    assert dist[1]["doc_count"] == 12


def test_1c_corpus_summary_answer_text_format():
    """Answer text includes counts and distribution in human-readable format."""
    executor = FakeSqlExecutor(
        count_rows=[_count_row(total=60, canonical=56)],
        class_rows=[_class_row("contract", 40)],
    )
    result = _run_node(executor, _intent())
    answer = result["metadata_result"]["answer_text"]

    assert "Total registered documents: 60" in answer
    assert "Canonical documents: 56" in answer
    assert "contract: 40" in answer


def test_1d_corpus_summary_empty_corpus():
    """Empty corpus returns zero counts and no distribution."""
    executor = FakeSqlExecutor(
        count_rows=[_count_row(total=0, canonical=0)],
        class_rows=[],
    )
    result = _run_node(executor, _intent())
    meta = result["metadata_result"]

    assert meta["doc_counts"]["total_documents"] == 0
    assert meta["doc_counts"]["canonical_documents"] == 0
    assert meta["doc_class_distribution"] == []
    assert "no canonical documents found" in meta["answer_text"]


def test_1e_corpus_summary_no_count_rows():
    """If count query returns no rows, defaults to zero."""
    executor = FakeSqlExecutor(count_rows=[], class_rows=[])
    result = _run_node(executor, _intent())
    assert result["metadata_result"]["doc_counts"]["total_documents"] == 0


# -- Test group 2: field coverage -----------------------------------------

def test_2a_field_coverage_returns_counts():
    """Field coverage returns total/with-record/no-record counts."""
    executor = FakeSqlExecutor(
        coverage_rows=[_coverage_row(total=56, with_record=56, no_record=0)],
    )
    result = _run_node(executor, _intent(target_field="governing_law"))
    meta = result["metadata_result"]

    assert meta["mode"] == "field_coverage"
    assert meta["target_field"] == "governing_law"
    cov = meta["coverage"]
    assert cov["total_canonical_docs"] == 56
    assert cov["docs_with_record"] == 56
    assert cov["docs_no_record"] == 0


def test_2b_field_coverage_status_distribution():
    """Field coverage reports status distribution (found/not_found/inferred/etc)."""
    executor = FakeSqlExecutor(
        coverage_rows=[_coverage_row(found=50, not_found=3, inferred=2,
                                     conflicting=1, redacted=0)],
    )
    result = _run_node(executor, _intent(target_field="parties"))
    cov = result["metadata_result"]["coverage"]

    assert cov["docs_found_explicit"] == 50
    assert cov["docs_not_found"] == 3
    assert cov["docs_inferred"] == 2
    assert cov["docs_conflicting"] == 1
    assert cov["docs_redacted"] == 0


def test_2c_field_coverage_no_record_distinction():
    """Documents with no extraction record are counted separately."""
    executor = FakeSqlExecutor(
        coverage_rows=[_coverage_row(total=56, with_record=50, no_record=6,
                                     found=48, not_found=2)],
    )
    result = _run_node(executor, _intent(target_field="expiration_or_term"))
    cov = result["metadata_result"]["coverage"]

    assert cov["docs_no_record"] == 6
    assert cov["docs_with_record"] == 50
    answer = result["metadata_result"]["answer_text"]
    assert "no extraction record: 6" in answer


def test_2d_field_coverage_missing_caveat_present():
    """No-record or not_found triggers the no-proof-of-absence caveat."""
    executor = FakeSqlExecutor(
        coverage_rows=[_coverage_row(no_record=5, not_found=3)],
    )
    result = _run_node(executor, _intent(target_field="governing_law"))
    answer = result["metadata_result"]["answer_text"]
    assert MISSING_CAVEAT in answer


def test_2e_field_coverage_no_caveat_when_all_found():
    """No caveat when all documents have found_explicit values."""
    executor = FakeSqlExecutor(
        coverage_rows=[_coverage_row(total=56, with_record=56, no_record=0,
                                     found=56, not_found=0, inferred=0)],
    )
    result = _run_node(executor, _intent(target_field="governing_law"))
    answer = result["metadata_result"]["answer_text"]
    assert MISSING_CAVEAT not in answer


def test_2f_field_coverage_all_ten_fields():
    """Field coverage works for each of the ten allowlisted fields."""
    for field in ALLOWED_FIELDS:
        executor = FakeSqlExecutor(coverage_rows=[_coverage_row()])
        result = _run_node(executor, _intent(target_field=field))
        assert result["metadata_result"]["target_field"] == field
        sql = executor.calls[0]["sql"]
        assert "field_name = '" + field + "'" in sql


def test_2g_field_coverage_empty_result():
    """Empty coverage result returns zero total."""
    executor = FakeSqlExecutor(coverage_rows=[])
    result = _run_node(executor, _intent(target_field="governing_law"))
    cov = result["metadata_result"]["coverage"]
    assert cov["total_canonical_docs"] == 0


# -- Test group 3: SQL safety (allowlisting, no user input in SQL) -------

def test_3a_only_allowlisted_field_interpolated():
    """The field name in the SQL is always an allowlisted identifier."""
    executor = FakeSqlExecutor(coverage_rows=[_coverage_row()])
    _run_node(executor, _intent(target_field="governing_law"))
    sql = executor.calls[0]["sql"]
    assert "field_name = 'governing_law'" in sql
    assert "workspace.legal_assistant_v2.extracted_fields" in sql
    assert "workspace.legal_assistant_v2.documents" in sql


def test_3b_non_allowlisted_field_falls_back_to_corpus_summary():
    """A non-allowlisted target_field is ignored; corpus summary is returned."""
    executor = FakeSqlExecutor(count_rows=[_count_row()], class_rows=[])
    # Bypass Intent validation by using a dict-style intent
    node = make_corpus_overview_node(executor)
    out = node({"intent": {"target_field": "evil_field", "label": "corpus_overview"}})
    assert out["metadata_result"]["mode"] == "corpus_summary"
    # No field_coverage SQL should have been issued
    assert all("field_name" not in c["sql"] for c in executor.calls)


def test_3c_no_params_in_field_coverage_query():
    """Field coverage SQL has no user-derived parameters (no user input)."""
    executor = FakeSqlExecutor(coverage_rows=[_coverage_row()])
    _run_node(executor, _intent(target_field="parties"))
    params = executor.calls[0]["params"]
    assert params == {}


def test_3d_corpus_summary_no_params():
    """Corpus summary queries carry no parameters."""
    executor = FakeSqlExecutor(count_rows=[_count_row()], class_rows=[_class_row()])
    _run_node(executor, _intent())
    for call in executor.calls:
        assert call["params"] == {}


def test_3e_sql_injection_in_field_name_blocked_by_allowlist():
    """An injection payload in target_field never reaches the SQL builder."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _intent(target_field="x; DROP TABLE documents; --")


# -- Test group 4: bounded results -----------------------------------------

def test_4a_class_distribution_bounded_by_max():
    """Document-class distribution is bounded by MAX_CATEGORY_ROWS."""
    executor = FakeSqlExecutor(count_rows=[_count_row()], class_rows=[])
    _run_node(executor, _intent())
    class_sql = [c for c in executor.calls if "document_class" in c["sql"]][0]["sql"]
    assert "LIMIT " + str(MAX_CATEGORY_ROWS) in class_sql


def test_4b_class_distribution_limit_is_integer_literal():
    """The LIMIT is an integer literal, not a parameter."""
    executor = FakeSqlExecutor(count_rows=[_count_row()], class_rows=[])
    _run_node(executor, _intent())
    class_call = [c for c in executor.calls if "document_class" in c["sql"]][0]
    assert "limit" not in str(class_call["params"]).lower()


# -- Test group 5: error handling -----------------------------------------

def test_5a_database_error_recorded_not_propagated():
    """A database error is recorded in state, not raised."""
    executor = FakeSqlExecutor(error=RuntimeError("connection refused"))
    result = _run_node(executor, _intent())
    assert "error" in result
    assert "connection refused" in result["error"]
    assert "metadata_result" not in result
    assert result["trace"][0]["action"] == "failed"


def test_5b_database_error_in_field_coverage_recorded():
    """A database error during field coverage is recorded with field name."""
    executor = FakeSqlExecutor(error=RuntimeError("timeout"))
    result = _run_node(executor, _intent(target_field="governing_law"))
    assert "error" in result
    assert "timeout" in result["error"]
    trace = result["trace"][0]
    assert trace["details"]["field"] == "governing_law"
    assert trace["details"]["mode"] == "field_coverage"


def test_5c_error_not_disguised_as_empty_corpus():
    """An error does not produce a zero-count metadata_result."""
    executor = FakeSqlExecutor(error=RuntimeError("auth failed"))
    result = _run_node(executor, _intent())
    assert "error" in result
    assert "metadata_result" not in result


# -- Test group 6: graph routing and response formatting ------------------

def test_6a_corpus_overview_routes_to_real_node_with_executor():
    """With a SQL executor, corpus_overview intent routes to the real node."""
    executor = FakeSqlExecutor(
        count_rows=[_count_row(total=60, canonical=56)],
        class_rows=[_class_row("contract", 40)],
    )
    graph = _make_graph(executor)
    state = _run(graph, CORPUS_QUERY)

    nodes = _trace_nodes(state)
    assert "corpus_overview" in nodes
    assert "corpus_overview_stub" not in nodes
    assert nodes.index("query_planner") < nodes.index("corpus_overview")
    assert "final_answer" in nodes


def test_6b_corpus_summary_formatted_by_final_answer():
    """The final_answer node formats the real metadata_result answer_text."""
    executor = FakeSqlExecutor(
        count_rows=[_count_row(total=60, canonical=56)],
        class_rows=[_class_row("contract", 40), _class_row("amendment", 16)],
    )
    graph = _make_graph(executor)
    state = _run(graph, CORPUS_QUERY)

    answer = state["answer"]
    assert "Total registered documents: 60" in answer
    assert "Canonical documents: 56" in answer
    assert "contract: 40" in answer
    assert "amendment: 16" in answer


def test_6c_field_coverage_routes_through_graph():
    """Field-coverage queries route through the graph and produce an answer."""
    executor = FakeSqlExecutor(
        coverage_rows=[_coverage_row(total=56, with_record=56, found=54,
                                     not_found=2)],
    )
    graph = _make_graph(executor)
    state = _run(graph, FIELD_COVERAGE_QUERY)

    answer = state["answer"]
    assert "termination_for_convenience" in answer
    assert "Canonical documents: 56" in answer
    assert MISSING_CAVEAT in answer


def test_6d_corpus_overview_stub_without_executor():
    """Without a SQL executor, the corpus_overview stub is used."""
    graph = build_graph(classifier=RuleBasedIntentClassifier())
    state = _run(graph, CORPUS_QUERY)

    nodes = _trace_nodes(state)
    assert "corpus_overview_stub" in nodes
    meta = state.get("metadata_result", {})
    assert meta.get("stub") is True


# -- Test group 7: preservation of existing paths -------------------------

def test_7a_hybrid_qa_path_preserved():
    """Hybrid Q&A routing and execution still work with corpus_overview active."""
    fake_retriever = FakeRetriever(results=[
        {"chunk_id": "c1", "doc_id": "d1", "doc_name": "Test.pdf",
         "section_path": "Section 1", "page_start": 1, "page_end": 2,
         "chunk_text": "Termination conditions apply.",
         "rrf_score": 0.03}
    ])
    executor = FakeSqlExecutor(count_rows=[_count_row()], class_rows=[])
    graph = _make_graph(executor, retriever=fake_retriever)
    state = _run(graph, HYBRID_QUERY)

    nodes = _trace_nodes(state)
    assert "hybrid_qa" in nodes
    assert "corpus_overview" not in nodes
    assert "evidence_validation" in nodes
    assert "final_answer" in nodes


def test_7b_structured_lookup_path_preserved():
    """Structured lookup routing and execution still work with corpus_overview active."""
    class CombinedExecutor(FakeSqlExecutor):
        def __init__(self, **kwargs):
            self.field_rows = kwargs.pop("field_rows", [
                {"doc_id": "d1", "file_name": "Bellring.pdf",
                 "is_canonical": True, "document_class": "contract",
                 "field_id": "f1", "value_index": 1,
                 "value_raw": "Delaware", "value_normalized": "Delaware",
                 "value_status": "found_explicit",
                 "extraction_method": "llm", "run_id": "r1"}
            ])
            self.evidence_rows = kwargs.pop("evidence_rows", [])
            super().__init__(**kwargs)

        def execute(self, sql, params=None):
            self.calls.append({"sql": sql, "params": dict(params or {})})
            if self.error is not None:
                raise self.error
            if "legal_assistant_v2.evidence" in sql:
                return list(self.evidence_rows)
            if "total_documents" in sql and "document_class" not in sql:
                return list(self.count_rows)
            if "document_class" in sql and "GROUP BY" in sql:
                return list(self.class_rows)
            if "total_canonical_docs" in sql:
                return list(self.coverage_rows)
            if "field_name = '" in sql and "value_status" in sql:
                return list(self.field_rows)
            return []

    executor = CombinedExecutor(count_rows=[_count_row()], class_rows=[])
    graph = _make_graph(executor)
    state = _run(graph, LOOKUP_QUERY)

    nodes = _trace_nodes(state)
    assert "structured_lookup" in nodes
    assert "corpus_overview" not in nodes
    assert "final_answer" in nodes


def test_7c_routing_tests_still_pass_without_executor():
    """All existing routing tests run without a SQL executor (DI preserved)."""
    graph = build_graph(classifier=RuleBasedIntentClassifier())
    state = _run(graph, CORPUS_QUERY)
    assert state.get("answer") is not None
    assert not state.get("error")


def test_7d_guard_path_preserved():
    """Unsupported dump requests route away from corpus_overview."""
    executor = FakeSqlExecutor(count_rows=[_count_row()], class_rows=[])
    graph = _make_graph(executor)
    state = _run(graph, "show me all text from all documents")

    nodes = _trace_nodes(state)
    assert "corpus_overview" not in nodes
    assert "corpus_overview_stub" not in nodes