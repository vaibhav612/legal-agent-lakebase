"""Unit tests for Milestone 3 - Structured Lookup node in LangGraph.

Covers the ten allowlisted fields, the three query modes (lookup,
value_filter, status_filter), missing/uncertain/no-record semantics,
no-proof-of-absence caveats, SQL-injection safety, parameter separation,
bounded results, database-error handling, evidence linkage degradation,
graph routing, and preservation of the existing hybrid Q&A path.

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
from pydantic import ValidationError

from workflow.state import ALLOWED_FIELDS, Intent
from workflow.classifier import RuleBasedIntentClassifier
from workflow.structured_lookup import (
    MAX_EVIDENCE_PER_VALUE,
    MAX_LOOKUP_ROWS,
    MISSING_CAVEAT,
    StatementExecutionSqlExecutor,
    make_structured_lookup_node,
)
from workflow.graph import build_graph


LOOKUP_QUERY = "What is the governing law of the Bellring agreement?"
# D2 completion (2026-10-29): a document-NAMED hybrid query now uses the
# executor for canonical-doc name resolution (covered in
# test_hybrid_qa_langgraph.py).  This constant keeps the name-less shape so
# this suite's "hybrid queries never touch the SQL executor" contract
# remains about the corpus-wide hybrid path.
HYBRID_QUERY = "What are the termination conditions in this contract?"


# -- Fakes (no network, no warehouse, no LLM) ---------------------------

class FakeSqlExecutor:
    """Fake SqlExecutorLike: records every (sql, params) call and returns
    deterministic rows.  Dispatches on table name so field queries and
    evidence queries can be scripted independently."""

    def __init__(self, field_rows=None, evidence_rows=None, error=None,
                 evidence_error=None):
        self.field_rows = list(field_rows or [])
        self.evidence_rows = list(evidence_rows or [])
        self.error = error
        self.evidence_error = evidence_error
        self.calls = []

    @property
    def field_calls(self):
        return [c for c in self.calls
                if "legal_assistant_v2.evidence" not in c["sql"]]

    @property
    def evidence_calls(self):
        return [c for c in self.calls
                if "legal_assistant_v2.evidence" in c["sql"]]

    def execute(self, sql, params=None):
        self.calls.append({"sql": sql, "params": dict(params or {})})
        if "legal_assistant_v2.evidence" in sql:
            if self.evidence_error is not None:
                raise self.evidence_error
            return list(self.evidence_rows)
        if self.error is not None:
            raise self.error
        return list(self.field_rows)


class FakeRetriever:
    """Minimal fake for the hybrid Q&A regression test."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def search(self, query, num_results=5):
        self.calls.append({"query": query, "num_results": num_results})
        return self.results[:num_results]


def _frow(doc_id="d1", file_name="BellringBrands_MSA.pdf", field_id="f1",
          value_index=1, value_raw="California", value_normalized="California",
          value_status="found_explicit", is_canonical=True,
          document_class="contract", extraction_method="llm", run_id="run-1"):
    """One row shaped exactly like the field query output."""
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "is_canonical": is_canonical,
        "document_class": document_class,
        "field_id": field_id,
        "value_index": value_index,
        "value_raw": value_raw,
        "value_normalized": value_normalized,
        "value_status": value_status,
        "extraction_method": extraction_method,
        "run_id": run_id,
    }


def _erow(field_id="f1", evidence_id="e1", chunk_id="c1",
          quote="This Agreement is governed by the laws of California",
          page_start=3, page_end=4, evidence_type="direct_quote"):
    """One row shaped exactly like the evidence query output."""
    return {
        "field_id": field_id,
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "quote": quote,
        "page_start": page_start,
        "page_end": page_end,
        "evidence_type": evidence_type,
    }


def _intent(**kwargs):
    base = dict(label="structured_lookup", scope="specific_docs",
                target_doc_names=["Bellring"], target_field="governing_law",
                needs_evidence=False)
    base.update(kwargs)
    return Intent(**base)


def _run_node(executor, intent, max_rows=None):
    node = make_structured_lookup_node(executor, max_rows=max_rows)
    return node({"user_query": "lookup", "intent": intent})


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


# -- Test group 1: the ten allowlisted fields are mapped correctly --------

def test_1a_classifier_field_patterns_cover_all_ten_fields():
    """Every allowlisted field has at least one classifier phrase mapping."""
    mapped = {field for _, field in RuleBasedIntentClassifier.FIELD_PATTERNS}
    assert mapped == set(ALLOWED_FIELDS)
    assert len(ALLOWED_FIELDS) == 10


def test_1b_node_accepts_each_of_the_ten_fields():
    """The node builds a valid query for each allowlisted field."""
    for field in ALLOWED_FIELDS:
        executor = FakeSqlExecutor(field_rows=[_frow()])
        result = _run_node(executor, _intent(target_field=field))
        assert "error" not in result
        sql = executor.field_calls[0]["sql"]
        assert f"field_name = '{field}'" in sql
        assert result["lookup_result"]["target_field"] == field


def test_1c_non_allowlisted_field_rejected_by_intent_model():
    """A hostile field name never reaches the SQL builder."""
    with pytest.raises(ValidationError):
        _intent(target_field="value_raw; DROP TABLE documents; --")


def test_1d_node_guards_against_unvalidated_field():
    """Even with a hand-built state dict, the node refuses bad fields."""
    node = make_structured_lookup_node(FakeSqlExecutor())
    out = node({"intent": {"target_field": "doc_id", "label": "structured_lookup"}})
    assert "error" in out
    assert "not allowlisted" in out["error"]


# -- Test group 2: valid lookup returns the expected structured result ----

def test_2_lookup_returns_values_statuses_and_evidence():
    """A found_explicit lookup returns value, status, method and evidence."""
    executor = FakeSqlExecutor(
        field_rows=[_frow()],
        evidence_rows=[_erow()],
    )
    result = _run_node(executor, _intent())
    lookup = result["lookup_result"]

    assert lookup["stub"] is False
    assert lookup["capability"] == "structured_lookup"
    assert lookup["mode"] == "lookup"
    assert lookup["target_field"] == "governing_law"

    doc = lookup["results"][0]
    assert doc["doc_id"] == "d1"
    assert doc["doc_name"] == "BellringBrands_MSA.pdf"
    assert doc["record_status"] == "has_record"
    value = doc["values"][0]
    assert value["value_normalized"] == "California"
    assert value["value_status"] == "found_explicit"
    assert value["extraction_method"] == "llm"
    assert value["run_id"] == "run-1"
    assert value["evidence"][0]["quote"].startswith("This Agreement is governed")
    assert value["evidence"][0]["page_start"] == 3

    assert "California" in lookup["answer_text"]
    # Stage 2 fix (2026-10-29): the answer text now uses the readable
    # display title; the canonical stored name stays in the machine fields
    # (doc_name above and lookup sources below).
    assert "BellringBrands MSA" in lookup["answer_text"]
    # provenance preserved: sources carry doc + page + quote
    assert lookup["sources"][0]["doc_name"] == "BellringBrands_MSA.pdf"
    assert lookup["sources"][0]["page_start"] == 3


def test_2b_multi_value_field_all_values_returned():
    """Multi-valued fields (multiple value_index rows) return all values."""
    executor = FakeSqlExecutor(field_rows=[
        _frow(value_index=1, value_raw="Florida", value_normalized="Florida"),
        _frow(value_index=2, value_raw="Florida", value_normalized="Florida",
              field_id="f2"),
    ])
    result = _run_node(executor, _intent())
    values = result["lookup_result"]["results"][0]["values"]
    assert [v["value_index"] for v in values] == [1, 2]


# -- Test group 3: supported filter queries return matching documents ----

def test_3a_value_filter_returns_matching_documents():
    """'Which agreements are governed by Delaware?' filters by value."""
    executor = FakeSqlExecutor(field_rows=[
        _frow(doc_id="d9", file_name="Armstrong_IP.pdf", value_raw="Delaware",
              value_normalized="Delaware"),
    ])
    result = _run_node(executor, _intent(
        scope="filtered", target_doc_names=[], target_value="Delaware"))
    lookup = result["lookup_result"]
    assert lookup["mode"] == "value_filter"
    assert lookup["filter_value"] == "Delaware"
    assert lookup["results"][0]["doc_name"] == "Armstrong_IP.pdf"

    sql = executor.field_calls[0]["sql"]
    params = executor.field_calls[0]["params"]
    # value bound as a parameter, never in the SQL text
    assert params["value_pattern"] == "%delaware%"
    assert "delaware" not in sql.lower()
    assert "l.field_id IS NOT NULL" in sql


def test_3b_value_filter_routed_end_to_end():
    """The rule-based classifier routes a governed-by query to the node."""
    classifier = RuleBasedIntentClassifier()
    intent = classifier.classify("Which agreements are governed by Delaware?")
    assert intent.label == "structured_lookup"
    assert intent.target_field == "governing_law"
    assert intent.target_value == "Delaware"

    executor = FakeSqlExecutor(field_rows=[_frow(value_raw="Delaware",
                                                 value_normalized="Delaware")])
    state = _run(_make_graph(executor), "Which agreements are governed by Delaware?")
    assert state["lookup_result"]["mode"] == "value_filter"
    assert "structured_lookup" in _trace_nodes(state)


def test_3c_status_filter_missing_and_uncertain():
    """'missing or uncertain values' maps to the right status set."""
    classifier = RuleBasedIntentClassifier()
    intent = classifier.classify(
        "Which agreements have missing or uncertain values for the governing law?"
    )
    assert intent.label == "structured_lookup"
    assert intent.status_filter == "missing_or_uncertain"

    executor = FakeSqlExecutor(field_rows=[
        _frow(value_status="not_found", value_raw=None, value_normalized=None),
    ])
    result = _run_node(executor, _intent(
        scope="filtered", target_doc_names=[], status_filter="missing_or_uncertain"))
    lookup = result["lookup_result"]
    assert lookup["mode"] == "status_filter"

    sql = executor.field_calls[0]["sql"]
    params = executor.field_calls[0]["params"]
    # LEFT JOIN null rows count as missing, but never as uncertain-only
    assert "l.field_id IS NULL OR" in sql
    assert params["status_0"] == "not_found"
    assert params["status_1"] == "inferred"
    assert params["status_2"] == "conflicting"
    # status names are parameters, not SQL text
    assert "not_found" not in sql


def test_3d_invalid_status_filter_rejected():
    """Only the three defined status_filter values are accepted."""
    with pytest.raises(ValidationError):
        _intent(status_filter="anything') OR 1=1 --")


# -- Test group 4: missing / uncertain values represented accurately ----

def test_4a_not_found_is_explicit_and_distinct_from_no_record():
    """not_found (extractor ran, no value) != no extraction record at all."""
    executor = FakeSqlExecutor(field_rows=[
        # LEFT JOIN null row: document exists but has no extraction record
        _frow(doc_id="d1", file_name="No_Record.pdf", field_id=None,
              value_index=None, value_raw=None, value_normalized=None,
              value_status=None, extraction_method=None, run_id=None),
        # Extractor ran but found nothing
        _frow(doc_id="d2", file_name="Not_Found.pdf", field_id="f2",
              value_raw=None, value_normalized=None, value_status="not_found"),
    ])
    result = _run_node(executor, _intent())
    docs = {d["doc_id"]: d for d in result["lookup_result"]["results"]}

    assert docs["d1"]["record_status"] == "no_record"
    assert docs["d1"]["values"] == []
    assert docs["d2"]["record_status"] == "has_record"
    assert docs["d2"]["values"][0]["value_status"] == "not_found"

    answer = result["lookup_result"]["answer_text"]
    assert "no extraction record" in answer
    assert "not_found" in answer
    # the two cases are worded differently - not conflated.  (Stage 2 fix:
    # answer text uses display names; canonical names remain in the result
    # dicts checked above.)
    from workflow.display import display_doc_name
    assert (display_doc_name(docs["d1"]["doc_name"]) in answer
            and display_doc_name(docs["d2"]["doc_name"]) in answer)

    counts = result["lookup_result"]["status_counts"]
    assert counts["no_record"] == 1
    assert counts["not_found"] == 1


def test_4b_missing_value_is_not_proof_of_clause_absence():
    """A missing extraction never claims the clause is absent."""
    executor = FakeSqlExecutor(field_rows=[
        _frow(doc_id="d2", file_name="Not_Found.pdf", field_id="f2",
              value_raw=None, value_normalized=None, value_status="not_found"),
    ])
    result = _run_node(executor, _intent())
    answer = result["lookup_result"]["answer_text"]
    assert "not proof" in answer
    assert MISSING_CAVEAT in answer
    # and no claim of absence anywhere
    assert "clause is absent" not in answer
    assert "does not contain" not in answer


def test_4c_uncertain_values_reported_verbatim():
    """inferred and conflicting statuses surface unchanged."""
    executor = FakeSqlExecutor(field_rows=[
        _frow(doc_id="d1", field_id="f1", value_status="inferred",
              value_raw="New York"),
        _frow(doc_id="d2", field_id="f2", value_status="conflicting",
              value_raw="Texas"),
    ])
    result = _run_node(executor, _intent())
    answer = result["lookup_result"]["answer_text"]
    assert "inferred" in answer
    assert "conflicting" in answer
    counts = result["lookup_result"]["status_counts"]
    assert counts["inferred"] == 1 and counts["conflicting"] == 1


def test_4d_no_matching_records_is_explicit():
    """An empty result is a clear, understandable answer - not an error."""
    executor = FakeSqlExecutor(field_rows=[])
    result = _run_node(executor, _intent(target_doc_names=["Nonexistent"]))
    lookup = result["lookup_result"]
    assert "error" not in result
    assert lookup["results"] == []
    assert lookup["status_counts"]["docs_examined"] == 0
    assert "No matching documents" in lookup["answer_text"]
    # no evidence query is issued when there is nothing to link
    assert executor.evidence_calls == []


# -- Test group 5: SQL safety - injection and parameter separation -------

INJECTION_PAYLOADS = [
    "'; DROP TABLE documents; --",
    "x' OR '1'='1",
    "'; DELETE FROM extracted_fields WHERE '1'='1",
]


def test_5a_malicious_doc_name_cannot_alter_sql():
    """Doc-name payloads are bound as parameters, never joined into SQL."""
    for payload in INJECTION_PAYLOADS:
        executor = FakeSqlExecutor(field_rows=[])
        result = _run_node(executor, _intent(target_doc_names=[payload]))
        assert "error" not in result
        sql = executor.field_calls[0]["sql"]
        params = executor.field_calls[0]["params"]
        assert payload not in sql, f"payload leaked into SQL: {payload}"
        assert payload.lower() not in sql.lower()
        assert params["doc_0"] == "%" + payload.lower() + "%"
        assert "LOWER(d.file_name) LIKE :doc_0" in sql


def test_5b_malicious_value_filter_cannot_alter_sql():
    """Value-filter payloads are bound as parameters."""
    for payload in INJECTION_PAYLOADS:
        executor = FakeSqlExecutor(field_rows=[])
        result = _run_node(executor, _intent(
            scope="filtered", target_doc_names=[], target_value=payload))
        assert "error" not in result
        sql = executor.field_calls[0]["sql"]
        params = executor.field_calls[0]["params"]
        assert payload not in sql
        assert payload.lower() not in sql.lower()
        assert params["value_pattern"].startswith("%")
        assert payload.lower() in params["value_pattern"]


def test_5b_d10_superset_value_fragment_matches_bidirectionally():
    """D10 regression (2026-10-29): a filter fragment that is a SUPERSET of
    the stored value ('Delaware law' vs value 'Delaware') must still match.
    Matching is bidirectional: forward LIKE is kept, and the reverse
    direction checks the fragment against the word-boundary-anchored value
    (regex metacharacters stripped from the value, user fragment bound as a
    parameter — never interpolated into the SQL text)."""
    executor = FakeSqlExecutor(field_rows=[])
    result = _run_node(executor, _intent(
        scope="filtered", target_doc_names=[], target_value="Delaware law"))
    assert "error" not in result
    sql = executor.field_calls[0]["sql"]
    params = executor.field_calls[0]["params"]
    # forward direction kept
    assert "LOWER(COALESCE(l.value_raw, '')) LIKE :value_pattern" in sql
    assert params["value_pattern"] == "%delaware law%"
    # reverse direction added: word-boundary-anchored value inside fragment
    assert ":value_token RLIKE" in sql
    assert params["value_token"] == "delaware law"
    # regex metacharacters are stripped from the stored value before RLIKE
    assert "'[^a-z0-9 ]', ' '" in sql
    # the user fragment never appears in the SQL text
    assert "delaware law" not in sql
    # paren balance — the first live run shipped an unbalanced D10 condition
    # (SQLSTATE 42601 parse error on the warehouse); the condition opens two
    # groups and must close all of them.
    assert sql.count("(") == sql.count(")"), "unbalanced parens in value-filter SQL"


def test_5c_parameters_passed_separately_from_sql_text():
    """The executor receives a parameter dict disjoint from the SQL text."""
    executor = FakeSqlExecutor(field_rows=[_frow()])
    _run_node(executor, _intent(target_doc_names=["Bellring"]))
    call = executor.field_calls[0]
    sql, params = call["sql"], call["params"]
    # SQL uses named placeholders and contains no literal user value
    assert " :doc_0" in sql or ":doc_0 " in sql or "LIKE :doc_0" in sql
    assert "bellring" not in sql.lower()
    assert set(params.keys()) == {"doc_0"}


def test_5d_sql_text_is_fixed_except_allowlisted_field():
    """Table names are constants; only the allowlisted field is interpolated."""
    executor = FakeSqlExecutor(field_rows=[])
    _run_node(executor, _intent())
    sql = executor.field_calls[0]["sql"]
    assert sql.startswith("WITH latest AS")
    assert "workspace.legal_assistant_v2.extracted_fields" in sql
    assert "workspace.legal_assistant_v2.documents" in sql
    assert "ORDER BY created_at DESC, field_id DESC" in sql  # latest-run dedup
    assert sql.rstrip().endswith("LIMIT 200")  # bounded, internal constant


def test_5e_row_limit_bounded_even_if_executor_ignores_limit():
    """Defensive truncation caps rows regardless of executor behaviour."""
    rows = [_frow(doc_id=f"d{i}", field_id=f"f{i}") for i in range(6)]
    executor = FakeSqlExecutor(field_rows=rows)
    result = _run_node(executor, _intent(), max_rows=3)
    lookup = result["lookup_result"]
    assert len(lookup["results"]) == 3
    assert lookup["truncated"] is True


def test_5f_row_limit_clamped_to_max():
    """An oversized max_rows request is clamped to MAX_LOOKUP_ROWS."""
    executor = FakeSqlExecutor(field_rows=[])
    _run_node(executor, _intent(), max_rows=100000)
    sql = executor.field_calls[0]["sql"]
    assert sql.rstrip().endswith(f"LIMIT {MAX_LOOKUP_ROWS}")


# -- Test group 6: database errors handled safely -------------------------

def test_6a_field_query_error_recorded_not_raised():
    """An executor exception becomes a state error + failure trace."""
    executor = FakeSqlExecutor(error=RuntimeError("warehouse unavailable"))
    result = _run_node(executor, _intent())
    assert "error" in result
    assert "warehouse unavailable" in result["error"]
    assert "lookup_result" not in result
    trace = result["trace"][0]
    assert trace["node"] == "structured_lookup"
    assert trace["action"] == "failed"
    assert "warehouse unavailable" in trace["reason"]


def test_6b_error_visible_in_final_answer():
    """Through the full graph, a DB error reaches the answer text."""
    executor = FakeSqlExecutor(error=RuntimeError("connection refused"))
    state = _run(_make_graph(executor), LOOKUP_QUERY)
    assert state["answer"].startswith("Error:")
    assert "connection refused" in state["answer"]


def test_6c_evidence_query_error_degrades_gracefully():
    """Evidence linkage failure keeps the field values usable."""
    executor = FakeSqlExecutor(
        field_rows=[_frow()],
        evidence_error=RuntimeError("evidence table unavailable"),
    )
    result = _run_node(executor, _intent())
    lookup = result["lookup_result"]
    assert "error" not in result
    assert lookup["results"][0]["values"][0]["value_normalized"] == "California"
    assert lookup["results"][0]["values"][0]["evidence"] == []
    assert "evidence table unavailable" in lookup["evidence_error"]
    assert lookup["evidence_error"] in result["trace"][0]["details"]["evidence_error"]


def test_6d_evidence_per_value_capped():
    """At most MAX_EVIDENCE_PER_VALUE evidence rows per value."""
    executor = FakeSqlExecutor(
        field_rows=[_frow()],
        evidence_rows=[_erow(evidence_id=f"e{i}") for i in range(6)],
    )
    result = _run_node(executor, _intent())
    value = result["lookup_result"]["results"][0]["values"][0]
    assert len(value["evidence"]) == MAX_EVIDENCE_PER_VALUE


def test_6e_sources_capped_at_three_documents():
    """Source refs never exceed MAX_SOURCE_DOCS documents."""
    executor = FakeSqlExecutor(
        field_rows=[_frow(doc_id=f"d{i}", field_id=f"f{i}") for i in range(5)],
        evidence_rows=[_erow(field_id=f"f{i}") for i in range(5)],
    )
    result = _run_node(executor, _intent())
    sources = result["lookup_result"]["sources"]
    assert len(sources) == 3


# -- Test group 7: graph routing and regression --------------------------

def test_7a_graph_routes_structured_lookup_to_real_node():
    """structured_lookup intent executes the SQL node, not the stub."""
    executor = FakeSqlExecutor(
        field_rows=[_frow()], evidence_rows=[_erow()])
    state = _run(_make_graph(executor), LOOKUP_QUERY)

    assert state["intent"].label == "structured_lookup"
    nodes = _trace_nodes(state)
    assert "structured_lookup" in nodes
    assert "structured_lookup_stub" not in nodes
    assert "final_answer" in nodes

    lookup = state["lookup_result"]
    assert lookup["stub"] is False
    assert lookup["mode"] == "lookup"
    # value flows into the final answer verbatim (no LLM generation)
    assert "California" in state["answer"]
    assert state["source_refs"][0].doc_name == "BellringBrands_MSA.pdf"


def test_7b_stub_used_without_executor():
    """Without an injected executor the foundation stub is preserved."""
    graph = build_graph(classifier=RuleBasedIntentClassifier())
    state = _run(graph, LOOKUP_QUERY)
    assert state["intent"].label == "structured_lookup"
    assert state["lookup_result"]["stub"] is True
    assert state["answer"].startswith("[Structured Lookup]")


def test_7c_hybrid_qa_path_unchanged():
    """Hybrid queries never touch the SQL executor or lookup node."""
    executor = FakeSqlExecutor()
    retriever = FakeRetriever(results=[{
        "chunk_id": "c1", "chunk_text": "termination upon material breach",
        "doc_id": "d1", "doc_name": "Cerence_MSA.pdf", "section_path": "9.",
        "page_start": 3, "page_end": 4, "rrf_score": 0.03,
    }])
    state = _run(_make_graph(executor, retriever=retriever), HYBRID_QUERY)

    assert state["intent"].label == "hybrid_qa"
    assert len(retriever.calls) == 1
    assert executor.calls == []  # SQL executor never invoked
    nodes = _trace_nodes(state)
    assert "structured_lookup" not in nodes
    assert "evidence_validation" in nodes
    assert state["evidence"][0].doc_name == "Cerence_MSA.pdf"


def test_7d_unsupported_never_falls_back_to_lookup_or_hybrid():
    """Corpus dumps stay rejected - no silent capability fallback."""
    executor = FakeSqlExecutor()
    state = _run(_make_graph(executor), "Show me all text from all documents")
    assert state["intent"].label == "unsupported"
    assert executor.calls == []
    assert "structured_lookup" not in _trace_nodes(state)
    assert "hybrid_qa" not in _trace_nodes(state)


def test_7e_canonical_only_for_corpus_wide_modes():
    """Corpus-wide lookups filter to canonical documents; scoped do not."""
    executor = FakeSqlExecutor(field_rows=[])
    _run_node(executor, _intent(scope="filtered", target_doc_names=[]))
    assert "d.is_canonical = true" in executor.field_calls[0]["sql"]

    executor2 = FakeSqlExecutor(field_rows=[])
    _run_node(executor2, _intent(target_doc_names=["Bellring"]))
    # doc-scoped queries carry no canonicality FILTER (the is_canonical
    # column is still SELECTed for provenance)
    assert "d.is_canonical = true" not in executor2.field_calls[0]["sql"]


def test_7f_executor_adapter_exists_but_is_not_live_wired():
    """The production adapter is defined for the deployment milestone;
    the graph itself never instantiates it (no hidden DB access)."""
    assert hasattr(StatementExecutionSqlExecutor, "execute")
    import inspect
    import workflow.graph as graph_module
    src = inspect.getsource(graph_module)
    # The graph module imports only the node factory (dependency
    # injection); it never imports or instantiates the live adapter
    # (any docstring mention carries no import/instantiation side effect).
    assert "from .structured_lookup import make_structured_lookup_node" in src
    assert "import StatementExecutionSqlExecutor" not in src
    assert "StatementExecutionSqlExecutor(" not in src


def test_7g_multi_value_docs_with_evidence_attached_per_value():
    """Evidence attaches to the specific value row it supports."""
    executor = FakeSqlExecutor(
        field_rows=[
            _frow(value_index=1, field_id="f1"),
            _frow(value_index=2, field_id="f2"),
        ],
        evidence_rows=[_erow(field_id="f1", quote="quote for value 1"),
                       _erow(field_id="f2", quote="quote for value 2")],
    )
    result = _run_node(executor, _intent())
    values = result["lookup_result"]["results"][0]["values"]
    assert values[0]["evidence"][0]["quote"] == "quote for value 1"
    assert values[1]["evidence"][0]["quote"] == "quote for value 2"


# -- Test group 8: live-adapter compatibility (SDK 0.122.0 verified) ----
# Regression tests for the two live defects found in the 2026-10-28
# smoke test: (1) the parameter class is StatementParameterListItem in
# databricks-sdk 0.122.0 (not StatementParameter); (2) the result manifest
# lives at the TOP LEVEL of the statement response (resp.manifest), not
# on resp.result.

class _FakeCol:
    def __init__(self, name):
        self.name = name


class _FakeManifest:
    def __init__(self, cols):
        self.schema = type("S", (), {"columns": [_FakeCol(c) for c in cols]})()


class _FakeResult:
    def __init__(self, data):
        self.data_array = data


class _FakeStatus:
    def __init__(self):
        self.state = "SUCCEEDED"
        self.error = None


class _FakeResponse:
    """Response shaped like databricks-sdk 0.122.0 StatementResponse:
    manifest at the top level, result holding data_array."""

    def __init__(self, cols, data):
        self.manifest = _FakeManifest(cols)
        self.result = _FakeResult(data)
        self.status = _FakeStatus()


class _FakeWorkspaceClient:
    def __init__(self, cols, data):
        self._resp = _FakeResponse(cols, data)
        self.calls = []
        self.statement_execution = self

    def execute_statement(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def test_8a_binds_named_string_parameters_via_installed_sdk():
    """The adapter resolves the parameter class from the installed SDK and
    binds name/value/type=STRING (no interpolation into the statement)."""
    from workflow.structured_lookup import _statement_parameter_cls

    cls = _statement_parameter_cls()  # must not raise on any SDK version
    p = cls(name="value_pattern", value="%x%", type="STRING")
    assert p.name == "value_pattern" and p.value == "%x%"

    client = _FakeWorkspaceClient(["v"], [["captured"]])
    executor = StatementExecutionSqlExecutor("wh-1", workspace_client=client)
    rows = executor.execute(
        "SELECT :value_pattern AS v", {"value_pattern": "%x%"})
    assert rows == [{"v": "captured"}]

    call = client.calls[0]
    assert call["warehouse_id"] == "wh-1"
    sent = call["parameters"]
    assert sent[0].name == "value_pattern"
    assert sent[0].value == "%x%"
    assert sent[0].type == "STRING"
    # value bound as a parameter object, never embedded in the SQL text
    assert "%x%" not in call["statement"]


def test_8b_reads_manifest_from_top_level_response():
    """Column names come from resp.manifest (SDK 0.122.0 shape); SQL NULLs
    are preserved as None."""
    client = _FakeWorkspaceClient(
        ["doc_id", "file_name"], [["d1", "A.pdf"], ["d2", None]])
    executor = StatementExecutionSqlExecutor("wh-1", workspace_client=client)
    rows = executor.execute("SELECT doc_id, file_name FROM t", {})
    assert rows[0] == {"doc_id": "d1", "file_name": "A.pdf"}
    assert rows[1]["doc_id"] == "d2"
    assert rows[1]["file_name"] is None  # NULL -> None (LEFT JOIN null rows)


def test_8c_empty_result_returns_empty_list():
    """No data_array -> [] (no-manifest crash, no rows invented)."""
    client = _FakeWorkspaceClient(["doc_id"], [])
    executor = StatementExecutionSqlExecutor("wh-1", workspace_client=client)
    assert executor.execute("SELECT doc_id FROM t", {}) == []
