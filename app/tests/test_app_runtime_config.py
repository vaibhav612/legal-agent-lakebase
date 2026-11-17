"""Stage C2 regression tests: explicit hybrid_v2 configuration (Milestone 5).

The integrated app (streamlit_app.py -> app_runtime.build_app_graph) must:
  * REQUIRE an explicit RETRIEVER_MODE=hybrid_v2 — any other value fails
    fast at build time (no silent legacy fallback);
  * reject a retriever instance reporting a different mode;
  * reject a legacy retriever whose search() predates the D1/D2 doc_ids
    document-scoping fix;
  * bind SQL parameters by NAME through the statement-execution executor
    (never interpolate them into the SQL text);
  * coerce warehouse cell types to what the workflow nodes expect; and
  * provide a credential-free trace summary.

All tests run WITHOUT Streamlit, Lakebase, or a warehouse: the app runtime
module is importable standalone (its heavy imports are lazy).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the app directory importable when tests run from a checkout.
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import app_runtime  # noqa: E402
from app_runtime import (  # noqa: E402
    REQUIRED_RETRIEVER_MODE,
    StatementExecutionSqlExecutor,
    _coerce_column_value,
    build_app_graph,
    trace_summary,
)


# ── Section 1: explicit hybrid_v2 mode + fail-fast ─────────────────────────

class FakeHybridRetriever:
    """Minimal stand-in satisfying the app's retriever contract."""

    retriever_mode = REQUIRED_RETRIEVER_MODE

    def search(self, query, num_results=5, doc_ids=None):
        return []

    def generate_response(self, query, context_chunks):
        return "answer"


class FakeLegacyRetriever:
    """A pre-D1 retriever: search() without the doc_ids parameter."""

    retriever_mode = REQUIRED_RETRIEVER_MODE

    def search(self, query, num_results=5):
        return []


class FakeWrongModeRetriever:
    retriever_mode = "legacy"

    def search(self, query, num_results=5, doc_ids=None):
        return []


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append({"sql": sql, "params": params})
        return []


def test_build_app_graph_requires_explicit_hybrid_v2_mode(monkeypatch):
    """RETRIEVER_MODE unset → fail-fast (no silent default, no legacy path)."""
    monkeypatch.delenv("RETRIEVER_MODE", raising=False)
    with pytest.raises(RuntimeError, match="RETRIEVER_MODE"):
        build_app_graph(retriever=FakeHybridRetriever(), sql_executor=FakeExecutor())


@pytest.mark.parametrize("bad", ["legacy", "", "hybrid", "HYBRID_V2", "bm25"])
def test_build_app_graph_rejects_wrong_modes(monkeypatch, bad):
    """Any mode other than exactly 'hybrid_v2' fails fast with the reason."""
    monkeypatch.setenv("RETRIEVER_MODE", bad)
    with pytest.raises(RuntimeError, match="hybrid_v2"):
        build_app_graph(retriever=FakeHybridRetriever(), sql_executor=FakeExecutor())


def test_build_app_graph_accepts_explicit_hybrid_v2(monkeypatch):
    """The explicit mode + a compatible retriever builds the real graph."""
    monkeypatch.setenv("RETRIEVER_MODE", REQUIRED_RETRIEVER_MODE)
    graph = build_app_graph(
        retriever=FakeHybridRetriever(),
        sql_executor=FakeExecutor(),
        answer_generator=lambda q, c: "ok",
    )
    # A compiled LangGraph exposes invoke; the production entry point is intact.
    assert hasattr(graph, "invoke")


def test_build_app_graph_rejects_wrong_mode_retriever(monkeypatch):
    """A retriever instance reporting a non-hybrid_v2 mode is rejected."""
    monkeypatch.setenv("RETRIEVER_MODE", REQUIRED_RETRIEVER_MODE)
    with pytest.raises(RuntimeError, match="mode mismatch"):
        build_app_graph(retriever=FakeWrongModeRetriever(), sql_executor=FakeExecutor())


def test_build_app_graph_rejects_legacy_retriever_api(monkeypatch):
    """A retriever whose search() lacks doc_ids (pre-D1) is rejected — the
    app cannot silently lose document-scoped retrieval."""
    monkeypatch.setenv("RETRIEVER_MODE", REQUIRED_RETRIEVER_MODE)
    with pytest.raises(RuntimeError, match="doc_ids"):
        build_app_graph(retriever=FakeLegacyRetriever(), sql_executor=FakeExecutor())


def test_fail_fast_messages_are_credential_free(monkeypatch):
    """Startup failure messages never leak token material."""
    monkeypatch.setenv("RETRIEVER_MODE", "hybrid_v2")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-SUPERSECRET-should-not-appear")
    retriever = FakeHybridRetriever()
    retriever.retriever_mode = "legacy"
    with pytest.raises(RuntimeError) as excinfo:
        build_app_graph(retriever=retriever, sql_executor=FakeExecutor())
    assert "SUPERSECRET" not in str(excinfo.value)


# ── Section 2: retriever-side mode validation (fail-fast, no fallback) ─────


def test_retriever_module_rejects_unknown_mode():
    """lakebase_retriever's validate_retriever_mode raises on any mode other
    than hybrid_v2/legacy (Stage C2) — an unrecognized RETRIEVER_MODE can
    never silently fall through to the legacy search path."""
    import lakebase_retriever as lr

    # recognized modes pass
    lr.validate_retriever_mode("hybrid_v2")
    lr.validate_retriever_mode("legacy")

    # anything else fails fast with a clear, credential-free message
    for bad in ["", "hybrid", "HYBRID_V2", "bm25", "semantic"]:
        with pytest.raises(ValueError, match="RETRIEVER_MODE"):
            lr.validate_retriever_mode(bad)


# ── Section 3: statement-execution executor binding + coercion ────────────

class _FakeResp:
    class _Status:
        def __init__(self, state):
            self.state = state

    class _Manifest:
        def __init__(self, columns):
            from types import SimpleNamespace
            self.schema = SimpleNamespace(
                columns=[SimpleNamespace(name=n, type_name=t) for n, t in columns]
            )

    class _Result:
        def __init__(self, rows):
            self.data_array = rows

    def __init__(self, state, columns, rows):
        self.status = self._Status(state)
        self.manifest = self._Manifest(columns)
        self.result = self._Result(rows)
        self.error = None


class FakeStatementClient:
    """Records the execute_statement call; returns scripted responses."""

    def __init__(self, response=None, state="SUCCEEDED"):
        self.statement_execution = self
        self.calls = []
        self.response = response
        self.state = state

    def execute_statement(self, **kwargs):
        self.calls.append(kwargs)
        if self.response is not None:
            return self.response
        return _FakeResp(self.state, [], [])


def _succeeded(columns, rows):
    from databricks.sdk.service.sql import StatementState
    return _FakeResp(StatementState.SUCCEEDED, columns, rows)


def test_executor_binds_named_params_never_interpolates():
    """Params are passed to the API as named parameters — the SQL text is
    untouched (no literal interpolation, no injection surface)."""
    client = FakeStatementClient(response=_succeeded([("n", "INT")], [["3"]]))
    ex = StatementExecutionSqlExecutor(client=client)
    sql = "SELECT file_name FROM t WHERE name = :nm AND kind LIKE :pat"
    rows = ex.execute(sql, params={"nm": "O'Brien", "pat": "%x'; DROP TABLE t;--"})
    sent = client.calls[0]
    assert sent["statement"] == sql  # SQL text unchanged
    bound = {p.name: p.value for p in sent["parameters"]}
    assert bound["nm"] == "O'Brien"
    assert "DROP TABLE" in bound["pat"]
    assert "DROP TABLE" not in sent["statement"]
    assert rows == [{"n": 3}]


def test_executor_coerces_types_from_manifest():
    """BOOLEAN/INT/DOUBLE cells become Python types the nodes compare."""
    client = FakeStatementClient(response=_succeeded(
        [("doc_id", "STRING"), ("is_canonical", "BOOLEAN"),
         ("page_start", "INT"), ("score", "DOUBLE"), ("value_raw", "STRING")],
        [["abc", "true", "12", "0.5", "Delaware"],
         ["def", "false", None, None, None]],
    ))
    ex = StatementExecutionSqlExecutor(client=client)
    rows = ex.execute("SELECT ...")
    assert rows[0]["is_canonical"] is True
    assert rows[0]["page_start"] == 12
    assert rows[0]["score"] == 0.5
    assert rows[0]["value_raw"] == "Delaware"
    assert rows[1]["is_canonical"] is False
    assert rows[1]["page_start"] is None


def test_executor_raises_on_failed_state():
    """A failed statement becomes a credential-free RuntimeError with state."""
    client = FakeStatementClient(state="FAILED")
    ex = StatementExecutionSqlExecutor(client=client)
    with pytest.raises(RuntimeError, match="FAILED"):
        ex.execute("SELECT 1")


def test_coerce_column_value_table():
    from types import SimpleNamespace

    assert _coerce_column_value("BOOLEAN", "true") is True
    assert _coerce_column_value("BOOLEAN", "false") is False
    assert _coerce_column_value("BIGINT", "42") == 42
    assert _coerce_column_value("DOUBLE", "1.5") == 1.5
    assert _coerce_column_value("STRING", "x") == "x"
    assert _coerce_column_value(None, None) is None
    assert _coerce_column_value("DOUBLE", "not-a-number") == "not-a-number"
    # The live manifest reports enum values (ColumnInfoTypeName), not plain
    # strings — regression for the app integration run (2026-10-29).
    assert _coerce_column_value(SimpleNamespace(value="BOOLEAN"), "true") is True
    assert _coerce_column_value(SimpleNamespace(value="INT"), "7") == 7


# ── Section 4: credential-free trace summary ──────────────────────────────

def test_trace_summary_only_node_action_reason():
    """The UI diagnostic helper never includes SQL, parameters, or text
    payloads — node/action/reason only."""
    state = {
        "trace": [
            {"node": "query_planner", "action": "classified",
             "reason": "hybrid_qa", "details": {"params": {"secret": "x"}}},
            {"node": "hybrid_qa", "action": "retrieved",
             "reason": "5 results", "scores": [{"chunk_text": "full text"}]},
        ]
    }
    summary = trace_summary(state)
    assert [s["node"] for s in summary] == ["query_planner", "hybrid_qa"]
    dumped = str(summary)
    assert "secret" not in dumped
    assert "full text" not in dumped
    assert "chunk_text" not in dumped
