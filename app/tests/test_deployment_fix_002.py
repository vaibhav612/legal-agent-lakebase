"""Regression tests for hosted deployment failure 002 (2026-10-29).

Two confirmed defects from the first hosted smoke test:

1. Warehouse statements FAILED for both corpus-overview and structured
   lookup. Root cause (see docs/deployment fix record 002): the Apps
   platform never applied the app.yaml ``uc_securable`` resources — the
   registered resources were stale June-era UI-configured entries — so the
   app service principal had no USE SCHEMA / SELECT on
   workspace.legal_assistant_v2. Fixed with direct owner grants. These tests
   lock the DIAGNOSTICS side of the same failure: the original executor read
   ``resp.error`` — an attribute that does not exist in the databricks-sdk
   used by the hosted app (the SQL error lives at ``resp.status.error`` +
   ``resp.status.sql_state``) — so hosted failures printed an empty message
   after the colon. A FAILED statement must now surface error_code, sql_state,
   statement_id and the message.
2. A pure greeting ('hi') got the confused "not sure what you're asking"
   clarification instead of a welcome: greetings fell into the <3-word
   ambiguity branch. The classifier now routes pure greetings to a welcome
   message via the existing clarification route (no corpus/warehouse
   query). The frozen 7-intent taxonomy is unchanged.
"""

from __future__ import annotations

import pytest


def _sid():
    from databricks.sdk.service.sql import StatementState
    return StatementState


# ── Section 1: executor diagnostics (deployment fix 002) ──────────────────


class _FakeStatus:
    def __init__(self, state, error=None, sql_state=None):
        self.state = state
        self.error = error
        self.sql_state = sql_state


class _FakeResp:
    def __init__(self, state, error=None, sql_state=None,
                 columns=(), rows=(), statement_id="sid-test-0001"):
        from types import SimpleNamespace
        self.status = _FakeStatus(state, error, sql_state)
        self.statement_id = statement_id
        self.manifest = SimpleNamespace(
            schema=SimpleNamespace(
                columns=[SimpleNamespace(name=n, type_name=t) for n, t in columns]
            )
        )
        self.result = SimpleNamespace(data_array=list(rows))


class _FakeClient:
    """Scripted statement-execution client (records every call)."""

    def __init__(self, execute_resp, get_responses=()):
        self.statement_execution = self
        self.execute_resp = execute_resp
        self.get_responses = list(get_responses)
        self.get_calls: list[str] = []

    def execute_statement(self, **kwargs):
        return self.execute_resp

    def get_statement(self, statement_id):
        self.get_calls.append(statement_id)
        return self.get_responses.pop(0)


def _executor(client):
    from app_runtime import StatementExecutionSqlExecutor
    return StatementExecutionSqlExecutor(client=client)


def test_failed_statement_surfaces_full_diagnostics():
    """FAILED must surface error_code, sql_state, statement_id and the SQL
    error message — the original code read the non-existent resp.error path
    and printed an empty message (exact hosted failure 002 signature)."""
    from databricks.sdk.service.sql import ServiceError, StatementState

    resp = _FakeResp(
        StatementState.FAILED,
        error=ServiceError(
            error_code="PERMISSION_DENIED",
            message="[PERMISSION_DENIED] User does not have SELECT on table "
                    "workspace.legal_assistant_v2.documents",
        ),
        sql_state="42501",
    )
    ex = _executor(_FakeClient(resp))
    with pytest.raises(RuntimeError) as exc_info:
        ex.execute("SELECT 1")
    msg = str(exc_info.value)
    assert "Warehouse query failed" in msg
    assert "FAILED" in msg
    assert "error_code=PERMISSION_DENIED" in msg
    assert "sql_state=42501" in msg
    assert "statement_id=sid-test-0001" in msg
    assert "PERMISSION_DENIED" in msg  # SQL error message present
    # not empty after the colon (the original hosted signature)
    assert not msg.rstrip().endswith(":")


def test_failed_state_without_error_fields_still_names_state():
    """A FAILED response with no error object still raises with the state."""
    resp = _FakeResp(_sid().FAILED)
    ex = _executor(_FakeClient(resp))
    with pytest.raises(RuntimeError, match="FAILED"):
        ex.execute("SELECT 1")


def test_pending_statement_is_polled_not_failed(monkeypatch):
    """PENDING (statement exceeded the 50s inline wait) must be polled via
    get_statement until terminal — not misreported as a failure."""
    S = _sid()
    monkeypatch.setattr("app_runtime.time.sleep", lambda *_: None)
    pending = _FakeResp(S.PENDING, statement_id="sid-poll-1")
    done = _FakeResp(S.SUCCEEDED, columns=[("n", "INT")], rows=[["7"]],
                     statement_id="sid-poll-1")
    client = _FakeClient(pending, get_responses=[done])
    rows = _executor(client).execute("SELECT 7 AS n")
    assert rows == [{"n": 7}]
    assert client.get_calls == ["sid-poll-1"]


def test_diagnostics_never_include_parameter_values():
    """The surfaced parts come only from status.error/sql_state/statement_id
    — no parameter values, no SQL text echo, no credential material."""
    from databricks.sdk.service.sql import ServiceError, StatementState

    resp = _FakeResp(
        StatementState.FAILED,
        error=ServiceError(error_code="BAD_REQUEST", message="oops"),
        sql_state="42000",
    )
    ex = _executor(_FakeClient(resp))
    with pytest.raises(RuntimeError) as exc_info:
        ex.execute("SELECT :secret_param", params={"secret_param": "tok-do-not-leak"})
    msg = str(exc_info.value)
    assert "tok-do-not-leak" not in msg
    assert "secret_param" not in msg


# ── Section 2: greeting routing (deployment fix 002) ──────────────────────


def _classifier():
    from workflow.classifier import RuleBasedIntentClassifier
    return RuleBasedIntentClassifier()


@pytest.mark.parametrize("greeting", [
    "hi", "HI", "Hi!", "hello", "Hello there.", "hey", "good morning",
    "hi?", "hello there",
])
def test_pure_greeting_gets_welcome_not_confusion(greeting):
    intent = _classifier().classify(greeting)
    assert intent.label == "unsupported"  # frozen taxonomy unchanged
    assert intent.clarification_needed
    assert "Hello" in intent.clarification_needed
    assert "not sure" not in intent.clarification_needed.lower()


def test_greeting_word_does_not_hijack_real_questions():
    """A greeting prefix on a real question still routes by content."""
    intent = _classifier().classify(
        "Hi, what is the governing law of the Armstrong Flooring intellectual "
        "property agreement?"
    )
    assert intent.label == "structured_lookup"
    assert intent.target_field == "governing_law"
    assert intent.clarification_needed is None


def test_greeting_flows_through_graph_without_corpus_query():
    """End-to-end: the greeting intent routes to clarification_node whose
    message becomes the answer — and no SQL executor is ever consulted
    (proven by building the graph with a counting fake executor)."""
    from workflow.classifier import IntentClassifier
    from workflow.graph import build_graph
    from workflow.state import Intent

    class _GreetingClassifier(IntentClassifier):
        def classify(self, query, conversation_history=None):
            return Intent(label="unsupported", scope="all",
                          clarification_needed="Hello! Welcome message.")

    calls = []

    class _CountingExecutor:
        def execute(self, sql, params=None):
            calls.append(sql)
            return []

    g = build_graph(classifier=_GreetingClassifier(), sql_executor=_CountingExecutor())
    final = g.invoke({"user_query": "hi", "conversation_history": []})
    assert "Hello! Welcome message." in (final.get("answer") or "")
    assert calls == []  # no warehouse/corpus SQL for a greeting
