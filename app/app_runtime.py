"""App runtime wiring for the Legal Intelligence Assistant (Milestone 5).

Bridges the Streamlit app to the LangGraph capability workflow with REAL
dependencies only (no testing stubs):

* LakebaseRetriever in hybrid_v2 mode — semantic (pgvector) + BM25 channels
  with RRF fusion, and document filters applied INSIDE both channels before
  fusion.  Also provides grounded answer generation for hybrid Q&A.
* StatementExecutionSqlExecutor — read-only warehouse SQL over the Databricks
  SQL Statement Execution API with NAMED PARAMETER BINDING (the workflow's
  ``:name`` placeholders are bound by the warehouse, never interpolated into
  the SQL text).
* RuleBasedIntentClassifier — the deterministic, unit-tested intent router
  used by every live verification run (Milestone 2–5).

Fail-fast contract (spec Section 1.2 / D2 remediation): the app requires an
EXPLICIT ``RETRIEVER_MODE=hybrid_v2``.  Any missing dependency, wrong
retrieval mode, or incompatible retriever API raises at build time with a
clear, credential-free message — there is no silent fallback to a legacy
retrieval path.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger("legal_assistant.app_runtime")

#: The only retrieval mode the app is allowed to run (Stage C2).
REQUIRED_RETRIEVER_MODE = "hybrid_v2"

#: Default SQL warehouse — must match the ``sql-warehouse`` resource in app.yaml.
DEFAULT_SQL_WAREHOUSE_ID = "c99228c48630a173"

#: Statement execution wait timeout (interactive app responses).
#: The API accepts 0s (no wait) or 5s-50s — 50s is the maximum wait that
#: still returns results inline for interactive queries.
DEFAULT_WAIT_TIMEOUT = "50s"


class SqlExecutorLike(Protocol):
    """Structural type accepted by the workflow's SQL-backed nodes."""

    def execute(self, sql: str, params: Optional[dict] = None) -> list[dict]:
        ...


def _coerce_column_value(type_name: Optional[str], raw: Optional[str]) -> Any:
    """Coerce a JSON_ARRAY cell to the Python type the workflow expects.

    The statement execution API returns every cell as a string (or null).
    The workflow's capability nodes compare booleans (``is_canonical``) and
    use counts as integers, so the manifest column types drive coercion.
    """
    if raw is None:
        return None
    # The manifest reports column types as ColumnInfoTypeName enum values
    # (or plain strings in tests) — normalize before comparing.
    t = str(getattr(type_name, "value", type_name) or "").upper()
    if t in {"BOOLEAN", "BOOL"}:
        return raw.strip().lower() == "true"
    if t in {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"}:
        return int(raw)
    if t in {"DOUBLE", "FLOAT", "DECIMAL", "REAL"}:
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


class StatementExecutionSqlExecutor:
    """Read-only SQL executor over the Databricks SQL Statement Execution API.

    Satisfies the workflow's SqlExecutorLike protocol.  Named parameters
    (``:name``) are bound by the warehouse via StatementParameterListItem —
    parameter values are never interpolated into the SQL text.
    """

    def __init__(
        self,
        warehouse_id: str = DEFAULT_SQL_WAREHOUSE_ID,
        client: Optional[Any] = None,
        wait_timeout: str = DEFAULT_WAIT_TIMEOUT,
    ):
        if client is None:
            from databricks.sdk import WorkspaceClient
            client = WorkspaceClient()
        self._client = client
        self.warehouse_id = warehouse_id
        self.wait_timeout = wait_timeout

    def execute(self, sql: str, params: Optional[dict] = None) -> list[dict]:
        from databricks.sdk.service.sql import (
            Disposition,
            Format,
            StatementParameterListItem,
            StatementState,
        )

        parameters = None
        if params:
            parameters = [
                StatementParameterListItem(name=str(k), value=str(v))
                for k, v in params.items()
            ]

        resp = self._client.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=self.warehouse_id,
            parameters=parameters,
            wait_timeout=self.wait_timeout,
            disposition=Disposition.INLINE,
            format=Format.JSON_ARRAY,
        )

        # Deployment fix 002: a statement can still be PENDING when it
        # exceeds the inline wait window (the API caps wait_timeout at 50s) —
        # poll via get_statement until a terminal state instead of treating
        # PENDING as a failure.
        state = getattr(resp.status, "state", None)
        polls = 0
        while state == StatementState.PENDING and polls < 30:
            time.sleep(2.0)
            polls += 1
            resp = self._client.statement_execution.get_statement(resp.statement_id)
            state = getattr(resp.status, "state", None)

        if state != StatementState.SUCCEEDED:
            # Deployment fix 002 diagnostics: in the databricks-sdk used by
            # the hosted app, the SQL error lives at resp.status.error (a
            # ServiceError with error_code + message) plus
            # resp.status.sql_state — NOT at resp.error. Reading the wrong
            # path is why the first hosted failures printed an empty message
            # after the colon. Surface statement id, error code, SQL state and
            # message; none of these fields carry credentials or tokens.
            status = getattr(resp, "status", None)
            err = getattr(status, "error", None)
            err_code = getattr(err, "error_code", None)
            err_code = getattr(err_code, "value", err_code)
            err_msg = (getattr(err, "message", None) or "").strip()
            sql_state = getattr(status, "sql_state", None)
            statement_id = getattr(resp, "statement_id", None)
            parts: list[str] = []
            if err_code is not None:
                parts.append(f"error_code={err_code}")
            if sql_state is not None:
                parts.append(f"sql_state={sql_state}")
            if statement_id is not None:
                parts.append(f"statement_id={statement_id}")
            if err_msg:
                parts.append(err_msg[:250])
            if state == StatementState.PENDING:
                parts.append(
                    f"query still PENDING after {polls} polls (~60s) — "
                    "too slow for inline results"
                )
            raise RuntimeError(
                f"Warehouse query failed (state={state}): " + " | ".join(parts)
            )

        manifest_columns = []
        if resp.manifest is not None and resp.manifest.schema is not None:
            manifest_columns = list(resp.manifest.schema.columns)
        names = [c.name for c in manifest_columns]
        types = [getattr(c, "type_name", None) for c in manifest_columns]

        rows: list[dict] = []
        for arr in (resp.result.data_array or []) if resp.result else []:
            row: dict = {}
            for idx, cell in enumerate(arr):
                if idx < len(names):
                    row[names[idx]] = _coerce_column_value(
                        types[idx] if idx < len(types) else None, cell
                    )
            rows.append(row)
        logger.debug("executor: %d rows", len(rows))
        return rows


def _validate_retriever(retriever: Any) -> None:
    """Fail-fast checks that the injected retriever is the hybrid_v2 one.

    Guards against two known failure modes:
      1. a stale/legacy retriever instance whose search() predates the D1/D2
         document-scoping fix (no ``doc_ids`` parameter); and
      2. a retriever configured for a non-hybrid_v2 mode.
    """
    actual_mode = getattr(retriever, "retriever_mode", None)
    if actual_mode != REQUIRED_RETRIEVER_MODE:
        raise RuntimeError(
            "Retriever mode mismatch: the app requires "
            f"{REQUIRED_RETRIEVER_MODE!r} but the retriever reports {actual_mode!r}. "
            "The app never silently falls back to a legacy retrieval path."
        )
    try:
        sig = inspect.signature(retriever.search)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Incompatible retriever (no inspectable search()): {exc}")
    if "doc_ids" not in sig.parameters:
        raise RuntimeError(
            "Incompatible retriever: search() must accept doc_ids "
            "(document-scoped retrieval, D1/D2). Found signature "
            f"{sig} — a legacy retriever may have been packaged."
        )


def build_app_graph(
    retriever: Optional[Any] = None,
    sql_executor: Optional[SqlExecutorLike] = None,
    answer_generator: Optional[Any] = None,
    num_results: int = 5,
) -> Any:
    """Build the PRODUCTION workflow graph with real capability dependencies.

    Called once per app process by ``streamlit_app.get_workflow()`` (cached).
    Raises RuntimeError (fail-fast, credential-free) when:
      * ``RETRIEVER_MODE`` is not explicitly set to ``hybrid_v2``;
      * the retriever instance reports a different mode;
      * the retriever's ``search`` lacks the ``doc_ids`` parameter; or
      * any real dependency cannot be constructed.
    """
    mode = os.environ.get("RETRIEVER_MODE", "")
    if mode != REQUIRED_RETRIEVER_MODE:
        raise RuntimeError(
            "RETRIEVER_MODE must be explicitly set to 'hybrid_v2' for the app "
            f"(got {mode!r}). Add RETRIEVER_MODE: hybrid_v2 to the app "
            "environment; the app never silently falls back to a legacy "
            "retrieval path."
        )

    if retriever is None:
        from lakebase_retriever import get_lakebase_retriever
        retriever = get_lakebase_retriever()
    _validate_retriever(retriever)

    if sql_executor is None:
        sql_executor = StatementExecutionSqlExecutor()

    if answer_generator is None:
        answer_generator = retriever.generate_response

    from workflow.classifier import RuleBasedIntentClassifier
    from workflow.graph import build_graph

    graph = build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        answer_generator=answer_generator,
        num_results=num_results,
        sql_executor=sql_executor,
    )
    logger.info("app graph built (mode=%s, num_results=%d)", mode, num_results)
    return graph


def trace_summary(state: dict) -> list[dict]:
    """Credential-free diagnostic summary of a completed graph state.

    Returns one compact entry per node (node, action, reason) so the UI and
    logs can show what the workflow did without exposing internals or
    credentials.  Full evidence and SQL parameters are NOT included.
    """
    summary: list[dict] = []
    for entry in state.get("trace") or []:
        if isinstance(entry, dict):
            summary.append({
                "node": entry.get("node"),
                "action": entry.get("action"),
                "reason": str(entry.get("reason"))[:200] if entry.get("reason") else None,
            })
    return summary