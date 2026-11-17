"""Structured Lookup capability node (Milestone 3; spec Section 1.5, 2.5-D).

Answers questions about the 10 extracted contract fields using structured
SQL over `workspace.legal_assistant_v2.extracted_fields` (joined to
`documents` for provenance and `evidence` for supporting quotes).  No
retrieval, no LLM generation of the value itself - raw extracted data is
returned with its `value_status` and extraction method.

Safety model (spec Section 1.5 + Milestone 3 task constraints):
  - Field names are allowlisted (ALLOWED_FIELDS in state.py) and are the
    ONLY identifiers interpolated into SQL text, after a membership check.
  - Every user-derived filter value (document-name fragments, value
    patterns, status names, evidence field_ids) is bound as a named SQL
    parameter - never concatenated into the statement.
  - Row limits are internal integer constants (never user input).
  - Table names are hardcoded module constants.
  - The LLM never generates SQL; it produces a validated Intent, and this
    node translates it into one of three fixed query templates:
      * lookup        - field values for the target document(s) / corpus
      * value_filter  - documents whose field matches a value fragment
      * status_filter - documents with missing/uncertain field values

Missing-value semantics (spec Section 1.5 - normative):
  1. found_explicit / inferred / conflicting / redacted rows are extracted
     values with their status reported verbatim.
  2. not_found rows mean the extractor found no value - reported as
     "no extracted value available".
  3. A LEFT JOIN null row means NO extraction record exists for that
     document+field (document not covered by the extraction run).
  4. An empty result set means no document matched the query scope.
  5. An executor exception is a query/database error, recorded in state.

A missing or uncertain extraction is NEVER presented as proof that the
underlying clause or obligation is absent from the contract - clause
absence requires a separate evidence-based assessment (clause discovery
or hybrid Q&A).

Run multiplicity (verified 2026-10-28): extracted_fields contains 4
extraction runs (latest full-corpus run f495e53a..., 728 rows / 56 docs).
Queries deduplicate with ROW_NUMBER() OVER (PARTITION BY doc_id,
field_name, value_index ORDER BY created_at DESC, field_id DESC) so each
document+field+value_index appears once, from the latest run.

Dependency injection follows the hybrid_qa pattern (Milestone 2B.1): the
node factory receives a SqlExecutorLike; unit tests pass a fake, and
production passes StatementExecutionSqlExecutor wrapping the existing
Databricks SDK statement_execution pattern (the same SQL-warehouse access
path already used by chat_history.py - but parameterized).  The adapter
was live-verified 2026-10-28 against warehouse c99228c48630a173
(databricks-sdk 0.122.0) - see the live smoke-test section of the
structured lookup integration report.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .state import ALLOWED_FIELDS, MAX_SOURCE_DOCS
from .traces import make_trace


# -- Constants (verified against information_schema 2026-10-28) ----------

EXTRACTED_FIELDS_TABLE = "workspace.legal_assistant_v2.extracted_fields"
DOCUMENTS_TABLE = "workspace.legal_assistant_v2.documents"
EVIDENCE_TABLE = "workspace.legal_assistant_v2.evidence"
# Stage 3 fix (2026-10-29): the evidence query now joins chunks_v2 to fetch
# section_path for citations (the authoritative per-chunk section metadata).
CHUNKS_TABLE = "workspace.legal_assistant_v2.chunks_v2"

# Row limit for the field query (bounded; internal constant, not user input)
MAX_LOOKUP_ROWS: int = 200
# Maximum documents whose file-name fragments a lookup may filter on
MAX_LOOKUP_DOC_FILTERS: int = 10
# Linked evidence rows kept per extracted value
MAX_EVIDENCE_PER_VALUE: int = 3
# Maximum characters of a value filter fragment (bound as a parameter)
MAX_VALUE_FILTER_CHARS: int = 100

# Actual value_status values in extracted_fields (verified via SQL)
VALUE_STATUSES: tuple = (
    "found_explicit", "not_found", "inferred", "redacted", "conflicting",
)

# status_filter (Intent field) -> value_status values it covers
STATUS_FILTER_TO_STATUSES: dict = {
    "missing": ("not_found",),
    "uncertain": ("inferred", "conflicting"),
    "missing_or_uncertain": ("not_found", "inferred", "conflicting"),
}

MISSING_CAVEAT = (
    "A missing or uncertain extraction is not proof that the corresponding "
    "clause or obligation is absent from the contract."
)


# -- Executor protocol (injected; no live database needed for tests) ----

class SqlExecutorLike(Protocol):
    """Structural type of the injected SQL executor.

    Contract: execute a read-only statement with named parameters and
    return rows as a list of column-name -> value dicts (values may be
    None for SQL NULL).  Implementations MUST keep the statement text and
    the parameter values separate (no client-side string interpolation of
    parameter values into the SQL text).
    """

    def execute(self, sql: str, params: dict) -> list:
        ...


def _statement_parameter_cls():
    """Resolve the named-parameter class across databricks-sdk versions.

    The class name differs across SDK releases (verified live 2026-10-28:
    `StatementParameterListItem` in databricks-sdk 0.122.0; some other
    versions expose `StatementParameter`).  Both accept (name, value,
    type) with type="STRING".
    """
    from databricks.sdk.service import sql as svc  # lazy import
    for cls_name in ("StatementParameter", "StatementParameterListItem"):
        cls = getattr(svc, cls_name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "databricks-sdk service.sql exposes neither StatementParameter "
        "nor StatementParameterListItem - cannot bind named parameters"
    )


class StatementExecutionSqlExecutor:
    """Production executor over the Databricks SQL Statement Execution API.

    Reuses the app's existing access pattern (chat_history.py executes on
    SQL warehouse c99228c48630a173 via the WorkspaceClient) but binds all
    filter values as named STRING parameters instead of interpolating them
    into the statement text.

    Live-verified 2026-10-28 against warehouse c99228c48630a173 on
    databricks-sdk 0.122.0 (session user OAuth): named parameters bind,
    and the result manifest is read from the TOP-LEVEL of the statement
    response (r.manifest, not r.result.manifest as assumed pre-fix).
    """

    def __init__(self, warehouse_id: str, workspace_client=None):
        self.warehouse_id = warehouse_id
        if workspace_client is not None:
            self.w = workspace_client
        else:
            # Lazy import keeps unit-test environments hermetic
            from databricks.sdk import WorkspaceClient
            self.w = WorkspaceClient()

    def execute(self, sql: str, params: dict) -> list:
        param_cls = _statement_parameter_cls()
        parameters = [
            param_cls(name=k, value=v, type="STRING")
            for k, v in (params or {}).items()
        ]
        resp = self.w.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id,
            statement=sql,
            parameters=parameters or None,
            wait_timeout="30s",
        )
        if resp.status and getattr(resp.status, "error", None):
            msg = getattr(resp.status.error, "message", None) or str(resp.status)
            raise RuntimeError(f"statement failed: {msg}")
        if not resp.result or not getattr(resp.result, "data_array", None):
            return []
        # Manifest location differs across SDK versions (verified live
        # 2026-10-28 on 0.122.0: top-level resp.manifest); tolerate both.
        manifest = getattr(resp, "manifest", None)
        if manifest is None:
            manifest = getattr(resp.result, "manifest", None)
        columns = [c.name for c in manifest.schema.columns]
        rows = []
        for arr in resp.result.data_array:
            row = {}
            for i, col in enumerate(columns):
                row[col] = arr[i] if i < len(arr) else None
            rows.append(row)
        return rows


# -- Query templates (fixed; only allowlisted field name interpolated) ---

_FIELD_QUERY_TEMPLATE = """
WITH latest AS (
  SELECT doc_id, field_id, value_index, value_raw, value_normalized,
         value_status, extraction_method, run_id,
         ROW_NUMBER() OVER (
           PARTITION BY doc_id, field_name, value_index
           ORDER BY created_at DESC, field_id DESC
         ) AS rn
  FROM {fields_table}
  WHERE field_name = '{field_name}'
)
SELECT d.doc_id, d.file_name, d.is_canonical, d.document_class,
       d.file_path, d.page_count,
       l.field_id, l.value_index, l.value_raw, l.value_normalized,
       l.value_status, l.extraction_method, l.run_id
FROM {documents_table} d
LEFT JOIN latest l
  ON l.doc_id = d.doc_id AND l.rn = 1
{where_clause}
ORDER BY d.file_name, l.value_index
LIMIT {row_limit}
""".strip()

_EVIDENCE_QUERY_TEMPLATE = """
SELECT e.field_id, e.evidence_id, e.chunk_id, e.quote, e.page_start, e.page_end,
       e.evidence_type, e.rects, e.rect_source, c.section_path
FROM {evidence_table} e
LEFT JOIN {chunks_table} c ON e.chunk_id = c.chunk_id
WHERE e.field_id IN ({field_id_placeholders})
ORDER BY e.field_id, e.evidence_id
LIMIT {row_limit}
""".strip()


# -- Helpers -------------------------------------------------------------

def _as_bool(value: Any) -> Optional[bool]:
    """Tolerantly parse booleans (executors may return 'true'/'false'/None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _row_limit_literal(limit: Optional[int]) -> int:
    """Clamp an internal row limit to [1, MAX_LOOKUP_ROWS].

    The limit is an internal integer constant - never user input - and is
    interpolated as a plain integer literal (parameterized LIMIT is not
    reliably supported across engines).
    """
    if limit is None:
        return MAX_LOOKUP_ROWS
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return MAX_LOOKUP_ROWS
    return max(1, min(n, MAX_LOOKUP_ROWS))


def _build_field_query(field_name: str, scope: str, doc_names: list,
                       target_value: Optional[str], status_filter: Optional[str],
                       row_limit: int):
    """Build the parameterized field query for the lookup request.

    Only `field_name` (already validated against ALLOWED_FIELDS) is
    interpolated into the SQL text.  Everything user-derived is bound as
    a named :parameter.
    """
    params = {}
    conditions = []

    doc_names = list(doc_names or [])[:MAX_LOOKUP_DOC_FILTERS]
    if doc_names:
        # scope = specific_docs: filter documents by file-name fragment
        # (parameterized LIKE patterns).  Preserves duplicate-name matches
        # (e.g. 'Bellring' matches Manufacturing Agreement1 and Agreement2).
        like_clauses = []
        for i, name in enumerate(doc_names):
            key = "doc_%d" % i
            params[key] = "%" + str(name or "").lower() + "%"
            like_clauses.append("LOWER(d.file_name) LIKE :" + key)
        conditions.append("(" + " OR ".join(like_clauses) + ")")
    elif scope != "specific_docs":
        # Corpus-wide modes (all / filtered) are bounded to the eligible
        # corpus: canonical documents only (spec Section 1.3).
        conditions.append("d.is_canonical = true")

    if target_value:
        # mode = value_filter: documents whose field matches the fragment.
        # LEFT-JOIN null rows cannot match a value and are excluded.
        #
        # D10 fix (2026-10-29): the user's filter fragment can be a SUPERSET
        # of the stored value ('Delaware law' vs value 'Delaware') -- the old
        # single-direction LIKE matched nothing in that case.  Matching is now
        # bidirectional: the fragment contains the value, OR the value
        # contains the fragment.  The reverse direction is word-boundary
        # anchored (via RLIKE on a value normalized to [a-z0-9 ] so it can
        # never inject regex metacharacters) so a bare generic word like
        # 'law' cannot match unrelated short values.
        fragment = str(target_value)[:MAX_VALUE_FILTER_CHARS].lower().strip()
        params["value_pattern"] = "%" + fragment + "%"
        params["value_token"] = fragment
        reverse_raw = (
            "REGEXP_REPLACE(TRIM(LOWER(COALESCE(l.value_raw, ''))), "
            "'[^a-z0-9 ]', ' ')"
        )
        reverse_normalized = (
            "REGEXP_REPLACE(TRIM(LOWER(COALESCE(l.value_normalized, ''))), "
            "'[^a-z0-9 ]', ' ')"
        )
        # NOTE (D10 live fix 2026-10-29): the condition opens TWO groups —
        # "(l.field_id IS NOT NULL" and the inner OR group — so the tail
        # must close CONCAT + inner group + outer group: THREE ')' after the
        # last regex literal.  (The first live run shipped two and was a
        # SQLSTATE 42601 parse error — caught by the live B-phase
        # value-filter check, now regression-tested for paren balance.)
        conditions.append(
            "(l.field_id IS NOT NULL AND ("
            "LOWER(COALESCE(l.value_raw, '')) LIKE :value_pattern "
            "OR LOWER(COALESCE(l.value_normalized, '')) LIKE :value_pattern "
            "OR :value_token RLIKE CONCAT('(^|[^a-z0-9])', " + reverse_raw + ", '([^a-z0-9]|$)') "
            "OR :value_token RLIKE CONCAT('(^|[^a-z0-9])', " + reverse_normalized + ", '([^a-z0-9]|$)')))"
        )

    if status_filter:
        statuses = STATUS_FILTER_TO_STATUSES[status_filter]
        placeholders = []
        for i, st in enumerate(statuses):
            key = "status_%d" % i
            params[key] = st
            placeholders.append(":" + key)
        cond = "l.value_status IN (" + ", ".join(placeholders) + ")"
        if status_filter in ("missing", "missing_or_uncertain"):
            # A document with NO extraction record at all also counts as
            # 'missing' - but never as 'uncertain'.
            cond = "(l.field_id IS NULL OR " + cond + ")"
        conditions.append(cond)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    sql = _FIELD_QUERY_TEMPLATE.format(
        fields_table=EXTRACTED_FIELDS_TABLE,
        documents_table=DOCUMENTS_TABLE,
        field_name=field_name,  # allowlisted identifier (validated by caller)
        where_clause=where_clause,
        row_limit=row_limit,
    )
    return sql, params


def _parse_rects(raw):
    """Normalise a stored rects/bboxes column (array<struct<page_id, coord>>)
    into a list of {'page_id': int, 'coord': [x1, y1, x2, y2]} dicts.

    The warehouse returns the column as a JSON string; some executors may
    already decode it. Tolerant parse: anything unrecognised yields [] --
    the source viewer then reports exact highlighting as unavailable
    (never a guessed highlight).
    """
    import json as _json
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("page_id"))
            coord = [int(c) for c in (item.get("coord") or [])]
        except (TypeError, ValueError):
            continue
        if len(coord) == 4 and pid >= 0:
            out.append({"page_id": pid, "coord": coord})
    return out


def _query_evidence(executor: SqlExecutorLike, field_ids: list, row_limit: int):
    """Fetch linked evidence rows for the given field_ids (all bound).

    Returns ({field_id: [evidence dicts]}, error_message or None).
    Degrades gracefully: if the evidence query fails, the field values
    already fetched remain usable and the failure is recorded.
    """
    if not field_ids:
        return {}, None
    placeholders = []
    params = {}
    for i, fid in enumerate(field_ids):
        key = "fid_%d" % i
        params[key] = str(fid)
        placeholders.append(":" + key)
    sql = _EVIDENCE_QUERY_TEMPLATE.format(
        evidence_table=EVIDENCE_TABLE,
        chunks_table=CHUNKS_TABLE,
        field_id_placeholders=", ".join(placeholders),
        row_limit=row_limit,
    )
    try:
        rows = executor.execute(sql, params)
    except Exception as exc:  # evidence linkage is supplemental - degrade
        return {}, ("Evidence linkage query failed: " + str(exc))[:200]
    by_field = {}
    for row in rows:
        fid = str(row.get("field_id") or "")
        if not fid:
            continue
        by_field.setdefault(fid, []).append({
            "evidence_id": row.get("evidence_id"),
            "chunk_id": row.get("chunk_id"),
            "quote": row.get("quote"),
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "evidence_type": row.get("evidence_type"),
            "section_path": row.get("section_path"),
            "rects": _parse_rects(row.get("rects")),
            "rect_source": row.get("rect_source"),
        })
    return by_field, None


# -- Result formatting ----------------------------------------------------

# Lead-sentence wording per exact extraction status. The value_status is
# ALWAYS reported verbatim in the detail lines below, whatever the lead says.
_STATUS_LEAD_PHRASES = {
    "found_explicit": "explicitly stated in the document",
    "inferred": "inferred from the document - not explicitly stated; verify against the source",
    "conflicting": "recorded with conflicting values; see details below",
    "redacted": "redacted in the source document",
}


def _doc_summary(doc: dict) -> str:
    """One-line status summary for a document entry (lookup modes)."""
    field = doc["field"]
    if doc["record_status"] == "no_record":
        return (
            "no extraction record for '" + field + "' - this document has "
            "no row in extracted_fields for that field"
        )
    values = doc["values"]
    if not values:
        return "no extracted value available"
    statuses = sorted({v["value_status"] for v in values if v["value_status"]})
    extracted = [v for v in values if v["value_status"] != "not_found"]
    if not extracted:
        return (
            "not_found - no extracted value available for '" + field + "' "
            "(value_status=not_found)"
        )
    shown = [repr(v.get("value_normalized") or v.get("value_raw") or "")
             for v in extracted]
    return "extracted (" + ", ".join(statuses) + "): " + ", ".join(shown)


def _build_answer_text(lookup: dict) -> str:
    """Human-readable multi-line answer for the structured lookup result.

    Stage 2 fix (2026-10-29): deterministic natural-language rendering -
    a concise lead sentence for the unambiguous single-document value case,
    readable display names (URL-decoded; canonical names stay untouched in
    the machine fields), section and physical page in the evidence lines,
    and the exact extraction status preserved. No LLM involvement; no
    invented values, sections, or page references.
    """
    from .display import display_doc_name, physical_page

    field = lookup["target_field"]
    mode = lookup["mode"]
    lines = []

    docs = lookup["results"]

    # Lead sentence (concise answer first). Only the cases where a single
    # unambiguous natural-language statement is justified by the data get
    # one; every other shape keeps the precise structured header below -
    # the renderer never guesses.
    lead = None
    if mode == "lookup" and len(docs) == 1:
        doc = docs[0]
        values = [v for v in doc["values"] if v["value_status"] != "not_found"]
        distinct = {str(v.get("value_normalized") or v.get("value_raw") or "")
                    for v in values}
        field_label = field.replace("_", " ")
        doc_title = display_doc_name(doc["doc_name"])
        if len(distinct) == 1:
            value = next(iter(distinct))
            statuses = sorted({v["value_status"] for v in values})
            phrase = _STATUS_LEAD_PHRASES.get(statuses[0], "extracted from the document")
            lead = ("The " + field_label + " of the " + doc_title
                    + " is '" + value + "' (" + phrase + ").")
        elif doc["record_status"] == "no_record":
            lead = ("The " + doc_title + " has no extraction record for "
                    + field_label + ".")
        elif not values:
            lead = ("The " + field_label + " of the " + doc_title
                    + " has no extracted value available (value_status=not_found).")
        else:
            lead = ("The " + doc_title + " has " + str(len(distinct))
                    + " distinct recorded " + field_label + " values (details below).")
    if lead:
        lines.append(lead)
        lines.append("")

    if mode == "value_filter":
        lines.append("Structured lookup - documents where " + field + " matches "
                     + repr(lookup.get("filter_value")) + ":")
    elif mode == "status_filter":
        lines.append("Structured lookup - documents with "
                     + str(lookup.get("status_filter"))
                     + " values for " + field + ":")
    else:
        lines.append("Structured lookup - " + field + ":")

    if not docs:
        lines.append("No matching documents were found for this query.")
        return "\n".join(lines)

    for doc in docs:
        lines.append("- " + display_doc_name(doc["doc_name"]) + ": " + _doc_summary(doc))
        for ev in doc.get("evidence", [])[:MAX_EVIDENCE_PER_VALUE]:
            quote = (ev.get("quote") or "").strip()
            if not quote:
                continue
            bits = []
            if ev.get("section_path"):
                bits.append("Section " + str(ev["section_path"]))
            pg = physical_page(ev.get("page_start"))
            if pg is not None:
                try:
                    total = int(doc.get("page_count"))
                    bits.append("page " + str(pg) + " of " + str(total))
                except (TypeError, ValueError):
                    bits.append("page " + str(pg))
            where = (" - " + ", ".join(bits)) if bits else ""
            shown = quote[:160] + ("..." if len(quote) > 160 else "")
            lines.append("    Evidence" + where + ": \"" + shown + "\"")

    counts = lookup.get("status_counts", {})
    parts = [k + "=" + str(v) for k, v in sorted(counts.items()) if v]
    if parts:
        lines.append("(" + ", ".join(parts) + ")")

    needs_caveat = bool(counts.get("no_record") or counts.get("not_found")
                        or counts.get("inferred") or counts.get("conflicting"))
    if needs_caveat:
        lines.append("Note: " + MISSING_CAVEAT)
    if lookup.get("truncated"):
        lines.append("(Result truncated at the row limit - refine the query for more documents.)")
    return "\n".join(lines)


# -- Node factory ---------------------------------------------------------

def make_structured_lookup_node(executor: SqlExecutorLike, max_rows: Optional[int] = None):
    """Factory: create a LangGraph structured-lookup node bound to a SQL
    executor (fake in unit tests, StatementExecutionSqlExecutor or an
    approved equivalent in production).
    """
    row_limit = _row_limit_literal(max_rows)

    def structured_lookup_node(state: dict) -> dict:
        intent = state.get("intent")

        # Tolerant intent access: the planner provides an Intent model, but
        # the node must fail safely even if state was built outside the
        # planner (never construct SQL from an unvalidated field name).
        def _iget(name, default=None):
            if intent is None:
                return default
            if isinstance(intent, dict):
                return intent.get(name, default)
            return getattr(intent, name, default)

        target_field = _iget("target_field")
        scope = _iget("scope", "all")
        doc_names = _iget("target_doc_names", []) or []
        target_value = _iget("target_value")
        status_filter = _iget("status_filter")

        if not target_field:
            return {
                "error": "Structured lookup requires a target_field intent.",
                "trace": [make_trace(
                    "structured_lookup", action="failed", route="structured_lookup",
                    reason="missing intent or target_field",
                )],
            }
        if target_field not in ALLOWED_FIELDS:
            return {
                "error": "Field '" + str(target_field) + "' is not allowlisted.",
                "trace": [make_trace(
                    "structured_lookup", action="failed", route="structured_lookup",
                    reason="non-allowlisted target_field",
                )],
            }

        field_name = target_field  # allowlisted - safe as identifier
        mode = (
            "status_filter" if status_filter
            else "value_filter" if target_value
            else "lookup"
        )

        sql, params = _build_field_query(
            field_name, scope, doc_names, target_value, status_filter, row_limit,
        )

        try:
            rows = executor.execute(sql, params)
        except Exception as exc:  # safe failure - recorded, never propagated
            return {
                "error": ("Structured lookup query failed: " + str(exc))[:300],
                "trace": [make_trace(
                    "structured_lookup", action="failed", route="structured_lookup",
                    reason=str(exc)[:300], mode=mode, field=field_name,
                )],
            }

        # Defensive truncation (the DB LIMIT already bounds rows; this also
        # bounds executors that do not honour LIMIT).
        truncated = len(rows) > row_limit
        rows = rows[:row_limit]

        # -- Group rows into per-document entries (LEFT JOIN aware) --
        docs = {}
        order = []
        for row in rows:
            doc_id = str(row.get("doc_id") or "")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": row.get("file_name") or doc_id,
                    "is_canonical": _as_bool(row.get("is_canonical")),
                    "document_class": row.get("document_class"),
                    "file_path": row.get("file_path"),
                    "page_count": row.get("page_count"),
                    "field": field_name,
                    "record_status": "no_record",  # LEFT JOIN null until proven
                    "values": [],
                }
                order.append(doc_id)
            field_id = row.get("field_id")
            if field_id is None:
                continue  # LEFT JOIN null row - keep as no_record
            docs[doc_id]["record_status"] = "has_record"
            docs[doc_id]["values"].append({
                "value_index": row.get("value_index"),
                "value_raw": row.get("value_raw"),
                "value_normalized": row.get("value_normalized"),
                "value_status": row.get("value_status"),
                "extraction_method": row.get("extraction_method"),
                "run_id": row.get("run_id"),
                "field_id": str(field_id),
            })

        results = [docs[d] for d in order]

        # -- Evidence linkage (graceful degradation on failure) --
        field_ids = []
        for doc in results:
            for v in doc["values"]:
                if v.get("field_id"):
                    field_ids.append(v["field_id"])
        evidence_by_field, evidence_error = _query_evidence(
            executor, field_ids, row_limit,
        )
        for doc in results:
            doc_evidence = []
            for v in doc["values"]:
                v["evidence"] = evidence_by_field.get(v.get("field_id"), [])[:MAX_EVIDENCE_PER_VALUE]
                doc_evidence.extend(v["evidence"])
            doc["evidence"] = doc_evidence

        # -- Status counts (per document, derived from record_status) --
        status_counts = {"docs_examined": len(results)}
        for doc in results:
            if doc["record_status"] == "no_record":
                status_counts["no_record"] = status_counts.get("no_record", 0) + 1
                continue
            doc_statuses = {v.get("value_status") for v in doc["values"]}
            if doc_statuses == {"not_found"} or not doc_statuses:
                status_counts["not_found"] = status_counts.get("not_found", 0) + 1
                continue
            for st in doc_statuses:
                if st:
                    status_counts[st] = status_counts.get(st, 0) + 1

        # -- Sources (for source refs; capped at MAX_SOURCE_DOCS docs) --
        sources = []
        seen_docs = set()
        for doc in results:
            if len(seen_docs) >= MAX_SOURCE_DOCS:
                break
            if doc["doc_id"] in seen_docs:
                continue
            ev = doc.get("evidence") or []
            if not ev:
                continue
            seen_docs.add(doc["doc_id"])
            first = ev[0]
            sources.append({
                "doc_id": doc["doc_id"],
                "doc_name": doc["doc_name"],
                "section_path": first.get("section_path"),
                "chunk_id": first.get("chunk_id"),
                "page_start": first.get("page_start"),
                "page_end": first.get("page_end"),
                "excerpt": (first.get("quote") or "")[:500],
                "rects": _parse_rects(first.get("rects")),
                "rect_source": first.get("rect_source"),
                "file_path": doc.get("file_path"),
                "page_count": doc.get("page_count"),
            })

        lookup_result = {
            "stub": False,
            "capability": "structured_lookup",
            "target_field": field_name,
            "mode": mode,
            "filter_value": target_value if mode == "value_filter" else None,
            "status_filter": status_filter if mode == "status_filter" else None,
            "target_doc_names": list(doc_names or []),
            "results": results,
            "status_counts": status_counts,
            "sources": sources,
            "truncated": truncated,
        }
        if evidence_error:
            lookup_result["evidence_error"] = evidence_error
        lookup_result["answer_text"] = _build_answer_text(lookup_result)

        trace_details = {
            "mode": mode,
            "field": field_name,
            "rows_returned": len(rows),
            "docs_examined": len(results),
            "truncated": truncated,
            "param_keys": sorted(params.keys()),
        }
        if evidence_error:
            trace_details["evidence_error"] = evidence_error

        return {
            "lookup_result": lookup_result,
            "trace": [make_trace(
                "structured_lookup",
                action="queried",
                route="structured_lookup",
                reason=(str(len(results)) + " document(s) for field '"
                        + field_name + "' (" + mode + ")"),
                **trace_details,
            )],
        }

    return structured_lookup_node
