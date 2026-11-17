"""Corpus Overview capability node (Capability A; spec Section 1.2).

Answers metadata questions about the document collection itself -- counts,
classifications, distributions, and field-level extraction coverage --
using structured SQL over workspace.legal_assistant_v2.documents and
workspace.legal_assistant_v2.extracted_fields.  No retrieval, no
embeddings, no LLM generation of the aggregate values.

Supported question types:
  * corpus_summary -- total registered documents, canonical document count,
    document-class distribution (e.g. 'How many documents are in the corpus?',
    'What types of documents do you have?')
  * field_coverage -- for a specified allowlisted field, count canonical
    documents with/without extraction records and the distribution of
    value_status (e.g. 'How many contracts have a termination-for-
    convenience clause?', 'What is the extraction status for governing_law?')

Safety model (spec Section 1.2 + Milestone 3 task constraints):
  - Field names are allowlisted (ALLOWED_FIELDS in state.py) and are the
    ONLY identifiers interpolated into SQL text, after a membership check.
  - No user-derived filter value is interpolated into SQL text -- this node
    performs only aggregate counts and distributions, not filtered
    document listings (those are served by structured_lookup).
  - Row limits are internal integer constants (never user input).
  - Table names are hardcoded module constants.
  - The LLM never generates SQL; the query planner produces a validated
    Intent, and this node translates it into one of a fixed set of
    parameterized aggregate query templates.

Missing-value semantics (spec Section 1.5 -- normative, same as structured
lookup):
  1. found_explicit / inferred / conflicting / redacted rows are extracted
     values with their status reported verbatim.
  2. not_found rows mean the extractor found no value -- reported as a
     distinct count.
  3. A LEFT JOIN null row means NO extraction record exists for that
     document+field (document not covered by the extraction run).
  4. An executor exception is a query/database error, recorded in state.

A missing or uncertain extraction is NEVER presented as proof that the
underlying clause or obligation is absent from the contract -- clause
absence requires a separate evidence-based assessment.

Run multiplicity (verified 2026-10-28): extracted_fields contains 4
extraction runs.  Field-coverage queries deduplicate with ROW_NUMBER()
OVER (PARTITION BY doc_id, field_name, value_index ORDER BY created_at
DESC, field_id DESC) so each document+field+value_index appears once,
from the latest run -- the same approach used by structured_lookup.

Dependency injection follows the structured_lookup pattern: the node
factory receives a SqlExecutorLike; unit tests pass a fake, and production
passes the same StatementExecutionSqlExecutor already live-verified
against warehouse c99228c48630a173.
"""

from __future__ import annotations

from typing import Any, Optional

from .state import ALLOWED_FIELDS, MAX_DOC_LISTING
from .traces import make_trace
from .structured_lookup import SqlExecutorLike, VALUE_STATUSES


# -- Constants (verified against information_schema 2026-10-28) ----------

DOCUMENTS_TABLE = "workspace.legal_assistant_v2.documents"
EXTRACTED_FIELDS_TABLE = "workspace.legal_assistant_v2.extracted_fields"

# Maximum category rows returned in a distribution (bounded; internal
# constant, not user input).
MAX_CATEGORY_ROWS: int = 50

MISSING_CAVEAT = (
    "A missing or uncertain extraction is not proof that the corresponding "
    "clause or obligation is absent from the contract."
)


# -- Query templates (fixed; only allowlisted field name interpolated) ---

_DOC_COUNT_QUERY = """
SELECT
  COUNT(*) AS total_documents,
  COUNT(CASE WHEN is_canonical = true THEN 1 END) AS canonical_documents
FROM {documents_table}
""".strip()

_DOC_CLASS_QUERY = """
SELECT
  COALESCE(NULLIF(document_class, ''), 'unclassified') AS document_class,
  COUNT(*) AS doc_count
FROM {documents_table}
WHERE is_canonical = true
GROUP BY document_class
ORDER BY doc_count DESC, document_class
LIMIT {row_limit}
""".strip()

_FIELD_COVERAGE_QUERY = """
WITH latest AS (
  SELECT doc_id, field_id, value_index, value_status,
    ROW_NUMBER() OVER (
      PARTITION BY doc_id, field_name, value_index
      ORDER BY created_at DESC, field_id DESC
    ) AS rn
  FROM {fields_table}
  WHERE field_name = '{field_name}'
),
doc_field AS (
  SELECT
    d.doc_id,
    MAX(CASE WHEN l.field_id IS NOT NULL THEN 1 ELSE 0 END) AS has_record,
    MAX(CASE WHEN l.value_status = 'found_explicit' THEN 1 ELSE 0 END) AS has_found_explicit,
    MAX(CASE WHEN l.value_status = 'not_found' THEN 1 ELSE 0 END) AS has_not_found,
    MAX(CASE WHEN l.value_status = 'inferred' THEN 1 ELSE 0 END) AS has_inferred,
    MAX(CASE WHEN l.value_status = 'conflicting' THEN 1 ELSE 0 END) AS has_conflicting,
    MAX(CASE WHEN l.value_status = 'redacted' THEN 1 ELSE 0 END) AS has_redacted
  FROM {documents_table} d
  LEFT JOIN latest l
    ON l.doc_id = d.doc_id AND l.rn = 1
  WHERE d.is_canonical = true
  GROUP BY d.doc_id
)
SELECT
  COUNT(*) AS total_canonical_docs,
  SUM(has_record) AS docs_with_record,
  SUM(1 - has_record) AS docs_no_record,
  SUM(has_found_explicit) AS docs_found_explicit,
  SUM(has_not_found) AS docs_not_found,
  SUM(has_inferred) AS docs_inferred,
  SUM(has_conflicting) AS docs_conflicting,
  SUM(has_redacted) AS docs_redacted
FROM doc_field
""".strip()


# -- Helpers -------------------------------------------------------------

def _to_int(value: Any, default: int = 0) -> int:
    """Tolerantly parse an integer from executor output (may be str/None)."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_doc_count_sql() -> str:
    return _DOC_COUNT_QUERY.format(documents_table=DOCUMENTS_TABLE)


def _build_doc_class_sql(row_limit: int) -> str:
    return _DOC_CLASS_QUERY.format(
        documents_table=DOCUMENTS_TABLE,
        row_limit=row_limit,
    )


def _build_field_coverage_sql(field_name: str) -> str:
    """Build the field-coverage query for the given allowlisted field.

    Only field_name (already validated against ALLOWED_FIELDS) is
    interpolated into the SQL text.  No user-derived values are present.
    """
    return _FIELD_COVERAGE_QUERY.format(
        fields_table=EXTRACTED_FIELDS_TABLE,
        documents_table=DOCUMENTS_TABLE,
        field_name=field_name,  # allowlisted identifier (validated by caller)
    )


# -- Result formatting ----------------------------------------------------

def _build_answer_text(result: dict) -> str:
    """Human-readable multi-line answer for the corpus overview result."""
    mode = result["mode"]
    lines = []

    if mode == "corpus_summary":
        counts = result.get("doc_counts", {})
        total = _to_int(counts.get("total_documents"))
        canonical = _to_int(counts.get("canonical_documents"))
        lines.append("Corpus overview:")
        lines.append("- Total registered documents: " + str(total))
        lines.append("- Canonical documents: " + str(canonical))

        distribution = result.get("doc_class_distribution", [])
        if distribution:
            lines.append("- Document class distribution (canonical):")
            for entry in distribution:
                cls = entry.get("document_class") or "unclassified"
                cnt = _to_int(entry.get("doc_count"))
                lines.append("  " + str(cls) + ": " + str(cnt))
        else:
            lines.append("- Document class distribution: no canonical documents found.")

    elif mode == "field_coverage":
        field = result.get("target_field", "(unknown)")
        coverage = result.get("coverage", {})
        total_canonical = _to_int(coverage.get("total_canonical_docs"))
        with_record = _to_int(coverage.get("docs_with_record"))
        no_record = _to_int(coverage.get("docs_no_record"))
        lines.append("Corpus overview -- field coverage for " + field + ":")
        lines.append("- Canonical documents: " + str(total_canonical))
        lines.append("- Documents with extraction record: " + str(with_record))
        lines.append("- Documents with no extraction record: " + str(no_record))
        lines.append("- Status distribution (documents with at least one value in each status):")
        for st in ("found_explicit", "not_found", "inferred", "conflicting", "redacted"):
            key = "docs_" + st
            cnt = _to_int(coverage.get(key))
            if cnt or st in ("found_explicit", "not_found"):
                lines.append("  " + st + ": " + str(cnt))

        if no_record > 0 or _to_int(coverage.get("docs_not_found")) > 0 or \
           _to_int(coverage.get("docs_inferred")) > 0 or \
           _to_int(coverage.get("docs_conflicting")) > 0:
            lines.append("Note: " + MISSING_CAVEAT)

    if result.get("limitations"):
        for lim in result["limitations"]:
            lines.append("Limitation: " + lim)

    return "\n".join(lines)


# -- Node factory ---------------------------------------------------------

def make_corpus_overview_node(executor: SqlExecutorLike):
    """Factory: create a LangGraph corpus-overview node bound to a SQL
    executor (fake in unit tests, StatementExecutionSqlExecutor or an
    approved equivalent in production).
    """

    def corpus_overview_node(state: dict) -> dict:
        intent = state.get("intent")

        # Tolerant intent access (same pattern as structured_lookup)
        def _iget(name, default=None):
            if intent is None:
                return default
            if isinstance(intent, dict):
                return intent.get(name, default)
            return getattr(intent, name, default)

        target_field = _iget("target_field")

        # Determine mode: field coverage if an allowlisted field is
        # specified; otherwise corpus-level summary.
        if target_field and target_field in ALLOWED_FIELDS:
            mode = "field_coverage"
        else:
            mode = "corpus_summary"
            target_field = None  # ignore non-allowlisted fields

        # -- Execute queries --
        try:
            if mode == "field_coverage":
                sql = _build_field_coverage_sql(target_field)
                rows = executor.execute(sql, {})
                if not rows:
                    coverage = {"total_canonical_docs": 0}
                else:
                    row = rows[0]
                    coverage = {
                        "total_canonical_docs": _to_int(row.get("total_canonical_docs")),
                        "docs_with_record": _to_int(row.get("docs_with_record")),
                        "docs_no_record": _to_int(row.get("docs_no_record")),
                        "docs_found_explicit": _to_int(row.get("docs_found_explicit")),
                        "docs_not_found": _to_int(row.get("docs_not_found")),
                        "docs_inferred": _to_int(row.get("docs_inferred")),
                        "docs_conflicting": _to_int(row.get("docs_conflicting")),
                        "docs_redacted": _to_int(row.get("docs_redacted")),
                    }

                result = {
                    "stub": False,
                    "capability": "corpus_overview",
                    "mode": mode,
                    "target_field": target_field,
                    "coverage": coverage,
                    "limitations": [],
                }
                result["answer_text"] = _build_answer_text(result)

                trace_details = {
                    "mode": mode,
                    "field": target_field,
                    "total_canonical_docs": coverage.get("total_canonical_docs", 0),
                    "docs_with_record": coverage.get("docs_with_record", 0),
                    "docs_no_record": coverage.get("docs_no_record", 0),
                }

            else:  # corpus_summary
                # Query 1: document counts
                count_sql = _build_doc_count_sql()
                count_rows = executor.execute(count_sql, {})
                if count_rows:
                    cr = count_rows[0]
                    doc_counts = {
                        "total_documents": _to_int(cr.get("total_documents")),
                        "canonical_documents": _to_int(cr.get("canonical_documents")),
                    }
                else:
                    doc_counts = {"total_documents": 0, "canonical_documents": 0}

                # Query 2: document-class distribution (canonical only)
                class_sql = _build_doc_class_sql(MAX_CATEGORY_ROWS)
                class_rows = executor.execute(class_sql, {})
                distribution = []
                for row in class_rows or []:
                    distribution.append({
                        "document_class": row.get("document_class"),
                        "doc_count": _to_int(row.get("doc_count")),
                    })

                result = {
                    "stub": False,
                    "capability": "corpus_overview",
                    "mode": mode,
                    "doc_counts": doc_counts,
                    "doc_class_distribution": distribution,
                    "limitations": [],
                }
                result["answer_text"] = _build_answer_text(result)

                trace_details = {
                    "mode": mode,
                    "total_documents": doc_counts.get("total_documents", 0),
                    "canonical_documents": doc_counts.get("canonical_documents", 0),
                    "class_categories": len(distribution),
                }

        except Exception as exc:
            # Safe failure -- recorded, never propagated
            return {
                "error": ("Corpus overview query failed: " + str(exc))[:300],
                "trace": [make_trace(
                    "corpus_overview", action="failed", route="corpus_overview",
                    reason=str(exc)[:300], mode=mode,
                    field=target_field or "",
                )],
            }

        return {
            "metadata_result": result,
            "trace": [make_trace(
                "corpus_overview",
                action="queried",
                route="corpus_overview",
                reason=("corpus overview (" + mode + ")"
                        + (" for " + target_field if target_field else "")),
                **trace_details,
            )],
        }

    return corpus_overview_node