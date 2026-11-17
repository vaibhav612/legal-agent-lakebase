"""Contract Comparison capability node (Capability E; spec Section 1.6).

Retrieves and aligns comparable provisions across two or more specified
agreements, producing a structured comparison with evidence.

Key design decisions (spec Section 1.6):
  1. Require at least two unambiguously resolved agreements.
  2. Enforce MAX_COMPARISON_DOCS (10) - excess documents produce a
     scope-limitation, never a silent truncation.
  3. For every agreement, independently retrieve evidence for the requested
     aspect using the document-filtered retrieval interface (doc_ids)
     from clause discovery.
  4. When the aspect maps to one of the 10 allowlisted extracted fields,
     also query extracted_fields for the structured value (with
     latest-run dedup, same pattern as structured_lookup).
  5. Evidence from one contract is NEVER attributed to another.
  6. Distinguish:
     a) provision supported by retrieved evidence
     b) no qualifying passage retrieved after a successful search
     c) document that could not be searched successfully
     d) extracted field marked missing or uncertain
     e) comparison that cannot be completed from available evidence
  7. A retrieval miss or missing extraction is NOT proof that a clause is
     absent (spec Section 1.6 missing-evidence handling).
  8. Preserve stable document IDs throughout retrieval, synthesis, and
     citation construction.
  9. The underlying evidence is preserved even if the source-display is
     capped at MAX_SOURCE_DOCS (3) distinct documents.

Dependency injection follows the clause_discovery pattern:
  - retriever: anything exposing search(query, num_results, doc_ids).
  - sql_executor: SqlExecutorLike for canonical-doc enumeration,
    document-name resolution, and optional structured-field lookup.
  - answer_generator: optional callable matching
    LakebaseRetriever.generate_response(query, context_chunks) -> str.
    When provided, generates a grounded comparison narrative.
    When None, a structured text comparison is built from the per-document
    findings.

Node contract (spec Section 2.5-E):
  Input:  user_query, intent (comparison_aspect, target_doc_names, scope)
  Output: evidence (list[EvidenceItem]) - preserved for evidence_validation
          -> final_answer.
          comparison_result - per-doc findings, coverage, answer_text.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Protocol

from .state import (
    ALLOWED_FIELDS,
    EvidenceItem,
    Intent,
    MAX_COMPARISON_DOCS,
    MAX_SOURCE_DOCS,
    SourceRef,
)
from .traces import make_trace
from .structured_lookup import SqlExecutorLike, DOCUMENTS_TABLE, _query_evidence
from .display import display_doc_name
from .clause_discovery import (
    _CANONICAL_DOCS_SQL,
    _resolve_doc_names,
    _to_str,
    search_result_to_evidence,
    MAX_NAME_PATTERN_CHARS,
)


# -- Constants ---------------------------------------------------------------

EXTRACTED_FIELDS_TABLE = "workspace.legal_assistant_v2.extracted_fields"

DEFAULT_PER_DOC_RESULTS: int = 5
MAX_TOTAL_EVIDENCE: int = 200

# Caveats (spec Section 1.6)
NO_MATCH_CAVEAT = (
    "No {aspect} provision was found in {doc_name}. "
    "This may indicate its absence, or it may use different terminology."
)

MISSING_EXTRACTION_CAVEAT = (
    "A missing or uncertain extraction is not proof that the corresponding "
    "clause or obligation is absent from the contract."
)

CANDIDATE_CAVEAT = (
    "Retrieved passages are candidate matches, not confirmed findings. "
    "Relevance calibration is pending (Milestone 4)."
)

SOURCE_DISPLAY_CAVEAT = (
    "Source references are capped at {max_docs} distinct documents. "
    "Evidence from additional compared agreements is preserved but not "
    "shown in the source display."
)


class ComparisonRetrieverLike(Protocol):
    """Structural type of the injected retriever."""
    def search(self, query: str, num_results: int = 5,
               doc_ids: Optional[list[str]] = None) -> list[dict]: ...


_ASPECT_FIELD_MAP: list[tuple[str, str]] = [
    ("governing law", "governing_law"),
    ("agreement date", "agreement_date"),
    ("effective date", "effective_date"),
    ("expiration", "expiration_or_term"),
    ("termination for convenience", "termination_for_convenience"),
    ("payment terms", "payment_terms"),
    ("renewal term", "renewal_term"),
    ("notice to terminate", "notice_to_terminate_renewal"),
    ("contract title", "contract_title"),
    ("parties", "parties"),
]


def _map_aspect_to_field(aspect: str) -> Optional[str]:
    if not aspect:
        return None
    aspect_lower = aspect.lower().strip()
    for phrase, field_name in _ASPECT_FIELD_MAP:
        if phrase in aspect_lower:
            return field_name
    if aspect_lower in ALLOWED_FIELDS:
        return aspect_lower
    return None


_FIELD_FOR_DOCS_SQL_TEMPLATE = """
WITH latest AS (
  SELECT doc_id, field_id, value_index, value_raw, value_normalized,
         value_status, extraction_method, run_id,
         ROW_NUMBER() OVER (
           PARTITION BY doc_id, field_name, value_index
           ORDER BY created_at DESC, field_id DESC
         ) AS rn
  FROM {fields_table}
  WHERE field_name = '{field_name}'
    AND doc_id IN ({doc_id_placeholders})
)
SELECT doc_id, field_id, value_index, value_raw, value_normalized,
       value_status, extraction_method, run_id
FROM latest
WHERE rn = 1
ORDER BY doc_id, value_index
LIMIT 200
""".strip()


def _build_field_query(field_name: str, doc_ids: list) -> tuple:
    placeholders = ", ".join(f":doc_{i}" for i in range(len(doc_ids)))
    params = {f"doc_{i}": did for i, did in enumerate(doc_ids)}
    sql = _FIELD_FOR_DOCS_SQL_TEMPLATE.format(
        fields_table=EXTRACTED_FIELDS_TABLE,
        field_name=field_name,
        doc_id_placeholders=placeholders,
    )
    return sql, params


def _structured_value_summary(values: list) -> str:
    if not values:
        return "(no extraction record)"
    parts = []
    for v in values:
        raw = v.get("value_raw", "")
        status = v.get("value_status", "")
        if raw:
            parts.append(f"{raw} ({status})")
        else:
            parts.append(f"({status})")
    return "; ".join(parts) if parts else "(empty)"


def _build_answer_text(aspect, per_doc_findings, coverage, limitations,
                       has_structured):
    lines = []
    lines.append(f'Comparison -- aspect: "{aspect}"')
    lines.append(f"- Documents compared: {coverage.get('documents_compared', 0)}")
    lines.append(f"- Documents with evidence: {coverage.get('documents_with_evidence', 0)}")
    lines.append(f"- Documents with no evidence: {coverage.get('documents_no_evidence', 0)}")
    lines.append(f"- Documents failed: {coverage.get('documents_failed', 0)}")

    if per_doc_findings:
        lines.append("")
        lines.append("Findings by agreement:")
        for f in per_doc_findings:
            doc_name = f.get("doc_name", "?")
            status = f.get("status", "unknown")
            lines.append("")
            lines.append(f"  {doc_name} ({status}):")
            if has_structured:
                sv = f.get("structured_values")
                if sv:
                    lines.append(f"    Extracted value: {_structured_value_summary(sv)}")
                else:
                    lines.append("    Extracted value: (no extraction record)")
            evidence_items = f.get("evidence_items", [])
            if evidence_items:
                lines.append(f"    Evidence ({len(evidence_items)} passage(s)):")
                for ev in evidence_items[:5]:
                    section = ev.get("section_path") or "(no section)"
                    pages = ""
                    if ev.get("page_start") is not None:
                        ps = ev.get("page_start")
                        pe = ev.get("page_end")
                        pages = f" p.{ps}" + (f"-{pe}" if pe and pe != ps else "")
                    text = (ev.get("supporting_text") or "")[:200]
                    lines.append(f"      [{section}{pages}] {text}...")
            else:
                lines.append("    Evidence: no qualifying passage retrieved")
            if f.get("error"):
                lines.append(f"    Error: {f['error']}")

    if limitations:
        lines.append("")
        for lim in limitations:
            lines.append("Limitation: " + lim)

    no_evidence = coverage.get("documents_no_evidence", 0)
    if no_evidence > 0:
        lines.append("")
        lines.append(
            "Note: A retrieval miss is not proof that the provision does "
            "not exist. The search may not have surfaced all relevant passages."
        )
    if has_structured:
        lines.append("Note: " + MISSING_EXTRACTION_CAVEAT)
    total_evidence = sum(len(f.get("evidence_items", [])) for f in per_doc_findings)
    if total_evidence > 0:
        lines.append("Note: " + CANDIDATE_CAVEAT)

    docs_with_ev = coverage.get("documents_with_evidence", 0)
    if docs_with_ev > MAX_SOURCE_DOCS:
        lines.append("")
        lines.append(SOURCE_DISPLAY_CAVEAT.format(max_docs=MAX_SOURCE_DOCS))

    return "\n".join(lines)


def _build_comparison_prompt(aspect, doc_names):
    return (
        f"Compare {aspect} across the following agreements: {', '.join(doc_names)}.\n"
        "Compare only claims supported by the provided evidence. "
        "Highlight material similarities and differences. "
        "Keep every finding associated with its source agreement. "
        "State where evidence is missing or not comparable. "
        "Do not fabricate terms or treat an unobserved clause as confirmed absence."
    )


# ── Fix 010 (2026-10-29, hosted smoke findings): comparison-candidate ─────
# discovery + accepted-suggestion execution.  'What documents can be
# compared?' must be answered from the CANONICAL DOCUMENT INVENTORY and
# verified metadata — never from semantic-retrieval chunks (which
# previously produced generic examples with unrelated sources).  When the
# user accepts the recommendation ('sure compare them then'), the planner
# resolves the exact recommended documents (classifier.resolve_pending_action)
# and this node executes a default-dimension matrix comparison from the
# per-document extracted fields, with per-cell extraction evidence.

_CANDIDATE_DOCS_SQL = (
    "SELECT doc_id, file_name, document_class, file_path, page_count\n"
    "FROM " + DOCUMENTS_TABLE + "\n"
    "WHERE is_canonical = true\n"
    "ORDER BY file_name"
).strip()

# Default comparison dimensions for an accepted 'compare them' request:
# the allowlisted extracted fields in a sensible contract order. Only
# these dimensions are compared; clause-level dimensions (indemnification,
# liability, insurance, confidentiality, IP) are NOT extracted fields and
# are reported as evidence gaps with a pointer to targeted clause
# discovery — never silently forced into the matrix.
DIMENSION_FIELDS: list = [
    ("parties", "Parties and scope"),
    ("contract_title", "Contract title"),
    ("agreement_date", "Agreement date"),
    ("effective_date", "Effective date"),
    ("expiration_or_term", "Term and expiration"),
    ("renewal_term", "Renewal term"),
    ("termination_for_convenience", "Termination for convenience"),
    ("notice_to_terminate_renewal", "Notice to terminate/renew"),
    ("payment_terms", "Fees, payment and financial obligations"),
    ("governing_law", "Governing law and dispute resolution"),
]

MAX_RECOMMENDED_PAIRS: int = 3
MAX_COMPARISON_SOURCE_REFS: int = 12


def _agreement_type(file_name: str) -> str:
    """Deterministic agreement type derived from the stored file_name ONLY.

    Parses the same title segment display_doc_name uses (company — title),
    strips trailing numbering ('Distributor Agreement2'), and normalizes
    a trailing ' contract' to ' agreement' so legacy all-caps filings
    ('JOINT VENTURE CONTRACT') group with their '...AGREEMENT' siblings.
    Never invents a type that is not present in the stored name.
    """
    disp = display_doc_name(file_name)
    title = disp.split("—", 1)[1].strip() if "—" in disp else disp
    title = re.sub(r"\d+$", "", title).strip()
    low = title.lower().strip()
    if low.endswith(" contract"):
        low = low[: -len(" contract")].strip() + " agreement"
    return low or "unclassified"


def _run_candidate_discovery(sql_executor) -> dict:
    """Answer 'what documents can be compared?' from the canonical inventory.

    Returns an update whose comparison_result.mode == 'candidates' — a
    metadata-backed answer with NO chunk evidence and NO retrieval-based
    source refs (unrelated chunks must never be presented as evidence for
    a recommendation).  Sets pending_action so the next turn can resolve
    an acceptance ('sure compare them then') to the exact recommended pair.
    """
    try:
        rows = sql_executor.execute(_CANDIDATE_DOCS_SQL, {}) or []
    except Exception as exc:
        return {
            "error": ("Comparison candidate enumeration failed: " + str(exc))[:300],
            "evidence": [],
            "trace": [make_trace(
                "comparison", action="failed", route="comparison",
                reason=str(exc)[:300], step="enumerate_candidate_documents",
            )],
        }
    if not rows:
        msg = "No canonical documents were found in the corpus."
        return {
            "evidence": [],
            "comparison_result": {
                "stub": False, "capability": "comparison", "mode": "candidates",
                "comparison_aspect": "candidates", "answer_text": msg,
                "coverage": {"documents_compared": 0}, "limitations": [msg],
            },
            "trace": [make_trace(
                "comparison", action="completed", route="comparison",
                reason="no canonical documents", aspect="candidates",
            )],
        }

    # Group canonical documents by agreement type (from stored metadata)
    groups = {}
    for d in rows:
        groups.setdefault(_agreement_type(_to_str(d.get("file_name"))), []).append(d)
    comparable = [(t, docs) for t, docs in sorted(groups.items()) if len(docs) >= 2]
    singletons = [(t, docs[0]) for t, docs in sorted(groups.items()) if len(docs) == 1]

    # Recommended shortlist: first pair (stable file_name order) of the
    # first MAX_RECOMMENDED_PAIRS same-type groups.
    pairs = []
    for t, docs in comparable[:MAX_RECOMMENDED_PAIRS]:
        pair_docs = [
            {
                "doc_id": _to_str(d.get("doc_id")),
                "file_name": _to_str(d.get("file_name")),
            }
            for d in docs[:2]
        ]
        pairs.append({
            "pair_id": len(pairs),
            "agreement_type": t,
            "docs": pair_docs,
            "reason": (
                "same agreement type — " + t
                + "; extracted contract fields are available for a "
                + "substantive clause-by-clause comparison"
            ),
        })

    # ── Answer text (metadata-grounded, no LLM, no fabricated names) ──
    lines = []
    lines.append(
        "Based on the canonical corpus inventory (" + str(len(rows)) + " documents), "
        "here is what can be compared."
    )
    lines.append("")
    if pairs:
        lines.append("**Recommended comparable pairs**")
        for p in pairs:
            names = " and ".join(
                display_doc_name(d["file_name"]) for d in p["docs"]
            )
            lines.append(
                str(p["pair_id"] + 1) + ". " + names + " — " + p["reason"] + "."
            )
        lines.append("")
    else:
        lines.append(
            "No two canonical documents share the same agreement type, so "
            "no direct clause-by-clause pair is available."
        )
        lines.append("")
    if len(comparable) > len(pairs):
        lines.append("**Other same-type groups** (also directly comparable):")
        for t, docs in comparable[len(pairs):]:
            names = ", ".join(display_doc_name(_to_str(d.get("file_name"))) for d in docs)
            lines.append("- " + t.title() + ": " + names)
        lines.append("")
    if singletons:
        lines.append(
            "**Structural comparisons** (different agreement types — a "
            "structural comparison of dates, parties and governing law is "
            "possible, but a substantive clause-by-clause comparison is "
            "not):"
        )
        shown = singletons[:8]
        lines.append(
            ", ".join(display_doc_name(_to_str(d.get("file_name"))) for _, d in shown)
            + (" and others" if len(singletons) > len(shown) else "")
        )
        lines.append("")
    lines.append("**Feasible comparison dimensions** (extracted for the recommended pairs):")
    lines.append(", ".join(label for _, label in DIMENSION_FIELDS) + ".")
    lines.append(
        "Clause-level dimensions (indemnification, liability limitations, "
        "insurance, confidentiality, intellectual property) are not "
        "extracted fields; ask for a targeted clause search if you need "
        "those."
    )
    lines.append("")
    if pairs:
        lines.append(
            "Say **'compare them'** (or 'compare the first/second pair') and "
            "I will run the comparison on the default dimensions above."
        )
    lines.append(
        "_Candidates above come from the verified canonical document "
        "inventory and its metadata — not from retrieval chunks._"
    )
    answer_text = "\n".join(lines)

    pending_action = {
        "type": "comparison_candidates",
        "pairs": pairs,
        "recommended_pair_id": 0,
        "dimensions": [f for f, _ in DIMENSION_FIELDS],
    }
    result = {
        "stub": False,
        "capability": "comparison",
        "mode": "candidates",
        "comparison_aspect": "candidates",
        "coverage": {"canonical_documents": len(rows), "comparable_types": len(comparable)},
        "candidates": [
            {"doc_id": _to_str(d.get("doc_id")), "file_name": _to_str(d.get("file_name"))}
            for d in rows
        ],
        "pending_action": pending_action,
        "limitations": [],
        "answer_text": answer_text,
    }
    return {
        "evidence": [],
        "comparison_result": result,
        "pending_action": pending_action if pairs else None,
        "trace": [make_trace(
            "comparison", action="completed", route="comparison",
            reason=(
                str(len(rows)) + " canonical docs, " + str(len(comparable))
                + " comparable types, " + str(len(pairs)) + " pairs recommended"
            ),
            aspect="candidates", num_recommended_pairs=len(pairs),
        )],
    }


def _value_summary(values) -> tuple:
    """Render one document's extracted values for a matrix cell.

    Returns (display_text, status_summary). Distinguishes found_explicit /
    not_found / inferred / redacted / conflicting verbatim; a missing
    extraction record is reported as such — never as proof of absence.
    """
    if not values:
        return "(no extraction record)", "no_record"
    parts = []
    statuses = set()
    for v in values:
        raw = _to_str(v.get("value_raw")).strip()
        status = _to_str(v.get("value_status"))
        statuses.add(status or "unknown")
        shown = raw if raw else "(no value text)"
        if status:
            shown = shown + " (" + status + ")"
        parts.append(shown)
    return " | ".join(parts), "+".join(sorted(statuses))


def _run_matrix_comparison(sql_executor, all_canonical, doc_names, user_query) -> dict:
    """Execute an accepted comparison on the default dimensions.

    Deterministic, metadata/evidence-grounded (no LLM): per dimension, the
    latest extracted value for EACH document is queried separately, the
    linked extraction evidence (quote, section, page, rects) is fetched
    per field, and the answer is a comparative matrix with per-cell
    citations.  Evidence is doc-bound by field_id — passages from one
    agreement can never be attributed to the other.
    """
    resolved, ambiguous, unresolved = _resolve_doc_names(
        sql_executor, all_canonical, list(doc_names or []),
    )
    unresolved_names = [u["requested_name"] for u in unresolved]
    if len(resolved) < 2:
        msg_parts = []
        if unresolved_names:
            msg_parts.append(
                "Could not find documents matching: " + ", ".join(unresolved_names) + "."
            )
        for a in ambiguous:
            msg_parts.append(
                "Name '" + a["requested_name"] + "' is ambiguous — matches: "
                + ", ".join(a["matching_files"]) + "."
            )
        if not msg_parts:
            msg_parts.append("Could not resolve at least two documents for the comparison.")
        msg = " ".join(msg_parts)
        return {
            "evidence": [],
            "pending_action": None,
            "comparison_result": {
                "stub": False, "capability": "comparison", "mode": "matrix",
                "comparison_aspect": "default", "answer_text": msg,
                "coverage": {"documents_compared": 0}, "limitations": [msg],
            },
            "trace": [make_trace(
                "comparison", action="clarification", route="comparison",
                reason=msg[:200], aspect="default",
                unresolved=unresolved_names,
            )],
        }
    if len(resolved) > MAX_COMPARISON_DOCS:
        resolved = resolved[:MAX_COMPARISON_DOCS]

    doc_ids = [_to_str(d.get("doc_id")) for d in resolved]
    doc_names_disp = [
        display_doc_name(_to_str(d.get("file_name")) or did)
        for did, d in zip(doc_ids, resolved)
    ]
    # Enrich metadata (file_path/page_count) from the full canonical rows —
    # _resolve_doc_names returns only {doc_id, file_name}.
    meta_by_id = {
        _to_str(d.get("doc_id")): d for d in (all_canonical or [])
    }

    # 1. Per-dimension extracted values for EACH document separately
    per_dim = []
    field_ids_all = []
    dim_failed = []
    for field, label in DIMENSION_FIELDS:
        by_doc = {}
        try:
            sql, params = _build_field_query(field, doc_ids)
            rows = sql_executor.execute(sql, params) or []
        except Exception:
            dim_failed.append(label)
            rows = []
        for r in rows:
            did = _to_str(r.get("doc_id"))
            by_doc.setdefault(did, []).append({
                "field_id": _to_str(r.get("field_id")),
                "value_raw": _to_str(r.get("value_raw")),
                "value_normalized": _to_str(r.get("value_normalized")),
                "value_status": _to_str(r.get("value_status")),
                "value_index": r.get("value_index"),
            })
        for vals in by_doc.values():
            field_ids_all.extend(v["field_id"] for v in vals if v.get("field_id"))
        per_dim.append({"field": field, "label": label, "by_doc": by_doc})

    # 2. Linked extraction evidence per field (doc-bound by field_id)
    ev_by_field, ev_err = _query_evidence(sql_executor, field_ids_all, 200)

    # 3. Build the matrix with per-cell citations + claim-mapped sources
    sources = []
    capped_sources = False
    matrix_rows = []
    differences = []
    gaps = []
    both_explicit = 0

    for dim in per_dim:
        by_doc = dim["by_doc"]
        cells = {}
        for did, dname in zip(doc_ids, doc_names_disp):
            text, status = _value_summary(by_doc.get(did))
            marker = ""
            # citation: first value of this doc/dimension with evidence
            for v in by_doc.get(did) or []:
                fid = v.get("field_id")
                ev_rows = ev_by_field.get(fid) or []
                if not ev_rows:
                    continue
                e = ev_rows[0]
                if len(sources) >= MAX_COMPARISON_SOURCE_REFS:
                    capped_sources = True
                    break
                meta = meta_by_id.get(did) or {}
                sources.append({
                    "doc_id": did,
                    "doc_name": _to_str(meta.get("file_name")) or did,
                    "section_path": e.get("section_path"),
                    "chunk_id": e.get("chunk_id"),
                    "page_start": e.get("page_start"),
                    "page_end": e.get("page_end"),
                    "excerpt": (_to_str(e.get("quote")) or "")[:500],
                    "rects": e.get("rects") or [],
                    "rect_source": e.get("rect_source"),
                    "file_path": meta.get("file_path"),
                    "page_count": meta.get("page_count"),
                })
                marker = " [" + str(len(sources)) + "]"
                break
            cells[did] = {"text": text, "status": status, "marker": marker}

        # Difference detection: single explicit value on both sides that differ
        vals = []
        for did in doc_ids:
            vlist = by_doc.get(did) or []
            if len(vlist) == 1 and _to_str(vlist[0].get("value_status")) == "found_explicit":
                vals.append(_to_str(vlist[0].get("value_raw")).strip().lower())
            else:
                vals.append(None)
        if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
            if vals[0] != vals[1]:
                differences.append(
                    "- " + dim["label"] + ": "
                    + doc_names_disp[0] + " = " + cells[doc_ids[0]]["text"]
                    + " vs " + doc_names_disp[1] + " = " + cells[doc_ids[1]]["text"] + " "
                    + cells[doc_ids[0]]["marker"] + cells[doc_ids[1]]["marker"]
                    + " (the extracted values differ — review the cited "
                      "provisions; no legal effect is inferred here)"
                )
            else:
                both_explicit += 1

        # Gap detection: any side lacking an explicit value
        for did, dname in zip(doc_ids, doc_names_disp):
            st = cells[did]["status"]
            if "found_explicit" not in st:
                gaps.append(
                    "- " + dim["label"] + " — " + dname + ": " + st
                    + " (no explicit extracted value; this is not proof the "
                      "provision is absent from the contract)"
                )

        matrix_rows.append({
            "dimension": dim["label"], "field": dim["field"],
            "cells": {did: cells[did] for did in doc_ids},
        })

    # 4. Answer text (markdown: summary / matrix / differences / gaps)
    n_dims = len(DIMENSION_FIELDS)
    gap_dims = set(g.split(" — ")[0] for g in gaps)
    lines = []
    lines.append("**Executive summary**")
    lines.append(
        "Compared " + doc_names_disp[0] + " and " + doc_names_disp[1]
        + " across " + str(n_dims) + " extracted contract dimensions: "
        + str(both_explicit) + " have the same explicit value on both sides, "
        + str(len(differences)) + " differ materially, and "
        + str(len(gap_dims))
        + " dimensions could not be compared reliably on at least one side "
        "(extraction gaps)."
    )
    lines.append(
        "Clause-level dimensions (indemnification, liability, insurance, "
        "confidentiality, intellectual property) are not extracted fields — "
        "request a targeted clause search for those."
    )
    lines.append("")
    lines.append("**Comparative matrix** (extraction status in parentheses; "
                 "[n] cites the supporting evidence in the Sources panel):")
    lines.append("")
    header = "| Dimension | " + " | ".join(doc_names_disp) + " |"
    sep = "|---" * (len(doc_ids) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for row in matrix_rows:
        cells_md = []
        for did in doc_ids:
            c = row["cells"][did]
            txt = (c["text"] or "(no extraction record)").replace("|", "/")
            if len(txt) > 120:
                txt = txt[:117] + "..."
            cells_md.append(txt + c["marker"])
        lines.append("| " + row["dimension"] + " | " + " | ".join(cells_md) + " |")
    lines.append("")

    lines.append("**Material differences**")
    if differences:
        lines.extend(differences)
    else:
        lines.append(
            "No dimension has differing single explicit values on both "
            "sides (dimensions with extraction gaps are excluded from "
            "this count — see below)."
        )
    lines.append("")

    lines.append("**Evidence gaps and conflicts**")
    if gaps:
        lines.extend(gaps)
    else:
        lines.append("None — every dimension has an explicit value for both agreements.")
    lines.append("Note: " + MISSING_EXTRACTION_CAVEAT)
    if dim_failed:
        lines.append(
            "Extraction query failed for: " + ", ".join(dim_failed)
            + "; those dimensions could not be compared."
        )
    if ev_err:
        lines.append(ev_err)
    if capped_sources:
        lines.append(
            "Source citations are capped at " + str(MAX_COMPARISON_SOURCE_REFS)
            + " entries; additional cells show values without a citation."
        )
    lines.append("")
    lines.append(
        "The numbered citations map to the Sources panel; each entry is "
        "the extraction evidence backing that cell's claim."
    )
    answer_text = "\n".join(lines)

    result = {
        "stub": False,
        "capability": "comparison",
        "mode": "matrix",
        "comparison_aspect": "default",
        "coverage": {
            "documents_compared": len(resolved),
            "dimensions": n_dims,
            "cited_cells": len(sources),
            "material_differences": len(differences),
            "gap_dimensions": len(gap_dims),
        },
        "matrix": matrix_rows,
        "doc_names": doc_names_disp,
        "sources": sources,
        "limitations": ([
            "Clause-level dimensions are not extracted fields and are "
            "outside this matrix."
        ] + (["Extraction query failed for: " + ", ".join(dim_failed)] if dim_failed else [])),
        "answer_text": answer_text,
    }
    return {
        "evidence": [],
        "pending_action": None,
        "comparison_result": result,
        "trace": [make_trace(
            "comparison", action="completed", route="comparison",
            reason=(
                "matrix comparison: " + str(len(resolved)) + " docs x "
                + str(n_dims) + " dims, " + str(len(sources))
                + " cited cells, " + str(len(differences)) + " differences"
            ),
            aspect="default", num_compared=len(resolved),
            total_evidence=len(sources),
        )],
    }


def make_comparison_node(
    retriever: ComparisonRetrieverLike,
    sql_executor: SqlExecutorLike,
    answer_generator=None,
    num_results: int = DEFAULT_PER_DOC_RESULTS,
    max_docs: int = MAX_COMPARISON_DOCS,
    max_per_doc: int = DEFAULT_PER_DOC_RESULTS,
):
    """Factory: create a LangGraph contract-comparison node.

    Args:
        retriever: anything exposing search(query, num_results, doc_ids).
        sql_executor: SqlExecutorLike for canonical-doc enumeration,
                      name resolution, and optional structured-field lookup.
        answer_generator: optional callable(query, context_chunks) -> str.
        num_results: default per-document top-k.
        max_docs: execution limit (MAX_COMPARISON_DOCS = 10).
        max_per_doc: per-document top-k (overrides num_results).
    """
    per_doc_k = max_per_doc if max_per_doc else num_results

    def comparison_node(state: dict) -> dict:
        intent = state.get("intent")
        user_query = state.get("user_query", "")

        def _iget(name, default=None):
            if intent is None:
                return default
            if isinstance(intent, dict):
                return intent.get(name, default)
            return getattr(intent, name, default)

        aspect = _iget("comparison_aspect") or "unspecified"
        target_doc_names = list(_iget("target_doc_names") or [])
        search_query = aspect if aspect != "unspecified" else user_query

        # ── Fix 010: mode dispatch ──
        # candidates: 'what documents can be compared?' — inventory-backed
        # discovery, no chunk evidence, sets pending_action.
        if aspect == "candidates":
            return _run_candidate_discovery(sql_executor)
        # default: an ACCEPTED comparison ('sure compare them then' resolved
        # by the planner) — multi-dimension matrix from extracted fields
        # with per-cell evidence.
        if aspect == "default":
            try:
                all_canonical = sql_executor.execute(_CANDIDATE_DOCS_SQL, {}) or []
            except Exception as exc:
                return {
                    "error": ("Comparison document enumeration failed: " + str(exc))[:300],
                    "evidence": [],
                    "trace": [make_trace(
                        "comparison", action="failed", route="comparison",
                        reason=str(exc)[:300], step="enumerate_documents",
                    )],
                }
            return _run_matrix_comparison(
                sql_executor, all_canonical, target_doc_names, user_query,
            )

        # 1. Enumerate eligible documents
        try:
            all_canonical = sql_executor.execute(_CANONICAL_DOCS_SQL, {})
        except Exception as exc:
            return {
                "error": ("Comparison document enumeration failed: " + str(exc))[:300],
                "evidence": [],
                "trace": [make_trace(
                    "comparison", action="failed", route="comparison",
                    reason=str(exc)[:300], step="enumerate_documents",
                )],
            }

        if not all_canonical:
            result = {
                "stub": False,
                "capability": "comparison",
                "comparison_aspect": aspect,
                "coverage": {
                    "documents_compared": 0,
                    "documents_with_evidence": 0,
                    "documents_no_evidence": 0,
                    "documents_failed": 0,
                },
                "per_doc_findings": [],
                "limitations": ["No canonical documents found in the corpus."],
                "answer_text": "No canonical documents found in the corpus.",
            }
            return {
                "evidence": [],
                "comparison_result": result,
                "trace": [make_trace(
                    "comparison", action="completed", route="comparison",
                    reason="no canonical documents", aspect=aspect,
                )],
            }

        # 2. Resolve document names
        resolved, ambiguous, unresolved = _resolve_doc_names(
            sql_executor, all_canonical, target_doc_names,
        )
        unresolved_names = [u["requested_name"] for u in unresolved]

        # 3. Require at least 2 resolved documents
        if len(resolved) < 2:
            msg_parts = []
            if unresolved_names:
                msg_parts.append(
                    f"Could not find documents matching: {', '.join(unresolved_names)}."
                )
            for a in ambiguous:
                msg_parts.append(
                    f"Name '{a['requested_name']}' is ambiguous -- "
                    f"matches: {', '.join(a['matching_files'])}."
                )
            if len(resolved) == 1:
                msg_parts.append(
                    "At least two agreements must be identified for comparison. "
                    f"Only one was resolved: {_to_str(resolved[0].get('file_name'))}."
                )
            if not msg_parts:
                msg_parts.append(
                    "Could not resolve at least two document names for comparison."
                )
            msg = " ".join(msg_parts)
            result = {
                "stub": False,
                "capability": "comparison",
                "comparison_aspect": aspect,
                "coverage": {
                    "documents_compared": 0,
                    "documents_with_evidence": 0,
                    "documents_no_evidence": 0,
                    "documents_failed": 0,
                },
                "per_doc_findings": [],
                "unresolved_names": unresolved_names,
                "ambiguous_names": ambiguous,
                "limitations": [msg],
                "answer_text": msg,
            }
            return {
                "evidence": [],
                "comparison_result": result,
                "trace": [make_trace(
                    "comparison", action="clarification", route="comparison",
                    reason=msg[:200], aspect=aspect,
                    unresolved=unresolved_names, ambiguous=ambiguous,
                )],
            }

        # 4. Enforce MAX_COMPARISON_DOCS
        if len(resolved) > max_docs:
            msg = (
                f"Comparison is limited to {max_docs} agreements. "
                f"{len(resolved)} were resolved: "
                + ", ".join(_to_str(d.get("file_name")) for d in resolved)
                + ". Please narrow your request."
            )
            result = {
                "stub": False,
                "capability": "comparison",
                "comparison_aspect": aspect,
                "coverage": {
                    "documents_compared": 0,
                    "documents_with_evidence": 0,
                    "documents_no_evidence": 0,
                    "documents_failed": 0,
                },
                "per_doc_findings": [],
                "unresolved_names": unresolved_names,
                "ambiguous_names": ambiguous,
                "limitations": [msg],
                "answer_text": msg,
            }
            return {
                "evidence": [],
                "comparison_result": result,
                "trace": [make_trace(
                    "comparison", action="scope_limited", route="comparison",
                    reason=f"{len(resolved)} docs > limit {max_docs}",
                    aspect=aspect, num_requested=len(resolved),
                )],
            }

        # 5. Map aspect to structured field
        aspect_field = _map_aspect_to_field(aspect)
        structured_by_doc = {}
        has_structured = False

        if aspect_field:
            try:
                doc_ids = [_to_str(d.get("doc_id")) for d in resolved]
                sql, params = _build_field_query(aspect_field, doc_ids)
                field_rows = sql_executor.execute(sql, params)
                for row in field_rows or []:
                    did = _to_str(row.get("doc_id"))
                    structured_by_doc.setdefault(did, []).append({
                        "value_raw": _to_str(row.get("value_raw")),
                        "value_normalized": _to_str(row.get("value_normalized")),
                        "value_status": _to_str(row.get("value_status")),
                        "extraction_method": _to_str(row.get("extraction_method")),
                        "field_id": _to_str(row.get("field_id")),
                    })
                has_structured = True
            except Exception:
                pass

        # 6. Per-document evidence retrieval
        all_evidence = []
        per_doc_findings = []

        for doc in resolved:
            doc_id = _to_str(doc.get("doc_id"))
            doc_name = _to_str(doc.get("file_name")) or doc_id
            doc_evidence = []
            doc_error = None

            try:
                results = retriever.search(
                    search_query,
                    num_results=per_doc_k,
                    doc_ids=[doc_id],
                )
                doc_evidence = [search_result_to_evidence(r) for r in (results or [])]
            except Exception as exc:
                doc_error = str(exc)[:200]

            for ev in doc_evidence:
                ev.coverage_status = "found" if doc_evidence else "no_match"

            all_evidence.extend(doc_evidence)

            if doc_error:
                status = "failed"
            elif doc_evidence:
                status = "searched_with_evidence"
            else:
                status = "searched_no_evidence"

            finding = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "status": status,
                "num_evidence": len(doc_evidence),
                "evidence_items": [
                    {
                        "chunk_id": ev.chunk_id,
                        "section_path": ev.section_path,
                        "page_start": ev.page_start,
                        "page_end": ev.page_end,
                        "supporting_text": ev.supporting_text,
                        "retrieval_score": ev.retrieval_score,
                    }
                    for ev in doc_evidence
                ],
            }
            if doc_error:
                finding["error"] = doc_error
            if has_structured:
                finding["structured_values"] = structured_by_doc.get(doc_id, [])

            per_doc_findings.append(finding)

        # 7. Apply total evidence cap
        evidence_truncated = False
        if len(all_evidence) > MAX_TOTAL_EVIDENCE:
            all_evidence = all_evidence[:MAX_TOTAL_EVIDENCE]
            evidence_truncated = True

        # 8. Compute coverage
        num_compared = len(resolved)
        num_with_evidence = sum(
            1 for f in per_doc_findings
            if f["status"] == "searched_with_evidence"
        )
        num_no_evidence = sum(
            1 for f in per_doc_findings
            if f["status"] == "searched_no_evidence"
        )
        num_failed = sum(
            1 for f in per_doc_findings if f["status"] == "failed"
        )

        coverage = {
            "documents_compared": num_compared,
            "documents_with_evidence": num_with_evidence,
            "documents_no_evidence": num_no_evidence,
            "documents_failed": num_failed,
        }

        # 9. Build limitations
        limitations = []
        if unresolved_names:
            limitations.append(
                f"Unresolved document names not compared: {', '.join(unresolved_names)}."
            )
        if ambiguous:
            for a in ambiguous:
                limitations.append(
                    f"Ambiguous name '{a['requested_name']}' -- "
                    f"matches: {', '.join(a['matching_files'])}. Not compared."
                )
        if evidence_truncated:
            limitations.append(
                f"Evidence truncated to {MAX_TOTAL_EVIDENCE} items. "
                f"Some candidate passages may not be shown."
            )
        if num_failed > 0:
            limitations.append(
                f"{num_failed} document(s) failed during retrieval. "
                f"Comparison is incomplete for those agreements."
            )

        # 10. Build answer text
        answer_text = _build_answer_text(
            aspect, per_doc_findings, coverage, limitations, has_structured,
        )

        # 11. Optional grounded generation
        if answer_generator is not None and all_evidence:
            try:
                doc_names = [_to_str(d.get("file_name")) for d in resolved]
                prompt = _build_comparison_prompt(aspect, doc_names)
                from .answers import evidence_to_context_chunks
                context_chunks = evidence_to_context_chunks(all_evidence)
                generated = answer_generator(prompt, context_chunks)
                if generated:
                    answer_text = generated
            except Exception:
                pass

        # Stage A remediation (2026-10-29): a genuine retrieval failure must
        # not masquerade as "no evidence found" or a no-evidence abstention
        # (spec 2.8).  Surface it as a state error so evidence_validation
        # passes it through and final_answer reports the failure distinctly;
        # the per-document 'failed' findings remain in the trace and result.
        node_error = None
        if num_failed > 0:
            node_error = (
                f"Retrieval failed for {num_failed} of {num_compared} "
                f"document(s); comparison is incomplete for those agreements. "
                f"Please try again later."
            )

        # 12. Build result
        result = {
            "stub": False,
            "capability": "comparison",
            "comparison_aspect": aspect,
            "aspect_field": aspect_field,
            "coverage": coverage,
            "per_doc_findings": per_doc_findings,
            "unresolved_names": unresolved_names,
            "ambiguous_names": ambiguous,
            "limitations": limitations,
            "answer_text": answer_text,
        }

        trace_details = {
            "aspect": aspect,
            "aspect_field": aspect_field,
            "num_compared": num_compared,
            "with_evidence": num_with_evidence,
            "no_evidence": num_no_evidence,
            "failed": num_failed,
            "total_evidence": len(all_evidence),
        }

        return {
            "evidence": all_evidence,
            "comparison_result": result,
            "error": node_error,
            "trace": [make_trace(
                "comparison",
                action="completed",
                route="comparison",
                reason=(
                    f"{num_compared} docs compared, "
                    f"{num_with_evidence} with evidence"
                ),
                **trace_details,
            )],
        }

    return comparison_node
