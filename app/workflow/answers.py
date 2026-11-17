"""Final Answer generation + Source Reference assembly (Milestone 2B.1).

Replaces the foundation stub's final_answer node with a factory that can
be injected with the existing response-generation implementation.  The
injected generator matches LakebaseRetriever.generate_response(query,
context_chunks) -> str — no LLM integration is duplicated; production
passes the bound method of the real retriever singleton.

Generation happens only AFTER evidence validation (this node runs after
evidence_validation in the graph), only when validated evidence exists,
and never when the state is abstaining or already errored.

Source references: structured SourceRef objects built from the validated
evidence, deduplicated by chunk and capped at MAX_SOURCE_DOCS (3)
distinct documents for the initial source display (spec Section 1.7).
Section and page metadata is preserved.  Physical PDF opening and
bounding-box highlighting are out of scope for this slice.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .state import EvidenceItem, MAX_SOURCE_DOCS, SourceGroup, SourceRef
from .traces import make_trace
from .table_answer import build_table_answer, html_table_to_markdown


ABSTAIN_MESSAGE = (
    "The available documents do not contain enough information to answer this question."
)
EXCERPT_CHARS = 500

# Fix 010 (rules E/F): the answer generator is prompt-instructed to return
# this exact sentence when the retrieved evidence does not support an
# answer (lakebase_retriever.generate_response).  A generated abstention
# must NOT display the retrieved chunks as sources — they were examined
# and found insufficient, so showing them would present irrelevant
# passages as evidence for a refusal.
GENERATED_ABSTAIN_SENTENCES = (
    ABSTAIN_MESSAGE,
    "The available documents don't contain enough information to answer this question.",
)


def _is_generated_abstention(answer: str) -> bool:
    """True when a generated answer is the instructed abstention sentence."""
    a = " ".join((answer or "").split())
    return any(s in a for s in GENERATED_ABSTAIN_SENTENCES)


def evidence_to_context_chunks(evidence: list[EvidenceItem]) -> list[dict]:
    """Convert validated EvidenceItems back to the chunk-dict shape consumed
    by LakebaseRetriever.generate_response (chunk_text, doc_name,
    section_path, page_start, page_end, ...).

    Fix 011 (2026-10-30, Stage C): table chunks additionally carry
    'table_markdown' — the preserved table structure rendered as a Markdown
    pipe table.  The retriever's context assembly uses it INSTEAD of the
    flattened chunk_text, so the LLM sees the actual table, not glued
    values.
    """
    chunks = []
    for e in evidence:
        chunk = {
            "chunk_id": e.chunk_id,
            "doc_id": e.doc_id,
            "doc_name": e.doc_name,
            "section_path": e.section_path,
            "page_start": e.page_start,
            "page_end": e.page_end,
            "chunk_text": e.supporting_text or "",
        }
        if e.table_html:
            md = html_table_to_markdown(e.table_html)
            if md:
                chunk["table_markdown"] = md
        chunks.append(chunk)
    return chunks


def build_source_refs(
    evidence: list[EvidenceItem],
    max_docs: int = MAX_SOURCE_DOCS,
) -> list[SourceRef]:
    """Build structured source references from validated evidence.

    Deduplicates by chunk_id (one ref per chunk) and caps the number of
    DISTINCT documents at max_docs (3), taking documents in evidence
    (fused-rank) order.  Preserves document name, section path, and page
    range; the excerpt is the supporting text truncated to 500 chars.
    """
    refs: list[SourceRef] = []
    seen_chunks: set = set()
    seen_docs: list = []
    for e in evidence:
        if e.chunk_id in seen_chunks:
            continue
        doc_key = e.doc_id or e.doc_name
        if doc_key not in seen_docs:
            if len(seen_docs) >= max_docs:
                break
            seen_docs.append(doc_key)
        seen_chunks.add(e.chunk_id)
        refs.append(
            SourceRef(
                doc_id=e.doc_id,
                doc_name=e.doc_name,
                section_path=e.section_path,
                page_start=e.page_start,
                page_end=e.page_end,
                excerpt=(e.supporting_text or "")[:EXCERPT_CHARS],
                # Stage 3 fix (2026-10-29): carry the chunk identity through so
                # the source viewer can fetch the authoritative chunk bboxes
                # from chunks_v2 for exact highlighting on hybrid routes.
                chunk_id=e.chunk_id,
            )
        )
    return refs


def _cited_indices(answer: str) -> list[int]:
    """Inline [n] citation numbers used in an answer, 1-based, sorted."""
    return sorted(
        {int(m) for m in re.findall(r"\[(\d+)\]", answer or "") if m.isdigit()}
    )


def select_and_group_sources(
    evidence: list[EvidenceItem],
    answer: Optional[str] = None,
    max_docs: int = MAX_SOURCE_DOCS,
) -> list[SourceRef]:
    """Fix 011 (2026-10-30, Stage B): claim-cited, grouped source selection.

    Two rules replace the old 'show every technically-validated chunk':

    1. CLAIM FILTER — when the answer carries inline [n] citations (the
       synthesis prompt requires one per factual claim), ONLY the cited
       evidence positions become sources (1-based position in the evidence
       order, matching the [Document n] context headers).  Without
       citations, all evidence is eligible (honest fallback).

    2. GROUPING — chunks from the same document AND section merge into ONE
       source entry with the section's merged PDF page range, so a
       provision split across chunks (or a multi-page table) appears as a
       single coherent citation instead of several lookalike rows.

    Distinct documents are still capped at max_docs (3) in evidence order.
    """
    if not evidence:
        return []
    cited = _cited_indices(answer or "")
    selected = [evidence[i - 1] for i in cited if 1 <= i <= len(evidence)]
    if not selected:
        selected = list(evidence)

    groups: dict[tuple, list[EvidenceItem]] = {}
    group_order: list[tuple] = []
    for e in selected:
        key = (e.doc_id, e.section_path)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(e)

    doc_order: list[str] = []
    for key in group_order:
        d = groups[key][0].doc_id
        if d not in doc_order:
            doc_order.append(d)
    allowed_docs = set(doc_order[:max_docs])

    refs: list[SourceRef] = []
    for key in group_order:
        items = groups[key]
        if items[0].doc_id not in allowed_docs:
            continue
        first = items[0]
        pages_start = [e.page_start for e in items if e.page_start is not None]
        pages_end = [e.page_end for e in items if e.page_end is not None]
        excerpt_item = max(items, key=lambda e: len(e.supporting_text or ""))
        refs.append(
            SourceRef(
                doc_id=first.doc_id,
                doc_name=first.doc_name,
                section_path=first.section_path,
                page_start=min(pages_start) if pages_start else None,
                page_end=max(pages_end) if pages_end else None,
                excerpt=(excerpt_item.supporting_text or "")[:EXCERPT_CHARS],
                chunk_id=first.chunk_id,
            )
        )
    return refs


def build_source_groups(
    evidence: list[EvidenceItem],
    answer: Optional[str] = None,
    max_docs: int = MAX_SOURCE_DOCS,
) -> list[SourceGroup]:
    """Fix 012 (2026-10-31, Defect B): document-level source grouping.

    Groups supporting evidence by CANONICAL doc_id - one top-level group
    per document - while every passage stays individually addressable.
    The same claim filter as select_and_group_sources applies: with inline
    [n] citations only the cited context positions become evidence items,
    and each item records its [n] position in evidence_ref so inline
    citations resolve to the exact passage.  Without citations all
    evidence is eligible (honest fallback - e.g. table answers, whose
    answer text carries no [n] markers).

    Distinct documents are capped at max_docs (3) in first-cited order.
    Chunks are deduplicated by chunk_id; section identity and page ranges
    per item are preserved - nothing is merged away.
    """
    if not evidence:
        return []
    cited = _cited_indices(answer or "")
    positions = set(cited) if cited else None

    items_by_doc: dict = {}
    doc_order: list = []
    doc_meta: dict = {}
    seen_chunks: dict = {}
    for pos, e in enumerate(evidence, start=1):
        if positions is not None and pos not in positions:
            continue
        doc_key = e.doc_id or e.doc_name or "(unknown)"
        if doc_key not in items_by_doc:
            items_by_doc[doc_key] = []
            doc_order.append(doc_key)
            doc_meta[doc_key] = e
            seen_chunks[doc_key] = set()
        if e.chunk_id and e.chunk_id in seen_chunks[doc_key]:
            continue
        if e.chunk_id:
            seen_chunks[doc_key].add(e.chunk_id)
        items_by_doc[doc_key].append(
            SourceRef(
                doc_id=e.doc_id or "",
                doc_name=e.doc_name or "",
                section_path=e.section_path,
                page_start=e.page_start,
                page_end=e.page_end,
                excerpt=(e.supporting_text or "")[:EXCERPT_CHARS],
                chunk_id=e.chunk_id,
                evidence_ref=pos if positions is not None else None,
            )
        )

    groups: list = []
    for doc_key in doc_order[:max_docs]:
        items = items_by_doc[doc_key]
        if not items:
            continue
        first = doc_meta[doc_key]
        pages_start = [r.page_start for r in items if r.page_start is not None]
        pages_end = [r.page_end for r in items if r.page_end is not None]
        sections = {r.section_path for r in items}
        groups.append(
            SourceGroup(
                doc_id=first.doc_id or "",
                doc_name=first.doc_name or "",
                section_path=next(iter(sections)) if len(sections) == 1 else None,
                page_start=min(pages_start) if pages_start else None,
                page_end=max(pages_end) if pages_end else None,
                evidence=items,
            )
        )
    return groups


def group_refs_by_doc(
    refs,
    max_docs: int = MAX_SOURCE_DOCS,
) -> list[SourceGroup]:
    """Fix 012 (2026-10-31, Defect B): group already-built SourceRefs
    (structured-lookup / comparison sources, or legacy history messages
    carrying only source_refs) by canonical doc_id into SourceGroups.
    Ref order, sections, pages, excerpts, and evidence_ref values are
    preserved as-is; refs without doc_id are grouped under the same
    '(unknown)' key so they stay visible (never silently dropped).
    """
    def _get(r, k, default=None):
        return r.get(k, default) if isinstance(r, dict) else getattr(r, k, default)

    items_by_doc: dict = {}
    doc_order: list = []
    for r in refs or []:
        doc_key = _get(r, "doc_id") or _get(r, "doc_name") or "(unknown)"
        if doc_key not in items_by_doc:
            items_by_doc[doc_key] = []
            doc_order.append(doc_key)
        items_by_doc[doc_key].append(r)

    groups: list = []
    for doc_key in doc_order[:max_docs]:
        items = items_by_doc[doc_key]
        r0 = items[0]
        pages_start = [p for p in (_get(x, "page_start") for x in items) if p is not None]
        pages_end = [p for p in (_get(x, "page_end") for x in items) if p is not None]
        sections = {s for s in (_get(x, "section_path") for x in items) if s}
        groups.append(
            SourceGroup(
                doc_id=_get(r0, "doc_id"),
                doc_name=_get(r0, "doc_name") or "(unknown)",
                section_path=next(iter(sections)) if len(sections) == 1 else None,
                page_start=min(pages_start) if pages_start else None,
                page_end=max(pages_end) if pages_end else None,
                evidence=list(items),
            )
        )
    return groups


def make_final_answer_node(
    answer_generator: Optional[Callable[[str, list[dict]], str]] = None,
):
    """Factory: create the final-answer LangGraph node.

    Args:
        answer_generator: callable(query, context_chunks) -> str matching
                          LakebaseRetriever.generate_response.  When None
                          (foundation/tests), behaves exactly like the
                          original stub final_answer_node.
    """

    def final_answer_node(state: dict) -> dict:
        abstain = bool(state.get("abstain"))
        error = state.get("error")
        evidence = state.get("evidence", []) or []

        source_refs: list[SourceRef] = []
        source_groups: list = []
        generated = False
        gen_error: Optional[str] = None
        cited_indices: list[int] = []

        # Fix 011 (2026-10-30, Stage C): a 'show me the patents/table' style
        # request gets a DETERMINISTIC structured-table answer built from
        # the preserved table_html — never the prose LLM template (which
        # flattens tables and corrupts cells).  Only applies when no earlier
        # node already set an answer, the run is not abstaining or erroring,
        # and the evidence actually contains a usable table.
        table_payload = None
        if not abstain and not error and not state.get("answer") and evidence:
            table_payload = build_table_answer(state.get("user_query", ""), evidence)

        if abstain:
            # Rule: never generate an answer when abstaining (spec 2.7).
            # Exception (D11 completion, 2026-10-29): when a capability node
            # explicitly asked the user for clarification (e.g. the requested
            # document names could not be resolved, so no search was run),
            # that clarification question must reach the user -- the generic
            # abstention message would mask it and misrepresent a
            # name-resolution problem as a no-evidence outcome.
            clarification = None
            for t in state.get("trace") or []:
                if isinstance(t, dict) and t.get("action") == "clarification":
                    clarification = t.get("reason")
                    break
            if clarification:
                answer = clarification
            else:
                answer = ABSTAIN_MESSAGE
        elif error:
            answer = f"Error: {error}"
            # Stage A remediation (2026-10-29): a partially failed
            # multi-document run still shows its partial results (spec 2.8:
            # return partial results if any evidence was collected), and the
            # failure itself is reported distinctly from a no-evidence
            # abstention — a retrieval failure must never masquerade as a
            # legitimate "no supporting evidence" outcome.
            for key in ("metadata_result", "comparison_result"):
                meta = state.get(key)
                if isinstance(meta, dict) and not meta.get("stub") and meta.get("answer_text"):
                    answer = f"{answer}\n\nPartial results:\n{meta['answer_text']}"
                    break
            if evidence:
                source_refs = build_source_refs(evidence)
        elif state.get("answer"):
            answer = state["answer"]
            if evidence:
                source_refs = build_source_refs(evidence)
        elif table_payload:
            # Fix 011 (Stage C): deterministic table answer — exact cell
            # values from the source table; the UI renders the structure.
            answer = table_payload["answer_text"]
            source_refs = table_payload["source_refs"]
        elif state.get("metadata_result"):
            meta = state["metadata_result"]
            if meta.get("stub"):
                cap = meta.get("capability", "unknown")
                label = cap.replace("_", " ").title()
                answer = f"[{label}] {meta.get('message', 'Stub result.')}"
            elif meta.get("capability") == "clause_discovery":
                # Milestone 3: clause-discovery results carry a pre-formatted
                # answer_text (coverage summary + per-document status) AND
                # candidate-passage evidence.  Build source refs from evidence
                # so citations are available alongside the coverage report.
                answer = meta.get("answer_text") or str(meta)
                if evidence:
                    source_refs = build_source_refs(evidence)
            else:
                # Milestone 3: real corpus-overview results carry a
                # pre-formatted answer_text (SQL aggregate counts and
                # distributions, no LLM generation).
                answer = meta.get("answer_text") or str(meta)
        elif state.get("lookup_result"):
            lookup = state["lookup_result"]
            if lookup.get("stub"):
                answer = f"[Structured Lookup] {lookup.get('message', 'Stub result.')}"
            else:
                # Milestone 3: real structured-lookup results carry a
                # pre-formatted answer_text (raw extracted data, no LLM
                # generation of the value itself) and optional sources
                # (linked evidence quotes) for citation display.
                answer = lookup.get("answer_text") or str(lookup)
                sources = lookup.get("sources") or []
                if sources:
                    source_refs = [
                        SourceRef(**s)
                        for s in sources[:MAX_SOURCE_DOCS]
                    ]
        elif state.get("comparison_result"):
            comp = state["comparison_result"]
            if comp.get("stub"):
                answer = f"[Comparison] {comp.get('message', 'Stub result.')}"
            else:
                # Milestone 3: real comparison results carry a pre-formatted
                # answer_text (structured comparison with per-document
                # findings, coverage, and caveats) AND evidence.  Build
                # source refs from evidence, capped at MAX_SOURCE_DOCS (3)
                # distinct documents.  Underlying evidence from additional
                # compared agreements is preserved in the comparison_result.
                answer = comp.get("answer_text") or str(comp)
                if comp.get("sources"):
                    # Fix 010: the metadata-backed matrix comparison carries
                    # claim-mapped sources (one per cited matrix cell,
                    # doc-bound extraction evidence) — use them instead of
                    # chunk-evidence source refs (there are none in that
                    # mode; sources substantiate the specific matrix rows
                    # and material differences they accompany).
                    source_refs = [SourceRef(**s) for s in comp["sources"]]
                elif evidence:
                    source_refs = build_source_refs(evidence)
        elif answer_generator is not None and evidence:
            # Grounded generation from validated evidence only
            try:
                answer = answer_generator(
                    state.get("user_query", ""),
                    evidence_to_context_chunks(evidence),
                )
                generated = True
                if _is_generated_abstention(answer):
                    # Fix 010 (rules E/F): the model examined the evidence
                    # and abstained — the chunks were judged insufficient,
                    # so they are NOT displayable sources for the refusal.
                    source_refs = []
                else:
                    # Fix 011 (Stage B): display only the evidence actually
                    # cited in the answer ([n] positions), merged by document
                    # and section — not every technically-validated chunk.
                    cited_indices = _cited_indices(answer)
                    source_refs = select_and_group_sources(evidence, answer)
            except Exception as exc:
                gen_error = f"Answer generation failed: {exc}"
                answer = f"Error: {gen_error}"
        else:
            answer = "No answer generated."

        # Fix 012 (2026-10-31, Defect B): document-level source groups for
        # the UI.  Grounded [n]-cited answers group the CITED evidence with
        # per-item [n] mapping; table answers group their table evidence
        # (no [n] markers in a table answer text); every other route groups
        # whatever source_refs were already built.
        if not source_groups:
            if table_payload and evidence:
                source_groups = build_source_groups(evidence)
            elif generated and evidence:
                source_groups = build_source_groups(evidence, answer)
            elif source_refs:
                source_groups = group_refs_by_doc(source_refs)

        updates: dict = {
            "answer": answer,
            "source_refs": source_refs,
            "source_groups": source_groups,
            # Fix 011 (Stage C): structured table payload for the UI (None on
            # prose answers keeps history replay simple).
            "table_result": table_payload["table_result"] if table_payload else None,
            "trace": [
                make_trace(
                    "final_answer",
                    action="table_answer" if table_payload else "generated",
                    reason=(
                        "deterministic structured-table answer (exact cell values)"
                        if table_payload else "final answer formatted"
                    ),
                    abstain=abstain,
                    has_error=bool(error or gen_error),
                    generated=generated,
                    num_sources=len(source_refs),
                    cited_indices=cited_indices,
                    table_answer=bool(table_payload),
                )
            ],
        }
        if gen_error:
            updates["error"] = gen_error
        return updates

    return final_answer_node
