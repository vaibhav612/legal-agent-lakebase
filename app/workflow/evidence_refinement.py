"""Post-retrieval evidence refinement (fix 011, Stages A/B, 2026-10-30).

Separates the evidence lifecycle states that the behavior spec previously
conflated (hosted findings, 2026-10-30):

  1. retrieved candidate chunks      (hybrid top-k fusion output)
  2. technically usable evidence      (evidence_validation: text/dedup/budget)
  3. question-relevant evidence       (THIS node: topic-term filter)
  4. claim-supporting evidence        (final_answer: cited [n] groups)
  5. displayed sources               (final_answer: grouped SourceRefs)

Hosted defect this fixes: for 'What are the insurance requirements under
the Upjohn manufacturing and supply agreement?' the top-5 fusion results
were cover page, introduction, Entire Agreement, Witnesseth, and only the
OPENING of Section 11 — all five passed technical validation and all five
were displayed as sources, while the amounts/limits subclauses (which rank
7th/15th in fused order) never entered the context, so the LLM could only
paraphrase the opening sentence or abstain.

Refinement, deterministic and LLM-free:

  * TOPIC FILTER — score each chunk by term-frequency overlap with the
    query's TOPIC terms: query content words minus generic legal terms and
    minus words that appear in the retrieved documents' FILE NAMES (those
    identify the document, not the topic: 'upjohn', 'manufacturing',
    'supply' match the cover page but say nothing about insurance).
    Keep chunks scoring >= 50% of the best chunk (a RELATIVE rule — no
    absolute hardcoded relevance threshold).  When no topic terms can be
    determined, or none match anywhere, no filtering is applied (honest
    fallback, never a silent drop).

  * SECTION EXPANSION — for each surviving chunk, fetch the SAME SECTION's
    remaining chunks (and chunk_index-adjacent neighbours) so a provision
    split across chunks enters evidence whole (Section 11's limits/
    endorsement subclauses; a schedule continued over PDF pages).  Expansion
    only expands sections that look like real provisions (numbered
    sections / Schedules / Articles) — never title or recital sections,
    which would pull in boilerplate.  Expanded items are marked
    evidence_role='section_expansion'.

The trace entry exposes every count so the UI trace shows the true
decision chain instead of a generic '5 usable evidence items'.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Callable, Optional

from .state import EvidenceItem
from .traces import make_trace

logger = logging.getLogger(__name__)


# ── Topic-term extraction ──────────────────────────────────────────

# Words that appear in almost every contract and say nothing about the
# question's subject (they are excluded from topic scoring).
_GENERIC_LEGAL_TERMS: frozenset[str] = frozenset({
    "agreement", "agreements", "contract", "contracts", "document",
    "documents", "section", "sections", "clause", "clauses", "party",
    "parties", "term", "terms", "inc", "llc", "corp", "corporation",
    "company", "ltd", "co", "dated", "form", "exhibit", "pursuant",
    "thereof", "herein", "hereunder", "what", "which", "who", "whose",
    "when", "where", "does", "did", "are", "was", "were", "have", "has",
    "had", "the", "and", "for", "with", "under", "from", "that", "this",
    "you", "your", "can", "could", "would", "should", "shall", "will",
    "me", "my", "i", "a", "an", "of", "in", "on", "to", "is", "it",
    "tell", "about", "please", "any", "all", "per",
})


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9&./-]*", (text or "").lower())
            if len(t) > 1]


def topic_terms(query: str, doc_names: list[str]) -> list[str]:
    """Query content words that describe the QUESTION'S SUBJECT.

    Removes generic legal/English words and removes words that appear in
    the retrieved documents' file names (document-identifying words match
    boilerplate like cover pages).  When EVERY content word is
    document-identifying ('Tell me about the Upjohn manufacturing and
    supply agreement'), no topic terms remain — the honest 'cannot
    determine' result (empty list).  The caller then keeps ALL evidence
    rather than filtering to the chunk that best matches the document's
    own name (which would select the cover page and drop the substance).
    """
    content = [t for t in _tokens(query) if t not in _GENERIC_LEGAL_TERMS]
    if not content:
        return []
    title_words: set[str] = set()
    for name in doc_names or []:
        title_words.update(_tokens(name))

    def _titleish(term: str) -> bool:
        # Exact title token, or a >=4-char prefix match: file names glue the
        # company name into one token ('UpjohnInc'), so the query word
        # ('upjohn') must still count as document-identifying.
        return any(
            tw == term
            or (len(term) >= 4 and (tw.startswith(term) or term.startswith(tw) and len(tw) >= 4))
            for tw in title_words
        )

    focus = [t for t in content if not _titleish(t)]
    # Fix 011 (follow-up to Stage A, 2026-10-30): do NOT fall back to the
    # unfiltered content words.  focus is empty exactly when every content
    # word is document-identifying — 'honest fallback' then means KEEP ALL
    # evidence (empty topic terms), never 'filter by the document name'
    # (which returned only the cover page for whole-document queries).
    return focus


def _focus_score(evidence_item: EvidenceItem, focus: list[str]) -> float:
    """Term-frequency overlap of one chunk with the topic terms.

    Section-path matches count double (the section heading is strong
    evidence the chunk is about the topic).
    """
    section = (evidence_item.section_path or "").lower()
    text = (evidence_item.supporting_text or "").lower()
    score = 0.0
    for term in focus:
        sec_tf = section.count(term)
        txt_tf = text.count(term)
        if sec_tf:
            score += 2.0 * math.log(1 + sec_tf)
        if txt_tf:
            score += math.log(1 + txt_tf)
    return score


#: Keep chunks scoring at least this fraction of the best chunk's score.
#: A RELATIVE rule (scales with the actual score distribution) — not an
#: absolute semantic threshold (spec 2.6 forbids those pending calibration).
RELEVANCE_RELATIVE_KEEP = 0.5


def filter_relevant(
    query: str, evidence: list[EvidenceItem],
) -> tuple[list[EvidenceItem], list[str], list[EvidenceItem]]:
    """Split retrieved candidates into (relevant, focus_terms, dropped).

    Never returns an empty 'relevant' list for non-empty evidence: the
    best-scoring chunk always survives (it defines the threshold).  When
    topic terms cannot be determined or match nowhere, everything is kept.
    """
    if not evidence:
        return [], [], []
    doc_names = [e.doc_name for e in evidence if e.doc_name]
    focus = topic_terms(query, doc_names)
    if not focus:
        return evidence, [], []
    scored = [(e, _focus_score(e, focus)) for e in evidence]
    top = max(s for _, s in scored)
    if top <= 0.0:
        # No topic term occurs anywhere — cannot judge relevance honestly.
        return evidence, focus, []
    keep_at = RELEVANCE_RELATIVE_KEEP * top
    relevant = [e for e, s in scored if s >= keep_at]
    dropped = [e for e, s in scored if s < keep_at]
    return relevant, focus, dropped


# ── Section / adjacent expansion ──────────────────────────────────

# A section path is expandable only when it names a real provision:
# numbered sections ('11. Insurance.'), Schedules, Articles, Exhibits.
# Title/recital sections ('MANUFACTURING AND SUPPLY AGREEMENT',
# 'W I T N E S S E T H:', '') are NOT expandable — expanding them would
# pull boilerplate chunks into the evidence.
_EXPANDABLE_SECTION: re.Pattern = re.compile(
    r"^\s*(?:\d+[.)]|schedule\b|article\b|exhibit\b|annex\b|appendix\b)",
    re.IGNORECASE,
)

#: Maximum chunks added by expansion per query (bounded context growth).
MAX_EXPANSION_CHUNKS = 8


def _chunk_index(chunk_id: str) -> Optional[int]:
    m = re.search(r"_(\d+)$", chunk_id or "")
    return int(m.group(1)) if m else None


def _result_to_evidence(row: dict) -> EvidenceItem:
    from urllib.parse import unquote
    doc_name = row.get("doc_name") or row.get("file_name") or row.get("doc_id") or "Unknown"
    return EvidenceItem(
        doc_id=str(row.get("doc_id") or "") or "unknown_doc",
        doc_name=str(unquote(doc_name)) if row.get("doc_name") else str(doc_name),
        chunk_id=str(row.get("chunk_id") or "") or "unknown_chunk",
        section_path=row.get("section_path"),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        supporting_text=row.get("chunk_text") or "",
        retrieval_score=None,
        retrieval_method="section_expansion",
        table_html=row.get("table_html") or None,
        has_table=bool(row.get("has_table")),
        evidence_role="section_expansion",
    )


def expand_sections(
    evidence: list[EvidenceItem],
    chunk_fetcher: Optional[Callable[..., list[dict]]],
    max_added: int = MAX_EXPANSION_CHUNKS,
) -> list[EvidenceItem]:
    """Fetch the remaining chunks of each relevant chunk's section (+/- the
    adjacent chunk_index neighbours) so complete provisions and multi-page
    tables enter the evidence.  Failures are logged and skipped (expansion
    is an enhancement, never a hard dependency).
    """
    if chunk_fetcher is None or not evidence:
        return []
    have = {e.chunk_id for e in evidence}
    additions: list[EvidenceItem] = []
    for e in evidence:
        if e.evidence_role == "section_expansion":
            continue
        idx = _chunk_index(e.chunk_id)
        neighbours = [i for i in (idx - 1, idx + 1) if i is not None and i >= 0]
        try:
            rows = chunk_fetcher(
                doc_id=e.doc_id,
                section_path=(e.section_path if _EXPANDABLE_SECTION.match(e.section_path or "") else None),
                chunk_indexes=neighbours or None,
            )
        except Exception as exc:
            logger.warning("section expansion fetch failed for %s: %s", e.chunk_id, exc)
            continue
        for row in sorted(rows, key=lambda r: r.get("chunk_index") or 0):
            cid = str(row.get("chunk_id") or "")
            if not cid or cid in have:
                continue
            have.add(cid)
            additions.append(_result_to_evidence(row))
            if len(additions) >= max_added:
                return additions
    return additions


# ── LangGraph node ─────────────────────────────────────────────

def make_evidence_refinement_node(
    chunk_fetcher: Optional[Callable[..., list[dict]]] = None,
):
    """Factory: create the evidence-refinement LangGraph node.

    Args:
        chunk_fetcher: callable(doc_id=..., section_path=..., chunk_indexes=...)
                   -> list[chunk dicts] — the real retriever's
                   get_chunks_for_expansion in production, None in tests
                   (topic filtering still applies; expansion is skipped).
    """

    def evidence_refinement_node(state: dict) -> dict:
        if state.get("error"):
            return {
                "trace": [
                    make_trace(
                        "evidence_refinement",
                        action="skipped",
                        reason="upstream error — passing through",
                        has_error=True,
                    )
                ]
            }

        evidence = state.get("evidence", []) or []
        query = state.get("user_query", "")
        relevant, focus, dropped = filter_relevant(query, evidence)
        added = expand_sections(relevant, chunk_fetcher) if relevant else []
        refined = relevant + added

        abstain = not refined
        reason = None
        if abstain:
            reason = "No evidence retrieved." if not evidence else (
                "Retrieved chunks were filtered as irrelevant to the question."
            )

        details = {
            "received": len(evidence),
            "relevant": len(relevant),
            "dropped_irrelevant_chunk_ids": [e.chunk_id for e in dropped],
            "expansion_added_chunk_ids": [e.chunk_id for e in added],
            "focus_terms": focus,
        }
        return {
            "evidence": refined,
            "abstain": abstain,
            "abstain_reason": reason,
            "trace": [
                make_trace(
                    "evidence_refinement",
                    action="refined" if refined else "rejected",
                    reason=(
                        f"{len(relevant)}/{len(evidence)} chunks relevant to the "
                        f"question (focus: {', '.join(focus) or 'undetermined'}); "
                        f"{len(added)} added by section expansion"
                        if refined
                        else (reason or "no relevant evidence")
                    ),
                    **details,
                )
            ],
        }

    return evidence_refinement_node
