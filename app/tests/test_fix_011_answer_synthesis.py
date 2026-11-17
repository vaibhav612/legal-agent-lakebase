"""Fix 011 (Milestone 5) regression tests — answer synthesis, evidence
precision, and table-aware retrieval (2026-10-30).

Offline fixtures recreate the exact hosted failure modes verified live on
2026-10-30:

  A. Upjohn insurance question — RRF top-5 was cover page / introduction /
     Entire Agreement / Witnesseth / Section-11-opening only; the amounts
     and limits chunks (136/137) ranked 7th/15th and never entered the
     context; all 5 boilerplate chunks were displayed as sources.
  B. Repeat turn — 'no i couldnt see the answer can you repeat again?'
     matched no fix-008 marker and fell through to corpus-wide hybrid_qa,
     which abstained.
  C. Arizona Assigned Patents — Schedule 1.1(e) is a 6-column table
     spanning 3 PDF pages; the flattened chunk_text glued the cells
     together and the prose LLM template corrupted them.

No LLM endpoint, Lakebase, SQL warehouse, or PDF volume access: a
FakeRetriever (with get_chunks_for_expansion) and FakeGenerator provide
all inputs.
"""

from __future__ import annotations

import os
import sys

_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

from workflow.state import EvidenceItem
from workflow.classifier import RuleBasedIntentClassifier
from workflow.hybrid_qa import make_hybrid_qa_node
from workflow.validation import make_evidence_validation_node
from workflow.answers import (
    ABSTAIN_MESSAGE,
    _cited_indices,
    evidence_to_context_chunks,
    make_final_answer_node,
    select_and_group_sources,
)
from workflow.evidence_refinement import (
    expand_sections,
    filter_relevant,
    make_evidence_refinement_node,
    topic_terms,
)
from workflow.table_answer import (
    build_table_answer,
    detect_table_request,
    html_table_to_markdown,
    merge_table_chunks,
    parse_table_html,
)
from workflow.graph import build_graph


# ── Fixtures modeled on the hosted corpus (2026-10-30 findings) ──────────

UPJOHN_DOC = "upjohn_19940621"
UPJOHN_NAME = (
    "UpjohnInc_19940621_EX-10.1_0000912057-94-009810_EX-10.1_"
    "Manufacturing%20and%20Supply%20Agreement.pdf"
)
ARMSTRONG_DOC = "armstrong_20040510"
ARMSTRONG_NAME = (
    "ArmstrongFlooringInc_20040510_EX-10.2_0000950144-04-006244_EX-10.2_"
    "Intellectual%20Property%20Agreement.pdf"
)

UPJOHN_QUERY = (
    "What are the insurance requirements under the Upjohn manufacturing "
    "and supply agreement?"
)

UPJOHN_TOP5 = [
    # The hosted top-5: boilerplate + ONLY the Section 11 opening.
    {"chunk_id": f"{UPJOHN_DOC}_0", "chunk_text": (
        "MANUFACTURING AND SUPPLY AGREEMENT between The Upjohn Company "
        "and a Supplier, dated as of June 21, 1994. Exhibit 10.1."),
     "doc_id": UPJOHN_DOC, "doc_name": UPJOHN_NAME, "chunk_index": 0,
     "section_path": "", "page_start": 0, "page_end": 0,
     "rrf_score": 0.032, "table_html": None, "has_table": False},
    {"chunk_id": f"{UPJOHN_DOC}_6", "chunk_text": (
        "This Manufacturing and Supply Agreement is entered into by and "
        "between the parties for the supply of pharmaceutical products."),
     "doc_id": UPJOHN_DOC, "doc_name": UPJOHN_NAME, "chunk_index": 6,
     "section_path": "MANUFACTURING AND SUPPLY AGREEMENT", "page_start": 1, "page_end": 1,
     "rrf_score": 0.030, "table_html": None, "has_table": False},
    {"chunk_id": f"{UPJOHN_DOC}_8", "chunk_text": (
        "W I T N E S S E T H: WHEREAS the parties desire to enter into this "
        "agreement..."),
     "doc_id": UPJOHN_DOC, "doc_name": UPJOHN_NAME, "chunk_index": 8,
     "section_path": "W I T N E S S E T H:", "page_start": 1, "page_end": 2,
     "rrf_score": 0.028, "table_html": None, "has_table": False},
    {"chunk_id": f"{UPJOHN_DOC}_130", "chunk_text": (
        "17.15 Entire Agreement. This agreement constitutes the entire "
        "agreement between the parties regarding the subject matter hereof."),
     "doc_id": UPJOHN_DOC, "doc_name": UPJOHN_NAME, "chunk_index": 130,
     "section_path": "17.15 Entire Agreement", "page_start": 14, "page_end": 14,
     "rrf_score": 0.026, "table_html": None, "has_table": False},
    {"chunk_id": f"{UPJOHN_DOC}_135", "chunk_text": (
        "11. Insurance. The Supplier shall maintain, at its own expense, "
        "insurance coverage with a responsible insurance company or "
        "companies..."),
     "doc_id": UPJOHN_DOC, "doc_name": UPJOHN_NAME, "chunk_index": 135,
     "section_path": "11. Insurance.", "page_start": 15, "page_end": 15,
     "rrf_score": 0.024, "table_html": None, "has_table": False},
]

UPJOHN_EXPANSION = [
    {"chunk_id": f"{UPJOHN_DOC}_136", "chunk_text": (
        "Amounts and Limits. The insurance coverage shall be not less than: "
        "employer's liability $1,000,000; commercial general liability "
        "$2,000,000 per occurrence; automobile liability $2,000,000 "
        "combined single limit."),
     "doc_id": UPJOHN_DOC, "doc_name": UPJOHN_NAME, "chunk_index": 136,
     "section_path": "11. Insurance.", "page_start": 15, "page_end": 16,
     "table_html": None, "has_table": False},
    {"chunk_id": f"{UPJOHN_DOC}_137", "chunk_text": (
        "The Supplier shall name the Upjohn Company as an additional "
        "insured and shall provide a waiver of subrogation endorsement."),
     "doc_id": UPJOHN_DOC, "doc_name": UPJOHN_NAME, "chunk_index": 137,
     "section_path": "11. Insurance.", "page_start": 16, "page_end": 16,
     "table_html": None, "has_table": False},
]

ARIZONA_SECTION = "Schedule 1.1(e) - Arizona Assigned Patents"
_TBL_COLS = ["Patent No.", "Title of Invention", "Inventor(s)", "Country", "Application No.", "Status"]


def _tbl(rows):
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in _TBL_COLS) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table>{head}{body}</table>"


ARIZONA_TABLE_CHUNKS = [
    {"chunk_id": f"{ARMSTRONG_DOC}_27", "chunk_text": "Schedule 1.1(e) Arizona Assigned Patents (flattened row 1 row 2)",
     "doc_id": ARMSTRONG_DOC, "doc_name": ARMSTRONG_NAME, "chunk_index": 27,
     "section_path": ARIZONA_SECTION, "page_start": 18, "page_end": 18,
     "rrf_score": 0.033,
     "table_html": _tbl([
         ["US 6,123,456", "Friction liner", "Smith", "US", "09/111,111", "Granted"],
         ["US 6,222,222", "Floor tile", "Jones; Lee", "US", "09/222,222", "Granted"],
     ]), "has_table": True},
    {"chunk_id": f"{ARMSTRONG_DOC}_28", "chunk_text": "(continuation) row 3 row 4",
     "doc_id": ARMSTRONG_DOC, "doc_name": ARMSTRONG_NAME, "chunk_index": 28,
     "section_path": ARIZONA_SECTION, "page_start": 19, "page_end": 19,
     "rrf_score": 0.031,
     "table_html": _tbl([
         ["US 6,333,333", "Backing material", "Patel", "US", "09/333,333", "Granted"],
         ["US 6,444,444", "Adhesive", "Kim", "US", "09/444,444", "Granted"],
     ]), "has_table": True},
    {"chunk_id": f"{ARMSTRONG_DOC}_29", "chunk_text": "(continuation) row 5",
     "doc_id": ARMSTRONG_DOC, "doc_name": ARMSTRONG_NAME, "chunk_index": 29,
     "section_path": ARIZONA_SECTION, "page_start": 20, "page_end": 20,
     "rrf_score": 0.029,
     # Unpublished patent: two cells are EMPTY in the source table.
     "table_html": _tbl([
         ["(pending)", "Floor covering", "Garcia", "US", "", ""],
     ]), "has_table": True},
]

ARIZONA_QUERY = "Can you show me the Arizona assigned patents?"


class FakeRetriever:
    """Fake LakebaseRetriever: search() + get_chunks_for_expansion()."""

    def __init__(self, results=None, expansion_corpus=None, error=None):
        self.results = list(results or [])
        self.expansion_corpus = list(expansion_corpus or [])
        self.error = error
        self.calls = []
        self.expansion_calls = []

    def search(self, query, num_results=5, doc_ids=None):
        self.calls.append({"query": query, "num_results": num_results, "doc_ids": doc_ids})
        if self.error is not None:
            raise self.error
        return self.results[:num_results]

    def get_chunks_for_expansion(self, doc_id=None, section_path=None, chunk_indexes=None):
        self.expansion_calls.append({"doc_id": doc_id, "section_path": section_path,
                                     "chunk_indexes": list(chunk_indexes or [])})
        rows = []
        for row in self.expansion_corpus:
            if row["doc_id"] != doc_id:
                continue
            if section_path and row["section_path"] == section_path:
                rows.append(dict(row))
                continue
            if chunk_indexes and row["chunk_index"] in [int(i) for i in chunk_indexes]:
                rows.append(dict(row))
        rows.sort(key=lambda r: r["chunk_index"])
        return rows


class FakeGenerator:
    def __init__(self, answer=None):
        self.answer = answer or "LLM answer grounded in evidence"
        self.calls = []

    def __call__(self, query, context_chunks):
        self.calls.append({"query": query, "chunks": list(context_chunks)})
        return self.answer


def _evidence(row, role="retrieved"):
    return EvidenceItem(
        doc_id=row["doc_id"],
        doc_name=row["doc_name"],
        chunk_id=row["chunk_id"],
        section_path=row.get("section_path") or None,
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        supporting_text=row.get("chunk_text") or "",
        retrieval_score=row.get("rrf_score"),
        retrieval_method="hybrid" if role == "retrieved" else "section_expansion",
        table_html=row.get("table_html") or None,
        has_table=bool(row.get("has_table")),
        evidence_role=role,
    )


# ── A. Topic filtering (evidence_refinement) ──────────────────────────────

def test_topic_terms_exclude_document_identifying_words():
    focus = topic_terms(UPJOHN_QUERY, [UPJOHN_NAME])
    assert "insurance" in focus
    assert "requirements" in focus
    # Words that merely NAME the document never become topic terms.
    assert "upjohn" not in focus
    assert "manufacturing" not in focus
    assert "supply" not in focus


def test_filter_relevant_upjohn_drops_boilerplate():
    evidence = [_evidence(r) for r in UPJOHN_TOP5]
    relevant, focus, dropped = filter_relevant(UPJOHN_QUERY, evidence)
    assert [e.chunk_id for e in relevant] == [f"{UPJOHN_DOC}_135"]
    assert set(focus) >= {"insurance", "requirements"}
    dropped_ids = {e.chunk_id for e in dropped}
    assert dropped_ids == {f"{UPJOHN_DOC}_0", f"{UPJOHN_DOC}_6", f"{UPJOHN_DOC}_8",
                          f"{UPJOHN_DOC}_130"}


def test_filter_relevant_honest_fallback_when_no_topic_terms():
    # Every content word names the document → cannot judge relevance → keep all.
    evidence = [_evidence(r) for r in UPJOHN_TOP5]
    query = "Tell me about the Upjohn manufacturing and supply agreement"
    relevant, focus, dropped = filter_relevant(query, evidence)
    assert len(relevant) == len(evidence)
    assert focus == []
    assert dropped == []


def test_filter_relevant_best_chunk_always_survives():
    evidence = [_evidence(r) for r in UPJOHN_TOP5]
    # Degenerate case: relative rule can never empty a non-empty list.
    relevant, _, _ = filter_relevant("insurance requirements insurance", evidence)
    assert 1 <= len(relevant) <= len(evidence)
    assert f"{UPJOHN_DOC}_135" in [e.chunk_id for e in relevant]


# ── B. Section / adjacent expansion ────────────────────────────────────────

def test_expand_sections_completes_the_provision():
    kept = [_evidence(UPJOHN_TOP5[4])]  # Section 11 opening only
    retriever = FakeRetriever(expansion_corpus=UPJOHN_TOP5 + UPJOHN_EXPANSION)
    added = expand_sections(kept, retriever.get_chunks_for_expansion)
    added_ids = [e.chunk_id for e in added]
    assert f"{UPJOHN_DOC}_136" in added_ids
    assert f"{UPJOHN_DOC}_137" in added_ids
    for e in added:
        assert e.evidence_role == "section_expansion"
        assert e.retrieval_method == "section_expansion"
    # The amounts/limits text is IN the evidence after expansion.
    joined = " ".join(e.supporting_text for e in kept + added)
    assert "$1,000,000" in joined
    assert "$2,000,000 per occurrence" in joined


def test_expand_sections_never_expands_title_or_recital_sections():
    # A title-section chunk must expand by ADJACENT INDEX only (section_path=None).
    cover = _evidence(UPJOHN_TOP5[0])
    retriever = FakeRetriever(expansion_corpus=UPJOHN_TOP5 + UPJOHN_EXPANSION)
    expand_sections([cover], retriever.get_chunks_for_expansion)
    call = retriever.expansion_calls[0]
    assert call["section_path"] is None
    assert call["chunk_indexes"] in ([0, 1], [1])  # neighbours of index 0


def test_evidence_refinement_node_trace_exposes_counts():
    node = make_evidence_refinement_node(
        FakeRetriever(expansion_corpus=UPJOHN_TOP5 + UPJOHN_EXPANSION).get_chunks_for_expansion
    )
    state = {"user_query": UPJOHN_QUERY, "evidence": [_evidence(r) for r in UPJOHN_TOP5]}
    out = node(state)
    assert out["abstain"] is False
    assert out["abstain_reason"] is None
    refined_ids = [e.chunk_id for e in out["evidence"]]
    assert f"{UPJOHN_DOC}_135" in refined_ids
    assert f"{UPJOHN_DOC}_136" in refined_ids
    assert f"{UPJOHN_DOC}_137" in refined_ids
    assert f"{UPJOHN_DOC}_0" not in refined_ids
    trace = out["trace"][0]
    assert trace["node"] == "evidence_refinement"
    assert trace["details"]["received"] == 5
    assert trace["details"]["relevant"] == 1
    assert len(trace["details"]["expansion_added_chunk_ids"]) == 2
    assert len(trace["details"]["dropped_irrelevant_chunk_ids"]) == 4


# ── C. Table-aware answers ───────────────────────────────────────────────

def test_detect_table_request():
    assert detect_table_request(ARIZONA_QUERY) is True
    assert detect_table_request("Show me the table of Arizona assigned patents") is True
    assert detect_table_request("list the trademarks in schedule 1.2") is True
    # Prose questions must NEVER take the table path.
    assert detect_table_request(UPJOHN_QUERY) is False
    assert detect_table_request("What is the governing law of the agreement?") is False


def test_parse_table_html_preserves_exact_cells_including_empty():
    parsed = parse_table_html(ARIZONA_TABLE_CHUNKS[2]["table_html"])
    assert parsed["columns"] == _TBL_COLS
    assert parsed["rows"] == [["(pending)", "Floor covering", "Garcia", "US", "", ""]]


def test_html_table_to_markdown_pipe_table():
    md = html_table_to_markdown(ARIZONA_TABLE_CHUNKS[0]["table_html"])
    lines = md.splitlines()
    assert lines[0].startswith("| Patent No. |")
    assert lines[1].startswith("|---")
    assert "| US 6,123,456 | Friction liner | Smith | US | 09/111,111 | Granted |" in lines


def test_merge_table_chunks_three_page_continuation():
    merged = merge_table_chunks([_evidence(r) for r in ARIZONA_TABLE_CHUNKS])
    assert merged["columns"] == _TBL_COLS
    assert len(merged["rows"]) == 5  # 2 + 2 + 1, repeated headers dropped
    # Exact cell values, never coerced — identifiers keep their format.
    assert merged["rows"][0][0] == "US 6,123,456"
    assert merged["rows"][4] == ["(pending)", "Floor covering", "Garcia", "US", "", ""]
    assert merged["page_start"] == 18
    assert merged["page_end"] == 20
    assert merged["chunk_ids"] == [f"{ARMSTRONG_DOC}_27", f"{ARMSTRONG_DOC}_28", f"{ARMSTRONG_DOC}_29"]
    assert merged["row_count"] == 5
    assert merged["continuation_note"] is not None


def test_build_table_answer_deterministic():
    payload = build_table_answer(ARIZONA_QUERY, [_evidence(r) for r in ARIZONA_TABLE_CHUNKS])
    assert payload is not None
    text = payload["answer_text"]
    assert ARIZONA_SECTION in text
    assert "5 rows" in text
    assert "19\u201321" in text  # physical PDF pages (0-indexed 18..20 → 19..21)
    assert payload["table_result"]["row_count"] == 5
    assert len(payload["source_refs"]) == 1
    src = payload["source_refs"][0]
    assert src.page_start == 18 and src.page_end == 20


def test_evidence_to_context_chunks_carries_table_markdown():
    chunks = evidence_to_context_chunks([_evidence(r) for r in ARIZONA_TABLE_CHUNKS])
    assert chunks[0]["table_markdown"].startswith("| Patent No. |")
    # Prose chunks carry no table_markdown key.
    prose = evidence_to_context_chunks([_evidence(UPJOHN_TOP5[4])])
    assert "table_markdown" not in prose[0]


# ── D. Claim-cited, grouped source selection ──────────────────────────────

def _grouping_evidence():
    rows = [
        dict(UPJOHN_TOP5[4], chunk_id=f"{UPJOHN_DOC}_135", section_path="11. Insurance.", page_start=15, page_end=15),
        dict(UPJOHN_TOP5[4], chunk_id=f"{UPJOHN_DOC}_136", section_path="11. Insurance.", page_start=15, page_end=16),
        dict(UPJOHN_TOP5[4], chunk_id=f"{UPJOHN_DOC}_137", section_path="11. Insurance.", page_start=16, page_end=16),
        dict(UPJOHN_TOP5[3], chunk_id=f"{UPJOHN_DOC}_130", section_path="17.15 Entire Agreement", page_start=14, page_end=14),
        dict(ARIZONA_TABLE_CHUNKS[0], chunk_id=f"{ARMSTRONG_DOC}_27"),
    ]
    return [_evidence(r) for r in rows]


def test_cited_indices_extracts_sorted_numbers():
    assert _cited_indices("a [2] and [1] plus [10]") == [1, 2, 10]
    assert _cited_indices("") == []


def test_select_and_group_sources_cited_only_and_merged():
    evidence = _grouping_evidence()
    # Answer cites positions 1,2,3 (Section 11 group) only.
    answer = "Requirements [1] with limits [2] and endorsements [3]."
    refs = select_and_group_sources(evidence, answer)
    assert len(refs) == 1  # 3 cited chunks → ONE grouped Section-11 source
    ref = refs[0]
    assert ref.section_path == "11. Insurance."
    assert ref.page_start == 15 and ref.page_end == 16  # merged range
    assert ref.chunk_id == f"{UPJOHN_DOC}_135"
    assert "insurance" in ref.excerpt.lower()


def test_select_and_group_sources_fallback_without_citations():
    evidence = _grouping_evidence()
    refs = select_and_group_sources(evidence, answer=None)
    # No citations → every group is eligible: 11. Insurance, 17.15, Schedule 1.1(e).
    sections = [r.section_path for r in refs]
    assert sections == ["11. Insurance.", "17.15 Entire Agreement", ARIZONA_SECTION]


def test_select_and_group_sources_caps_three_docs():
    rows = []
    for d in ("docA", "docB", "docC", "docD"):
        rows.append(dict(UPJOHN_TOP5[4], chunk_id=f"{d}_1", doc_id=d, doc_name=f"{d}.pdf"))
    evidence = [_evidence(r) for r in rows]
    refs = select_and_group_sources(evidence, "x [1] y [2] z [3] w [4]")
    assert len(refs) == 3  # MAX_SOURCE_DOCS cap


# ── E. Repeat + follow-up turns (classifier) ──────────────────────────────

def _classifier():
    return RuleBasedIntentClassifier()


def _history(prev_query, prev_answer):
    return [
        {"role": "user", "content": prev_query},
        {"role": "assistant", "content": prev_answer},
    ]


def test_hosted_repeat_phrasing_repeats_previous_answer():
    prev = (
        "**Direct answer** — the Supplier must maintain insurance.\n"
        "**Key requirements** — employer's liability $1,000,000 [1]."
    )
    intent = _classifier().classify(
        "no i couldnt see the answer can you repeat again?",
        _history(UPJOHN_QUERY, prev),
    )
    assert intent.label == "unsupported"
    assert intent.clarification_needed is not None
    assert "Here is my previous answer" in intent.clarification_needed
    assert "$1,000,000" in intent.clarification_needed


def test_repeat_with_prior_error_is_honest():
    intent = _classifier().classify(
        "can you repeat that answer again?",
        _history(UPJOHN_QUERY, "Error: Hybrid retrieval failed: connection refused"),
    )
    msg = intent.clarification_needed
    assert msg is not None
    assert "error message rather than an answer" in msg
    assert "connection refused" in msg
    assert not msg.startswith("Here is my previous answer")


def test_repeat_with_prior_abstention_is_honest():
    intent = _classifier().classify(
        "couldn't see the answer, repeat again",
        _history(UPJOHN_QUERY, ABSTAIN_MESSAGE),
    )
    msg = intent.clarification_needed
    assert msg is not None
    assert "no answer to repeat" in msg
    assert not msg.startswith("Here is my previous answer")


def test_followup_keeps_document_scope_and_combines_query():
    prev_intent = _classifier().classify(UPJOHN_QUERY, None)
    assert prev_intent.label == "hybrid_qa"
    assert prev_intent.target_doc_names  # the Upjohn question names its document

    intent = _classifier().classify(
        "what about the limits?",
        _history(UPJOHN_QUERY, "answer text"),
    )
    assert intent.label == "hybrid_qa"
    # The document scope is CARRIED OVER from the previous turn.
    assert intent.target_doc_names == prev_intent.target_doc_names
    # The search query combines the previous question with the follow-up.
    assert intent.search_query is not None
    assert "insurance" in intent.search_query.lower()
    assert "limits" in intent.search_query.lower()


def test_breakdown_phrasing_is_a_followup():
    intent = _classifier().classify(
        "give me the breakdown of the limits",
        _history(UPJOHN_QUERY, "answer text"),
    )
    assert intent.label == "hybrid_qa"
    assert intent.search_query is not None
    assert "insurance" in intent.search_query.lower()


def test_normal_question_not_hijacked_by_new_markers():
    intent = _classifier().classify(
        "What insurance requirements does the Upjohn manufacturing and supply agreement impose?",
        _history(UPJOHN_QUERY, "previous answer"),
    )
    assert intent.label == "hybrid_qa"
    assert intent.clarification_needed is None


# ── F. Hybrid node uses the combined search query ──────────────────────────

def test_hybrid_qa_node_uses_intent_search_query():
    retriever = FakeRetriever(results=UPJOHN_TOP5)
    node = make_hybrid_qa_node(retriever)
    intent = _classifier().classify("what about the limits?", _history(UPJOHN_QUERY, "answer"))
    state = {"user_query": "what about the limits?", "intent": intent}
    out = node(state)
    assert retriever.calls[0]["query"] == intent.search_query
    assert out["trace"][0]["details"]["doc_scope"] == list(intent.target_doc_names)


# ── G. Validation preserves the refinement's abstain reason ───────────────

def test_validation_keeps_upstream_abstain_reason():
    node = make_evidence_validation_node()
    state = {
        "user_query": UPJOHN_QUERY,
        "evidence": [],
        "abstain_reason": "Retrieved chunks were filtered as irrelevant to the question.",
    }
    out = node(state)
    assert out["abstain"] is True
    assert out["abstain_reason"] == "Retrieved chunks were filtered as irrelevant to the question."


def test_validation_default_reason_unchanged_without_upstream():
    node = make_evidence_validation_node()
    out = node({"user_query": UPJOHN_QUERY, "evidence": []})
    assert out["abstain_reason"] == "No evidence retrieved."


# ── H. final_answer: table branch + cited sources ──────────────────────────

def test_final_answer_table_branch_deterministic():
    generator = FakeGenerator(answer="SHOULD NOT BE CALLED")
    node = make_final_answer_node(generator)
    state = {
        "user_query": ARIZONA_QUERY,
        "evidence": [_evidence(r) for r in ARIZONA_TABLE_CHUNKS],
        "abstain": False,
        "error": None,
        "trace": [],
    }
    out = node(state)
    assert generator.calls == []  # cell values never pass through the LLM
    assert out["table_result"] is not None
    assert out["table_result"]["row_count"] == 5
    assert ARIZONA_SECTION in out["answer"]
    assert out["source_refs"][0].page_end == 20
    trace = out["trace"][0]
    assert trace["action"] == "table_answer"
    assert trace["details"]["table_answer"] is True


def test_final_answer_cited_grouped_sources():
    answer = (
        "**Direct answer** — the Supplier must maintain insurance [1].\n"
        "**Key requirements** — employer's liability $1,000,000 [2]; "
        "additional insured + waiver of subrogation [3]."
    )
    generator = FakeGenerator(answer=answer)
    node = make_final_answer_node(generator)
    evidence = _grouping_evidence()
    state = {
        "user_query": UPJOHN_QUERY,
        "evidence": evidence,
        "abstain": False,
        "error": None,
        "trace": [],
    }
    out = node(state)
    assert out["answer"] == answer
    # 5 evidence chunks, 3 cited positions in ONE section → ONE source ref.
    assert len(out["source_refs"]) == 1
    assert out["source_refs"][0].section_path == "11. Insurance."
    assert out["table_result"] is None
    trace = out["trace"][0]
    assert trace["details"]["cited_indices"] == [1, 2, 3]
    assert trace["details"]["num_sources"] == 1


# ── I. Graph-level integration (offline) ─────────────────────────────────

def test_graph_upjohn_flow_expands_and_cites_one_group():
    retriever = FakeRetriever(
        results=UPJOHN_TOP5,
        expansion_corpus=UPJOHN_TOP5 + UPJOHN_EXPANSION,
    )
    answer = (
        "**Direct answer** — the Supplier must maintain insurance [1].\n"
        "**Key requirements** — employer's liability $1,000,000 [2]; "
        "additional insured + waiver of subrogation [3]."
    )
    generator = FakeGenerator(answer=answer)
    graph = build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        answer_generator=generator,
    )
    state = graph.invoke({"user_query": UPJOHN_QUERY, "conversation_history": []})

    # Expansion brought the amounts/limits chunks into the LLM context.
    ctx_text = "\n".join(str(c.get("chunk_text")) for c in generator.calls[0]["chunks"])
    assert "$1,000,000" in ctx_text
    assert "$2,000,000 per occurrence" in ctx_text

    # Boilerplate never reached the generator.
    assert "W I T N E S S E T H" not in ctx_text
    assert "Entire Agreement. This agreement constitutes" not in ctx_text

    # One grouped Section-11 source instead of 5 chunk sources.
    assert len(state["source_refs"]) == 1
    assert state["source_refs"][0].section_path == "11. Insurance."

    # Trace exposes the true decision chain.
    nodes = [t["node"] for t in state["trace"]]
    assert "evidence_refinement" in nodes
    refinement = next(t for t in state["trace"] if t["node"] == "evidence_refinement")
    assert refinement["details"]["received"] == 5
    assert refinement["details"]["relevant"] == 1


def test_graph_arizona_table_flow():
    retriever = FakeRetriever(
        results=ARIZONA_TABLE_CHUNKS + [UPJOHN_TOP5[0]],
        expansion_corpus=ARIZONA_TABLE_CHUNKS + UPJOHN_TOP5,
    )
    generator = FakeGenerator(answer="SHOULD NOT BE CALLED")
    graph = build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        answer_generator=generator,
    )
    state = graph.invoke({"user_query": ARIZONA_QUERY, "conversation_history": []})

    assert generator.calls == []  # deterministic table answer, no LLM cells
    assert state["table_result"] is not None
    assert state["table_result"]["row_count"] == 5
    assert state["table_result"]["columns"] == _TBL_COLS
    assert "US 6,123,456" in str(state["table_result"]["rows"])
    assert state["answer"]
    trace = next(t for t in state["trace"] if t["node"] == "final_answer")
    assert trace["action"] == "table_answer"


def test_graph_repeat_turn_no_retrieval():
    retriever = FakeRetriever(results=UPJOHN_TOP5)
    prev_answer = (
        "**Direct answer** — the Supplier must maintain insurance.\n"
        "**Key requirements** — employer's liability $1,000,000."
    )
    generator = FakeGenerator(answer=prev_answer)
    graph = build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        answer_generator=generator,
    )
    history = [
        {"role": "user", "content": UPJOHN_QUERY},
        {"role": "assistant", "content": prev_answer},
    ]
    state = graph.invoke({
        "user_query": "no i couldnt see the answer can you repeat again?",
        "conversation_history": history,
    })
    # The repeat turn performs NO retrieval and NO generation — it repeats.
    assert retriever.calls == []
    assert generator.calls == []
    assert "Here is my previous answer" in state["answer"]
    assert "$1,000,000" in state["answer"]
