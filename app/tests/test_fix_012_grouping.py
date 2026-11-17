"""Fix 012 (2026-10-31) offline regression tests - Defects B and D.

Grouping and citation integrity (spec 5.A cases 1-8) plus the follow-up
planner scope rules (spec 5.D cases 22-23).  All tests are offline: no
Lakebase, no LLM, no Streamlit (UI behavior is covered by
test_fix_012_ui_apptest.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.answers import (  # noqa: E402
    build_source_groups,
    group_refs_by_doc,
    make_final_answer_node,
)
from workflow.classifier import (  # noqa: E402
    RuleBasedIntentClassifier,
    make_query_planner_node,
)
from workflow.state import EvidenceItem, SourceRef  # noqa: E402


def ev(doc_id, doc_name, chunk_id, section, p_start, p_end, text="passage text"):
    return EvidenceItem(
        doc_id=doc_id,
        doc_name=doc_name,
        chunk_id=chunk_id,
        section_path=section,
        page_start=p_start,
        page_end=p_end,
        supporting_text=text,
    )


UPJOHN = "UpjohnInc_20200121_10-12G_EX-2.6_11948692_EX-2.6_Manufacturing%20Agreement_%20Supply%20Agreement.pdf"
ARMSTRONG = "ArmstrongFlooringInc_20040510_EX-10.2_1190017387_Consignment%20Agreement_16948692_EX-10.2_IP%20agreement.pdf"


# -- 5.A Grouping and citation integrity ----------------------------------


def test_case1_one_doc_three_chunks_one_top_level_group():
    """Three supporting chunks from one canonical document produce ONE
    document source (case 1) and each passage stays individually
    identifiable (case 2)."""
    evidence = [
        ev("d1", UPJOHN, "c1", "11. Insurance.", 67, 68, "general insurance"),
        ev("d1", UPJOHN, "c2", "11. Insurance.", 68, 69, "amounts and limits"),
        ev("d1", UPJOHN, "c3", "11. Insurance.", 69, 69, "waiver of subrogation"),
    ]
    groups = build_source_groups(evidence, answer="X [1], [2], [3].")
    assert len(groups) == 1
    g = groups[0]
    assert g.doc_id == "d1"
    assert [r.chunk_id for r in g.evidence] == ["c1", "c2", "c3"]
    # a citation beyond the evidence range is ignored (never fabricated):
    g2 = build_source_groups(evidence, answer="X [1], [9].")
    assert [r.chunk_id for r in g2[0].evidence] == ["c1"]
    assert g.page_start == 67 and g.page_end == 69


def test_case3_section_identity_preserved_within_document():
    """Evidence from different sections of the same document keeps its
    section identity per item; the group header carries no single section
    (it spans sections)."""
    evidence = [
        ev("d1", UPJOHN, "c1", "11. Insurance.", 67, 68),
        ev("d1", UPJOHN, "c2", "12. Term.", 70, 70),
    ]
    groups = build_source_groups(evidence)
    assert len(groups) == 1
    assert groups[0].section_path is None  # spans >1 section
    secs = [r.section_path for r in groups[0].evidence]
    assert secs == ["11. Insurance.", "12. Term."]


def test_case4_second_document_is_separate_group():
    """Evidence from a different document creates a separate top-level
    source group (case 4), in first-appearance order."""
    evidence = [
        ev("d1", UPJOHN, "c1", "11. Insurance.", 67, 68),
        ev("d2", ARMSTRONG, "k9", "5. Licenses.", 3, 4),
        ev("d1", UPJOHN, "c2", "11. Insurance.", 68, 69),
    ]
    groups = build_source_groups(evidence)
    assert [g.doc_id for g in groups] == ["d1", "d2"]
    assert len(groups[0].evidence) == 2 and len(groups[1].evidence) == 1


def test_case5_reordered_evidence_keeps_canonical_grouping():
    """Reordered retrieval results do not change the canonical document
    grouping (case 5) - grouping is by doc_id, never by order/text."""
    a = [ev("d1", UPJOHN, "c1", "11.", 67, 68), ev("d2", ARMSTRONG, "k1", "5.", 3, 4)]
    b = list(reversed(a))
    ga, gb = build_source_groups(a), build_source_groups(b)
    assert {g.doc_id for g in ga} == {g.doc_id for g in gb} == {"d1", "d2"}


def test_case6_inline_citations_resolve_to_evidence_items():
    """Inline [n] citations map to the cited context positions (case 6):
    [1] and [3] are retained with evidence_ref 1 and 3; [2] is not shown
    unless cited; no conflation of [n] with chunk ids."""
    evidence = [
        ev("d1", UPJOHN, "c1", "11. Insurance.", 67, 68),
        ev("d1", UPJOHN, "c2", "12. Term.", 70, 70),
        ev("d1", UPJOHN, "c3", "13. Notices.", 71, 72),
    ]
    groups = build_source_groups(evidence, answer="requires X [1] and Y [3].")
    refs = {r.evidence_ref: r.chunk_id for r in groups[0].evidence}
    assert refs == {1: "c1", 3: "c3"}
    assert groups[0].evidence[0].chunk_id != groups[0].evidence[0].evidence_ref


def test_case7_counts_distinguish_docs_and_passages():
    """Document-group count and evidence-passage count are separate
    quantities (case 7): 1 document, 3 passages - not Sources (3)."""
    evidence = [
        ev("d1", UPJOHN, "c1", "11.", 67, 68),
        ev("d1", UPJOHN, "c2", "11.", 68, 69),
        ev("d1", UPJOHN, "c3", "11.", 69, 69),
    ]
    groups = build_source_groups(evidence, answer="[1] [2] [3]")
    n_docs, n_ev = len(groups), sum(len(g.evidence) for g in groups)
    assert (n_docs, n_ev) == (1, 3)


def test_case8_missing_title_never_section_only():
    """A missing document title produces a SAFE fallback, never an empty
    title, and never the section number standing in for the title (case 8).
    build_source_groups tolerates missing doc_name/doc_id; the render layer
    substitutes 'Document <id>...' / 'Untitled document' (AppTest file)."""
    evidence = [
        EvidenceItem(doc_id="", doc_name="", chunk_id="c1",
                     section_path="11. Insurance.", page_start=67, page_end=68,
                     supporting_text="t"),
    ]
    groups = build_source_groups(evidence)
    assert len(groups) == 1
    assert groups[0].doc_name == ""  # no crash; render falls back explicitly
    assert groups[0].evidence[0].section_path == "11. Insurance."


def test_table_answer_groups_table_evidence():
    """The table branch (fix 011) groups its table chunks per document with
    every chunk individually addressable (no [n] markers in table text -
    honest fallback keeps all)."""
    tbl = "<table><tr><th>Patent</th></tr><tr><td>US 6,123,456</td></tr></table>"
    evidence = [
        ev("dA", ARMSTRONG, "t1", "Schedule 1.1(e)", 18, 19),
        ev("dA", ARMSTRONG, "t2", "Schedule 1.1(e)", 19, 20),
        ev("dA", ARMSTRONG, "t3", "Schedule 1.1(e)", 20, 21),
    ]
    for e in evidence:
        e.table_html = tbl
        e.has_table = True
    node = make_final_answer_node()  # no generator -> table branch
    out = node({
        "user_query": "Show me the Arizona assigned patents table",
        "evidence": evidence,
    })
    assert out.get("table_result")
    groups = out.get("source_groups") or []
    assert len(groups) == 1
    assert [r.chunk_id for r in groups[0].evidence] == ["t1", "t2", "t3"]
    assert groups[0].page_start == 18 and groups[0].page_end == 21


def test_group_refs_by_doc_legacy_refs():
    """Legacy messages with only source_refs are grouped by doc_id on the
    fly (order/pages/sections preserved)."""
    refs = [
        SourceRef(doc_id="d1", doc_name=UPJOHN, section_path="11.", page_start=5, page_end=6, chunk_id="c1"),
        SourceRef(doc_id="d2", doc_name=ARMSTRONG, section_path="5.", page_start=1, page_end=2, chunk_id="k1"),
        SourceRef(doc_id="d1", doc_name=UPJOHN, section_path="12.", page_start=7, page_end=7, chunk_id="c2"),
    ]
    groups = group_refs_by_doc(refs)
    assert [g.doc_id for g in groups] == ["d1", "d2"]
    assert [r.chunk_id for r in groups[0].evidence] == ["c1", "c2"]
    assert groups[0].page_start == 5 and groups[0].page_end == 7


# -- 5.D Conversation context (planner scope rules) ------------------------


def _hist():
    return [
        {"role": "user", "content": "Can you show me the Arizona assigned patents?"},
        {"role": "assistant", "content": "TABLE: assigned patents from the IP agreement"},
    ]


def _planner():
    return make_query_planner_node(RuleBasedIntentClassifier())


def test_case22_followup_scoped_to_prior_answer_doc():
    """'What about the limits?' after a table answer: the previous QUESTION
    names no agreement, so the scope comes from the documents that backed
    the previous ANSWER (prior_answer_docs) - one doc scopes the search."""
    out = _planner()({
        "user_query": "What about the limits?",
        "conversation_history": _hist(),
        "prior_answer_docs": [ARMSTRONG],
    })
    it = out["intent"]
    assert it.label == "hybrid_qa"
    assert it.scope == "specific_docs"
    assert it.target_doc_names == [ARMSTRONG]
    assert it.resolved_from_followup is True
    assert out.get("target_doc_names") == [ARMSTRONG]


def test_case22b_ambiguous_prior_docs_ask_one_clarification():
    """When the previous answer used several documents and the follow-up
    names none, ask ONE focused clarification instead of guessing."""
    out = _planner()({
        "user_query": "What about the limits?",
        "conversation_history": _hist(),
        "prior_answer_docs": [ARMSTRONG, UPJOHN],
    })
    it = out["intent"]
    assert it.label == "unsupported"
    assert it.clarification_needed
    assert "Which one" in it.clarification_needed
    assert ARMSTRONG not in it.clarification_needed  # display names, not raw file names


def test_case22c_no_prior_docs_keeps_legacy_behavior():
    """Without prior_answer_docs the planner behaves exactly as before fix
    012 (corpus-wide hybrid search with the combined query)."""
    out = _planner()({
        "user_query": "What about the limits?",
        "conversation_history": _hist(),
    })
    it = out["intent"]
    assert it.label == "hybrid_qa"
    assert it.scope == "all"
    assert not it.target_doc_names
    assert it.search_query and "patents" in it.search_query


def test_case23_named_agreement_switches_without_prior_leakage():
    """A follow-up that NAMES a new agreement switches to it; prior-answer
    docs never override an explicitly named document."""
    out = _planner()({
        "user_query": "What about the limits in the BellRing Supply Agreement?",
        "conversation_history": _hist(),
        "prior_answer_docs": [ARMSTRONG],
    })
    it = out["intent"]
    assert it.target_doc_names == ["BellRingSupply"]  # extractor output for this phrasing
    assert ARMSTRONG not in it.target_doc_names


def test_prior_docs_never_apply_to_non_followups():
    """A fresh, non-follow-up query never inherits prior-answer scope -
    the fix 012 branch requires resolved_from_followup."""
    out = _planner()({
        "user_query": "What insurance does the Upjohn agreement require?",
        "conversation_history": _hist(),
        "prior_answer_docs": [ARMSTRONG],
    })
    it = out["intent"]
    assert it.label == "hybrid_qa"
    assert it.resolved_from_followup is False
    assert ARMSTRONG not in it.target_doc_names
