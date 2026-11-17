"""Unit tests for Milestone 2B.1 — Hybrid Q&A integrated into LangGraph.

11 test areas covering retrieval mapping, metadata retention, dedup,
abstention, whitespace rejection, context budget, retrieval errors,
validation-before-generation ordering, source-reference capping, and
preservation of the existing safe routes.

No LLM endpoint, Lakebase, SQL warehouse, or PDF volume access: the graph
is built with a FakeRetriever and a FakeGenerator, both with call counters
proving answer generation never happens when evidence validation rejects.
"""

from __future__ import annotations

import os
import sys

# Add the app folder to sys.path so 'workflow' is importable
_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

import pytest

from workflow.state import (
    EvidenceItem,
    MAX_EVIDENCE_CONTEXT_CHARS,
    MAX_SOURCE_DOCS,
    SourceRef,
)
from workflow.classifier import RuleBasedIntentClassifier
from workflow.hybrid_qa import search_result_to_evidence, make_hybrid_qa_node
from workflow.validation import make_evidence_validation_node
from workflow.answers import (
    ABSTAIN_MESSAGE,
    build_source_refs,
    make_final_answer_node,
)
from workflow.graph import build_graph


QA_QUERY = "What are the termination conditions in the Cerence contract?"


# ── Fakes (no network, no LLM, no Lakebase) ──────────────────────────────

class FakeRetriever:
    """Fake stand-in for LakebaseRetriever.search() with call counting."""

    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def search(self, query, num_results=5):
        self.calls.append({"query": query, "num_results": num_results})
        if self.error is not None:
            raise self.error
        return self.results[:num_results]


class FakeGenerator:
    """Fake stand-in for LakebaseRetriever.generate_response() with call counting."""

    def __init__(self, answer="LLM answer grounded in evidence"):
        self.answer = answer
        self.calls = []

    def __call__(self, query, context_chunks):
        self.calls.append({"query": query, "chunks": list(context_chunks)})
        return self.answer


def _result(
    chunk_id="c1",
    doc_id="d1",
    doc_name="Alpha_MSA.pdf",
    text="termination of the agreement upon material breach",
    section="9. Termination",
    page_start=3,
    page_end=4,
    rrf_score=0.033,
    bm25_score=2.5,
    distance=0.12,
):
    """One row shaped exactly like search_hybrid_v2() output."""
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "chunk_index": 0,
        "section_path": section,
        "page_start": page_start,
        "page_end": page_end,
        "distance": distance,
        "bm25_score": bm25_score,
        "bm25_rank": 0,
        "semantic_rank": 0,
        "rrf_score": rrf_score,
        "rrf_rank": 0,
    }


def _make_graph(retriever, generator, num_results=5):
    return build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        answer_generator=generator,
        num_results=num_results,
    )


def _run(graph, query=QA_QUERY):
    return graph.invoke({"user_query": query, "conversation_history": []})


def _trace_nodes(state):
    return [e["node"] for e in state.get("trace", [])]


def _trace_entry(state, node):
    entries = [e for e in state.get("trace", []) if e["node"] == node]
    assert entries, f"no trace entry for node {node!r}"
    return entries[0]


# ── Test 1: retrieval results map correctly to EvidenceItem ────────────────

def test_1_search_result_maps_to_evidence_item():
    """Every search_hybrid_v2 field maps into EvidenceItem correctly."""
    item = search_result_to_evidence(_result())
    assert isinstance(item, EvidenceItem)
    assert item.doc_id == "d1"
    assert item.doc_name == "Alpha_MSA.pdf"
    assert item.chunk_id == "c1"
    assert item.supporting_text == "termination of the agreement upon material breach"
    assert item.retrieval_method == "hybrid"
    # rrf_score preferred as the retrieval score
    assert item.retrieval_score == pytest.approx(0.033)

    # Score fallbacks: bm25_score when no rrf, distance when nothing else
    r = _result(rrf_score=None)
    assert search_result_to_evidence(r).retrieval_score == pytest.approx(2.5)
    r = _result(rrf_score=None, bm25_score=None)
    assert search_result_to_evidence(r).retrieval_score == pytest.approx(0.12)
    r = _result(rrf_score=None, bm25_score=None, distance=None)
    assert search_result_to_evidence(r).retrieval_score is None

    # doc_name fallback chain: doc_name -> file_name -> doc_id -> Unknown
    r = _result(doc_name=None)
    r["file_name"] = "From_BM25_Corpus.pdf"
    assert search_result_to_evidence(r).doc_name == "From_BM25_Corpus.pdf"
    r = _result(doc_name=None)
    assert search_result_to_evidence(r).doc_name == "d1"


def test_1b_node_records_scores_and_scope_in_trace():
    """Channel scores and resolved doc scope are preserved in the trace."""
    retriever = FakeRetriever([_result()])
    generator = FakeGenerator()
    graph = _make_graph(retriever, generator)
    state = _run(graph)
    entry = _trace_entry(state, "hybrid_qa")
    assert entry["action"] == "retrieved"
    scores = entry["details"]["scores"]
    assert scores[0]["rrf_score"] == pytest.approx(0.033)
    assert scores[0]["bm25_score"] == pytest.approx(2.5)
    assert scores[0]["distance"] == pytest.approx(0.12)
    assert scores[0]["semantic_rank"] == 0 and scores[0]["bm25_rank"] == 0
    assert entry["details"]["num_requested"] == 5
    assert entry["details"]["num_returned"] == 1
    # D2 completion (2026-10-29): 'the Cerence contract' is now recognized as
    # a requested document scope in the intent.  This graph has no
    # sql_executor, so name resolution is skipped and retrieval stays
    # corpus-wide — doc_scope records the REQUESTED names, doc_ids stays None.
    assert entry["details"]["doc_scope"] == ["Cerence"]
    assert entry["details"]["doc_ids"] is None


def test_1c_num_results_passed_through():
    """The injected top-k reaches the retriever unchanged."""
    retriever = FakeRetriever([_result(), _result("c2"), _result("c3")])
    graph = _make_graph(retriever, FakeGenerator(), num_results=3)
    _run(graph)
    assert retriever.calls[0]["num_results"] == 3


# ── Test 2: section and page metadata retained end-to-end ─────────────────

def test_2_section_and_page_metadata_retained():
    """Section path and page range survive into evidence AND source refs."""
    retriever = FakeRetriever([
        _result("c1", "d1", "Alpha_MSA.pdf", section="4. Payment", page_start=5, page_end=7),
    ])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    evidence = state["evidence"]
    assert len(evidence) == 1
    assert evidence[0].section_path == "4. Payment"
    assert evidence[0].page_start == 5
    assert evidence[0].page_end == 7

    refs = state["source_refs"]
    assert len(refs) == 1
    assert refs[0].section_path == "4. Payment"
    assert refs[0].page_start == 5
    assert refs[0].page_end == 7
    assert refs[0].doc_name == "Alpha_MSA.pdf"
    assert refs[0].doc_id == "d1"
    assert refs[0].excerpt.startswith("termination of the agreement")


# ── Test 3: duplicate chunks handled ────────────────────────────────────────

def test_3_duplicate_chunk_ids_deduplicated():
    """Repeated chunk_id keeps the best-ranked occurrence; drops recorded."""
    retriever = FakeRetriever([
        _result("c1", text="first (best rank) occurrence"),
        _result("c1", text="duplicate occurrence", rrf_score=0.01),
        _result("c2", text="second unique chunk"),
    ])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    evidence = state["evidence"]
    assert [e.chunk_id for e in evidence] == ["c1", "c2"]
    assert evidence[0].supporting_text == "first (best rank) occurrence"

    # Dedup recorded in the validation trace without losing provenance
    ev = _trace_entry(state, "evidence_validation")
    assert ev["details"]["dropped_duplicate_chunk_ids"] == ["c1"]
    assert ev["details"]["received"] == 3
    assert ev["details"]["kept"] == 2

    # Generation received only the deduplicated evidence
    assert len(generator.calls) == 1
    assert [c["chunk_id"] for c in generator.calls[0]["chunks"]] == ["c1", "c2"]


# ── Test 4: empty retrieval produces abstention, skips generation ─────────

def test_4_empty_retrieval_abstains_and_skips_generation():
    """Zero results → abstain with reason, no answer generated."""
    retriever = FakeRetriever([])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    assert state["abstain"] is True
    assert state["abstain_reason"] == "No evidence retrieved."
    assert state["answer"] == ABSTAIN_MESSAGE
    assert state.get("source_refs") == []
    # Call counter: generation never happened
    assert generator.calls == []
    # Node order still runs the full path
    assert _trace_nodes(state) == [
        "query_planner", "hybrid_qa", "evidence_refinement",
        "evidence_validation", "final_answer",
    ]
    fa = _trace_entry(state, "final_answer")
    assert fa["details"]["abstain"] is True
    assert fa["details"]["generated"] is False


# ── Test 5: empty / whitespace-only evidence rejected ───────────────────────

def test_5a_whitespace_evidence_dropped_but_valid_kept():
    """Whitespace-only chunks are discarded; usable chunks survive."""
    retriever = FakeRetriever([
        _result("c1", text="   \n\t  "),
        _result("c2", text="valid passage of text"),
    ])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    assert [e.chunk_id for e in state["evidence"]] == ["c2"]
    ev = _trace_entry(state, "evidence_validation")
    assert ev["details"]["dropped_empty_text"] == 1
    assert state["abstain"] is False
    assert len(generator.calls) == 1


def test_5b_all_whitespace_evidence_abstains_and_skips_generation():
    """All-whitespace evidence → abstain with a clear reason, no generation."""
    retriever = FakeRetriever([
        _result("c1", text=" "),
        _result("c2", text="\n\t"),
    ])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    assert state["abstain"] is True
    assert "empty or whitespace-only" in state["abstain_reason"]
    assert state["answer"] == ABSTAIN_MESSAGE
    # Call counter: generation never happened
    assert generator.calls == []
    fa = _trace_entry(state, "final_answer")
    assert fa["details"]["generated"] is False


# ── Test 6: context-length limit respected ─────────────────────────────────

def test_6a_context_budget_unit_level():
    """Validation node drops chunks that would exceed the char budget."""
    node = make_evidence_validation_node(max_context_chars=50)
    evidence = [
        EvidenceItem(doc_id="d", doc_name="D", chunk_id="c1", supporting_text="a" * 30),
        EvidenceItem(doc_id="d", doc_name="D", chunk_id="c2", supporting_text="b" * 30),
        EvidenceItem(doc_id="d", doc_name="D", chunk_id="c3", supporting_text="c" * 10),
    ]
    result = node({"evidence": evidence})
    assert [e.chunk_id for e in result["evidence"]] == ["c1", "c3"]
    assert result["abstain"] is False
    details = result["trace"][0]["details"]
    assert details["dropped_context_overflow_chunk_ids"] == ["c2"]
    assert details["total_supporting_chars"] == 40
    assert details["context_limit"] == 50


def test_6b_default_24000_limit_enforced_end_to_end():
    """The existing 24,000-char budget is enforced on the real graph path."""
    big = "x" * 10000
    retriever = FakeRetriever([
        _result("c1", text=big),
        _result("c2", text=big),
        _result("c3", text=big),  # would push total to 30,000 > 24,000
    ])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    evidence = state["evidence"]
    assert [e.chunk_id for e in evidence] == ["c1", "c2"]
    total = sum(len(e.supporting_text) for e in evidence)
    assert total == 20000 <= MAX_EVIDENCE_CONTEXT_CHARS
    ev = _trace_entry(state, "evidence_validation")
    assert ev["details"]["dropped_context_overflow_chunk_ids"] == ["c3"]
    # Generation saw only the in-budget chunks
    assert len(generator.calls[0]["chunks"]) == 2


def test_6c_exact_boundary_included():
    """Evidence landing exactly on the budget boundary is kept."""
    node = make_evidence_validation_node(max_context_chars=100)
    evidence = [
        EvidenceItem(doc_id="d", doc_name="D", chunk_id="c1", supporting_text="a" * 60),
        EvidenceItem(doc_id="d", doc_name="D", chunk_id="c2", supporting_text="b" * 40),
    ]
    result = node({"evidence": evidence})
    assert [e.chunk_id for e in result["evidence"]] == ["c1", "c2"]
    assert result["trace"][0]["details"]["total_supporting_chars"] == 100


# ── Test 7: retrieval error handled safely and recorded ────────────────────

def test_7_retrieval_error_safe_and_recorded():
    """A retriever exception produces an error state, not a fake answer."""
    retriever = FakeRetriever(error=RuntimeError("Lakebase connection refused"))
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    assert "Hybrid retrieval failed" in state["error"]
    assert "Lakebase connection refused" in state["error"]
    assert state["answer"].startswith("Error:")
    assert state.get("evidence") == []

    # Trace records the failure and the skipped validation
    hq = _trace_entry(state, "hybrid_qa")
    assert hq["action"] == "failed"
    assert "Lakebase connection refused" in hq["reason"]
    ev = _trace_entry(state, "evidence_validation")
    assert ev["action"] == "skipped"

    # Call counter: generation never happened; not masked by abstention
    assert generator.calls == []
    assert not state.get("abstain")
    fa = _trace_entry(state, "final_answer")
    assert fa["details"]["has_error"] is True


# ── Test 8: evidence passes validation BEFORE generation ───────────────────

def test_8_validation_precedes_generation_node_order():
    """Valid query: query_planner → hybrid_qa → evidence_validation → final_answer,
    and the generator receives exactly the validated evidence."""
    retriever = FakeRetriever([
        _result("c1"),
        # Fix 011: c2 is now topically relevant to QA_QUERY so the
        # evidence-refinement topic filter keeps it (the old
        # 'governing law provisions' text is irrelevant to a termination
        # question and is correctly filtered out by fix 011).
        _result("c2", text="termination conditions require written notice"),
    ])
    generator = FakeGenerator(answer="grounded answer")
    state = _run(_make_graph(retriever, generator))

    # Node order (trace order is execution order)
    assert _trace_nodes(state) == [
        "query_planner", "hybrid_qa", "evidence_refinement",
        "evidence_validation", "final_answer",
    ]

    # Generation happened exactly once, with the validated evidence
    assert len(generator.calls) == 1
    assert generator.calls[0]["query"] == QA_QUERY
    passed = generator.calls[0]["chunks"]
    assert [c["chunk_id"] for c in passed] == ["c1", "c2"]
    # Chunks are converted back to the retriever's context-chunk shape
    assert passed[0]["chunk_text"] == "termination of the agreement upon material breach"
    assert passed[0]["doc_name"] == "Alpha_MSA.pdf"
    assert passed[0]["section_path"] == "9. Termination"

    # The generated answer is the final answer, grounded in evidence
    assert state["answer"] == "grounded answer"
    assert state.get("error") is None
    assert state["abstain"] is False
    assert retriever.calls[0]["query"] == QA_QUERY


# ── Test 9: source refs deduplicated and capped at 3 documents ─────────────

def test_9a_source_refs_capped_at_three_docs():
    """Five documents retrieved → at most MAX_SOURCE_DOCS distinct docs shown."""
    retriever = FakeRetriever([
        _result("c1", "d1", "Doc1.pdf"),
        _result("c2", "d2", "Doc2.pdf"),
        _result("c3", "d3", "Doc3.pdf"),
        _result("c4", "d4", "Doc4.pdf"),
        _result("c5", "d5", "Doc5.pdf"),
    ])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator))

    refs = state["source_refs"]
    assert all(isinstance(r, SourceRef) for r in refs)
    doc_ids = [r.doc_id for r in refs]
    assert len(set(doc_ids)) == len(doc_ids) <= MAX_SOURCE_DOCS == 3
    # Top-ranked documents win the three slots
    assert doc_ids == ["d1", "d2", "d3"]
    # Section and page metadata preserved on every displayed ref
    for r in refs:
        assert r.section_path == "9. Termination"
        assert r.page_start == 3 and r.page_end == 4
        assert 0 < len(r.excerpt) <= 500


def test_9b_source_refs_dedup_chunks():
    """build_source_refs never emits the same chunk twice."""
    evidence = [
        search_result_to_evidence(_result("c1", "d1", "Doc1.pdf", text="passage one")),
        search_result_to_evidence(_result("c1", "d1", "Doc1.pdf", text="passage one")),
        search_result_to_evidence(_result("c2", "d1", "Doc1.pdf", text="passage two")),
        search_result_to_evidence(_result("c3", "d2", "Doc2.pdf", text="passage three")),
    ]
    refs = build_source_refs(evidence)
    excerpts = [r.excerpt for r in refs]
    # The duplicated c1 appears exactly once; c2 and c3 preserved
    assert excerpts == ["passage one", "passage two", "passage three"]
    assert len(excerpts) == len(set(excerpts))


# ── Test 11: unsupported / ambiguous queries keep their safe routes ────────

def test_11a_unsupported_corpus_dump_never_retrieves():
    """Corpus-dump request → clarification; no retrieval, no generation."""
    retriever = FakeRetriever([_result()])
    generator = FakeGenerator()
    state = _run(
        _make_graph(retriever, generator),
        query="Show me all text from all documents",
    )

    assert state["intent"].label == "unsupported"
    assert state["intent"].clarification_needed is not None
    assert "cannot" in state["answer"].lower() or "can search" in state["answer"].lower()
    # Call counters: neither retrieval nor generation ran
    assert retriever.calls == []
    assert generator.calls == []
    assert "hybrid_qa" not in _trace_nodes(state)


def test_11b_ambiguous_query_requests_clarification():
    """Ambiguous query → clarification, not a silent hybrid_qa fallback."""
    retriever = FakeRetriever([_result()])
    generator = FakeGenerator()
    state = _run(_make_graph(retriever, generator), query="What about it?")

    assert state["intent"].clarification_needed is not None
    assert "rephrase" in state["answer"].lower() or "can search" in state["answer"].lower()
    assert retriever.calls == []
    assert generator.calls == []


def test_11c_other_capabilities_untouched_by_hybrid_wiring():
    """Corpus overview and structured lookup still route to their stubs
    without touching the injected hybrid retriever or generator."""
    retriever = FakeRetriever([_result()])
    generator = FakeGenerator()
    graph = _make_graph(retriever, generator)

    state = _run(graph, query="How many documents are in the corpus?")
    assert state["intent"].label == "corpus_overview"
    assert "corpus_overview_stub" in _trace_nodes(state)
    assert retriever.calls == []
    assert generator.calls == []

    state = _run(graph, query="What is the governing law of the Bellring agreement?")
    assert state["intent"].label == "structured_lookup"
    assert "structured_lookup_stub" in _trace_nodes(state)
    assert retriever.calls == []
    assert generator.calls == []


# ── Test 12 (bonus): generation failure is safe ─────────────────────────────

def test_12_generation_error_handled_safely():
    """A failing response generator yields an error answer, not a crash."""
    retriever = FakeRetriever([_result()])

    class ExplodingGenerator:
        def __init__(self):
            self.calls = []

        def __call__(self, query, context_chunks):
            self.calls.append(query)
            raise RuntimeError("LLM endpoint unavailable")

    generator = ExplodingGenerator()
    state = _run(_make_graph(retriever, generator))

    assert len(generator.calls) == 1
    assert "Answer generation failed" in state["error"]
    assert state["answer"].startswith("Error:")
    fa = _trace_entry(state, "final_answer")
    assert fa["details"]["has_error"] is True
    assert fa["details"]["generated"] is False


# ── Standalone node unit checks ─────────────────────────────────────────────

def test_final_answer_node_default_matches_foundation():
    """Without a generator, final_answer behaves like the foundation stub."""
    node = make_final_answer_node(None)
    result = node({"abstain": True})
    assert result["answer"] == ABSTAIN_MESSAGE
    result = node({"error": "boom"})
    assert result["answer"] == "Error: boom"
    result = node({})
    assert result["answer"] == "No answer generated."
    assert result["source_refs"] == []


def test_final_answer_reports_failure_distinct_from_abstention():
    """Stage A remediation 2026-10-29: a retrieval failure must reach the
    user as a distinct error report — never the generic abstention message
    (a failure is not a legitimate no-evidence outcome) — while partial
    results and their citations from partially failed multi-document runs
    are still shown (spec 2.8)."""
    from workflow.state import EvidenceItem

    node = make_final_answer_node(None)

    # (a) Total retrieval failure, no evidence: a plain error report,
    #     NOT the generic abstention message.  (In the real graph state,
    #     evidence_validation does not set abstain when an error is present,
    #     so only the error branch can fire.)
    result = node({
        "error": ("Retrieval failed for 2 of 2 document(s); "
                 "comparison is incomplete. Please try again later."),
    })
    assert result["answer"].startswith("Error:")
    assert result["answer"] != ABSTAIN_MESSAGE

    # (b) Partial failure: error reported AND the partial capability result
    #     shown with citations — not silently discarded.
    evidence = [EvidenceItem(
        doc_id="doc_a", doc_name="Alpha.pdf", chunk_id="c_a",
        section_path="9. Termination", page_start=3, page_end=4,
        supporting_text="termination upon material breach",
        retrieval_score=0.9, coverage_status="searched_with_matches",
    )]
    meta = {
        "stub": False,
        "capability": "comparison",
        "answer_text": ("Comparison -- aspect: termination clauses\n"
                        "- Documents compared: 2 (1 with evidence, 1 failed)"),
    }
    result = node({
        "error": ("Retrieval failed for 1 of 2 document(s); "
                  "comparison is incomplete. Please try again later."),
        "metadata_result": meta,
        "evidence": evidence,
    })
    assert result["answer"].startswith("Error:")
    assert "comparison is incomplete" in result["answer"]
    assert "Partial results:" in result["answer"]
    assert "Documents compared: 2" in result["answer"]
    assert result["answer"] != ABSTAIN_MESSAGE
    # Citations from the surviving document are still surfaced.
    assert result["source_refs"]
    assert result["source_refs"][0].doc_id == "doc_a"


# ── D2 regression: document-scoped hybrid retrieval (2026-10-29) ──────────
# D2: hybrid Q&A questions naming a document must retrieve ONLY from that
# document (doc_ids resolved and applied before fusion).  A requested name
# that resolves to nothing is a clarification question, never a silent
# corpus-wide search (same rule as clause_discovery, D11).
# (Regression tests below: 6 cases — node-level scope application,
# unresolvable-name clarification, corpus-wide defaults, and graph-level
# evidence scoping + user-facing clarification.)

from workflow.hybrid_qa import make_hybrid_qa_node
from workflow.state import Intent


class ScopedFakeRetriever(FakeRetriever):
    """FakeRetriever that accepts the doc_ids parameter and honors it like
    the real LakebaseRetriever: the document scope filters results BEFORE
    they are returned (pre-fusion), so out-of-scope chunks never surface."""

    def search(self, query, num_results=5, doc_ids=None):
        self.calls.append({"query": query, "num_results": num_results, "doc_ids": doc_ids})
        if self.error is not None:
            raise self.error
        results = self.results
        if doc_ids is not None:
            wanted = set(doc_ids)
            results = [r for r in results if r.get("doc_id") in wanted]
        return results[:num_results]


class FakeDocsExecutor:
    """Executor serving only the canonical-document enumeration query."""

    def __init__(self, doc_rows):
        self.doc_rows = doc_rows
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append(sql)
        return self.doc_rows


D2_DOC_ROWS = [
    {"doc_id": "doc_upjohn", "file_name": "Upjohn_Manufacturing_and_Supply_Agreement.pdf"},
    {"doc_id": "doc_other", "file_name": "Other_MSA.pdf"},
]


def test_hybrid_qa_node_resolves_and_applies_doc_scope():
    """A doc-named hybrid question retrieves with doc_ids restricted to the
    resolved document."""
    retriever = ScopedFakeRetriever([_result(doc_id="doc_upjohn")])
    node = make_hybrid_qa_node(retriever, sql_executor=FakeDocsExecutor(D2_DOC_ROWS))

    result = node({"user_query": "insurance requirements",
                   "intent": Intent(label="hybrid_qa", scope="specific_docs",
                                    target_doc_names=["Upjohn"])})

    assert retriever.calls[0]["doc_ids"] == ["doc_upjohn"]
    assert result["trace"][0]["details"]["doc_ids"] == ["doc_upjohn"]
    assert len(result["evidence"]) == 1
    assert result["evidence"][0].doc_id == "doc_upjohn"


def test_hybrid_qa_node_unresolvable_name_returns_clarification():
    """A name that resolves to NOTHING asks for clarification — no retrieval
    runs at all (never a silent corpus-wide search)."""
    retriever = ScopedFakeRetriever([_result()])
    node = make_hybrid_qa_node(retriever, sql_executor=FakeDocsExecutor(D2_DOC_ROWS))

    result = node({"user_query": "termination conditions",
                   "intent": Intent(label="hybrid_qa", scope="specific_docs",
                                    target_doc_names=["Acme"])})

    assert retriever.calls == []
    assert result["trace"][0]["action"] == "clarification"
    assert "Could not find documents matching" in result["metadata_result"]["answer_text"]
    assert result["evidence"] == []
    assert "error" not in result


def test_hybrid_qa_node_no_doc_names_corpus_wide():
    """Without target document names, retrieval stays corpus-wide top-k
    (doc_ids is None — unchanged behavior)."""
    retriever = ScopedFakeRetriever([_result()])
    node = make_hybrid_qa_node(retriever, sql_executor=FakeDocsExecutor(D2_DOC_ROWS))

    result = node({"user_query": "insurance requirements",
                   "intent": Intent(label="hybrid_qa", scope="all")})

    assert retriever.calls[0]["doc_ids"] is None
    assert result["trace"][0]["details"]["doc_ids"] is None


def test_hybrid_qa_node_without_executor_unchanged():
    """Without a sql_executor the node cannot resolve names — behavior is
    the original corpus-wide retrieval (backward compatible)."""
    retriever = ScopedFakeRetriever([_result()])
    node = make_hybrid_qa_node(retriever)

    result = node({"user_query": "insurance requirements",
                   "intent": Intent(label="hybrid_qa", scope="specific_docs",
                                    target_doc_names=["Upjohn"])})

    assert retriever.calls[0]["doc_ids"] is None


def test_graph_doc_named_question_scopes_evidence_to_document():
    """Graph-level D2: the final state carries evidence only from the named
    document."""
    retriever = ScopedFakeRetriever([
        _result(doc_id="doc_upjohn"),
        _result(chunk_id="c2", doc_id="doc_other", doc_name="Other_MSA.pdf"),
    ])
    graph = build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        sql_executor=FakeDocsExecutor(D2_DOC_ROWS),
    )

    state = graph.invoke({
        "user_query": "What are the insurance requirements under the Upjohn manufacturing and supply agreement?",
        "conversation_history": [],
    })

    assert state["intent"].label == "hybrid_qa"
    assert state["intent"].target_doc_names
    assert retriever.calls[0]["doc_ids"] == ["doc_upjohn"]
    assert {e.doc_id for e in state.get("evidence", [])} <= {"doc_upjohn"}


def test_graph_unresolvable_doc_name_user_sees_clarification():
    """D11 completion: the clarification question reaches the user instead
    of the generic abstention message."""
    retriever = ScopedFakeRetriever([_result()])
    graph = build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever,
        sql_executor=FakeDocsExecutor(D2_DOC_ROWS),
    )

    state = graph.invoke({
        "user_query": "What are the termination conditions in the Acme contract?",
        "conversation_history": [],
    })

    assert retriever.calls == [], "no retrieval may run for an unresolvable name"
    ans = state.get("answer") or ""
    assert ans != ABSTAIN_MESSAGE
    assert "Could not find documents matching" in ans
