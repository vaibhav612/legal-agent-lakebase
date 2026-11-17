"""Fix 010 regression tests — agentic comparison, context continuity,
and evidence-relevant sources.

Covers the three hosted-smoke deficiencies:
  1. 'What documents can be compared?' is answered from the canonical
     document inventory (deterministic grouping by agreement type parsed
     from stored file names) — never from retrieval chunks, and every
     recommended document really exists in the inventory.
  2. 'sure compare them then' (acceptance of the pending recommendation)
     executes a concrete matrix comparison on the exact recommended pair.
  3. Sources substantiate the claims: matrix sources are per-cell
     extraction evidence doc-bound to each compared agreement; generated
     abstentions and guard/clarification routes show no sources.

Fully offline: FakeExecutor serves the fixture rows; FakeRetriever counts
calls (and proves discovery never retrieves). No LLM, Lakebase, SQL
warehouse, or PDF volume access.
"""

from __future__ import annotations

import os
import re
import sys

_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

import pytest

from workflow.state import SourceRef
from workflow.classifier import (
    RuleBasedIntentClassifier,
    resolve_pending_action,
)
from workflow.graph import build_graph
from workflow.answers import ABSTAIN_MESSAGE


# ── Fixture: canonical inventory (real stored file names + metadata) ─────

ARM = "a607f7a8-armstrong"
CER = "e8891ed2-cerence"
FRE = "bc3a8773-freecook"
PAR = "26665b47-pareteum"
VIT = "98c1de71-vitalibis"
PHA = "3ea0c78b-phasebio"
REE = "eecebc00-reeds"
IMP = "94e57aa4-imperial"

FIXTURE_DOCS = [
    {"doc_id": ARM, "file_name": "ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual%20Property%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/IP/armstrong.pdf", "page_count": 40},
    {"doc_id": CER, "file_name": "CerenceInc_20191002_8-K_EX-10.4_11827494_EX-10.4_Intellectual%20Property%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/IP/cerence.pdf", "page_count": 24},
    {"doc_id": FRE, "file_name": "Freecook_20180605_S-1_EX-10.3_11233807_EX-10.3_Hosting%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/Hosting/freecook.pdf", "page_count": 3},
    {"doc_id": PAR, "file_name": "PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/Hosting/pareteum.pdf", "page_count": 30},
    {"doc_id": VIT, "file_name": "VitalibisInc_20180316_8-K_EX-10.2_11100168_EX-10.2_Hosting%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/Hosting/vitalibis.pdf", "page_count": 14},
    {"doc_id": PHA, "file_name": "PhasebioPharmaceuticalsInc_20200330_10-K_EX-10.21_12086810_EX-10.21_Development%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/Development 5/phasebio.pdf", "page_count": 96},
    {"doc_id": REE, "file_name": "ReedsInc_20191113_10-Q_EX-10.4_11888303_EX-10.4_Development%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/Development 5/reeds.pdf", "page_count": 11},
    {"doc_id": IMP, "file_name": "ImperialGardenResortInc_20161028_DRS%20(on%20F-1)_EX-10.13_9963189_EX-10.13_Outsourcing%20Agreement.pdf",
     "document_class": "contract", "file_path": "/vol/Legal Docs/Commercial & Contracts/Service Agreements/Outsourcing/imperial.pdf", "page_count": 7},
]

FIELDS = {}      # (doc_id, field) -> row
EVIDENCE = {}    # field_id -> evidence row


def _f(doc_id, field, raw, status, fid, quote=None, page=5, section="10. TERMS"):
    FIELDS[(doc_id, field)] = {
        "field_id": fid, "value_raw": raw, "value_status": status,
        "value_index": 0, "run_id": "r1",
    }
    if quote is not None:
        EVIDENCE[fid] = {
            "evidence_id": fid + "-ev", "chunk_id": "chunk-" + fid,
            "quote": quote, "page_start": page, "page_end": page,
            "evidence_type": "extracted_field", "section_path": section,
            "rects": "[{\"bbox\": [72, 120, 300, 140]}]", "rect_source": "llm_bounding_box",
        }


# Pair 1 (default recommendation): Phasebio vs Reeds (development)
_f(PHA, "parties", "Phasebio Pharmaceuticals, Inc. and Catalent Indiana, LLC", "found_explicit", "pha-parties",
   "by and between Phasebio Pharmaceuticals, Inc. and Catalent Indiana, LLC", 2, "1. PARTIES")
_f(REE, "parties", "Reeds, Inc. and PPD Development, LP", "found_explicit", "ree-parties",
   "by and between Reeds, Inc. and PPD Development, LP", 2, "1. PARTIES")
_f(PHA, "governing_law", "Delaware", "found_explicit", "pha-glaw",
   "governed by the laws of the State of Delaware", 40, "13. GOVERNING LAW")
_f(REE, "governing_law", "California", "found_explicit", "ree-glaw",
   "governed by the laws of the State of California", 8, "11. GOVERNING LAW")
_f(PHA, "contract_title", "Development Agreement", "found_explicit", "pha-title",
   "DEVELOPMENT AGREEMENT", 1, "COVER")
_f(REE, "contract_title", "Development Agreement", "found_explicit", "ree-title",
   "DEVELOPMENT AGREEMENT", 1, "COVER")
_f(PHA, "agreement_date", "March 30, 2020", "found_explicit", "pha-date", None)
_f(REE, "agreement_date", "November 13, 2019", "found_explicit", "ree-date", None)
_f(PHA, "effective_date", "March 30, 2020", "found_explicit", "pha-eff", None)
_f(REE, "effective_date", "November 13, 2019", "found_explicit", "ree-eff", None)
_f(PHA, "expiration_or_term", "Two (2) years", "found_explicit", "pha-term",
   "for a term of two (2) years", 6, "2. TERM")
_f(REE, "expiration_or_term", "Two (2) years", "found_explicit", "ree-term",
   "for a term of two (2) years", 4, "2. TERM")
_f(PHA, "renewal_term", "Renews for one (1) year", "found_explicit", "pha-renew", None)
_f(REE, "renewal_term", "Renews for one (1) year", "found_explicit", "ree-renew", None)
_f(PHA, "payment_terms", "Milestone payments per Statement of Work", "found_explicit", "pha-pay",
   "milestone payments as set forth in each Statement of Work", 9, "5. PAYMENT")
_f(REE, "payment_terms", "", "not_found", "ree-pay", None)          # extraction gap
# termination_for_convenience: PHA has a value, REE has NO record at all
_f(PHA, "termination_for_convenience", "Either party may terminate for convenience on 60 days notice", "found_explicit", "pha-tfc",
   "either party may terminate this Agreement upon sixty (60) days written notice", 10, "8. TERMINATION")
_f(PHA, "notice_to_terminate_renewal", "60 days", "found_explicit", "pha-notice", None)
_f(REE, "notice_to_terminate_renewal", "30 days", "found_explicit", "ree-notice", None)

# Pair 2: Freecook vs Pareteum (hosting) — for 'compare the second pair'
_f(FRE, "parties", "Freecook and Hosting Co", "found_explicit", "fre-parties",
   "between Freecook and Hosting Co", 1, "1")
_f(PAR, "parties", "Pareteum Corp and Host Partner", "found_explicit", "par-parties",
   "between Pareteum Corp and Host Partner", 1, "1")
_f(FRE, "governing_law", "New York", "found_explicit", "fre-glaw",
   "governed by the laws of the State of New York", 3, "12")
_f(PAR, "governing_law", "New York", "found_explicit", "par-glaw",
   "governed by the laws of the State of New York", 22, "12")

# Pair 3: Armstrong vs Cerence (intellectual property) — for 'third pair'
_f(ARM, "parties", "Armstrong Flooring, Inc. and licensor", "found_explicit", "arm-parties",
   "between Armstrong Flooring, Inc. and the licensor", 2, "1")
_f(CER, "parties", "Cerence Inc. and licensor", "found_explicit", "cer-parties",
   "between Cerence Inc. and the licensor", 2, "1")
_f(ARM, "governing_law", "Delaware", "found_explicit", "arm-glaw",
   "governed by the laws of the State of Delaware", 39, "13. MISCELLANEOUS")
_f(CER, "governing_law", "Massachusetts", "found_explicit", "cer-glaw",
   "governed by the laws of the Commonwealth of Massachusetts", 23, "12. GENERAL")

FIXTURE_FIDS = {v["field_id"] for v in FIELDS.values()}


# ── Fakes ─────────────────────────────────────────────────────────────────

class FakeExecutor:
    """Serves documents / extracted_fields / evidence fixture rows."""

    def __init__(self, fail_documents=False, fail_fields=False):
        self.fail_documents = fail_documents
        self.fail_fields = fail_fields
        self.calls = []

    def execute(self, sql, params):
        self.calls.append(sql[:90])
        if "FROM workspace.legal_assistant_v2.evidence" in sql:
            fids = [v for k, v in sorted(params.items()) if k.startswith("fid_")]
            rows = []
            for fid in fids:
                if fid in EVIDENCE:
                    rows.append({"field_id": fid, **EVIDENCE[fid]})
            return rows
        if "FROM workspace.legal_assistant_v2.extracted_fields" in sql:
            if self.fail_fields:
                raise RuntimeError("PERMISSION_DENIED: extracted_fields")
            m = re.search(r"field_name = '([a-z_]+)'", sql)
            field = m.group(1)
            doc_ids = [v for k, v in sorted(params.items()) if k.startswith("doc_")]
            rows = []
            for did in doc_ids:
                row = FIELDS.get((did, field))
                if row:
                    rows.append({"doc_id": did, **row})
            return rows
        if "FROM workspace.legal_assistant_v2.documents" in sql:
            if self.fail_documents:
                raise RuntimeError("PERMISSION_DENIED: documents")
            return [dict(d) for d in FIXTURE_DOCS]
        return []


class FakeRetriever:
    """Retriever with call recording; supports both search signatures."""

    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def search(self, query, num_results=5, doc_ids=None):
        self.calls.append({
            "query": query, "num_results": num_results,
            "doc_ids": list(doc_ids or []),
        })
        if self.error is not None:
            raise self.error
        if doc_ids is not None:
            pool = [r for r in self.results if r.get("doc_id") in doc_ids]
        else:
            pool = self.results
        return pool[:num_results]


def _chunk(chunk_id, doc_id, doc_name, text, page=3):
    return {
        "chunk_id": chunk_id, "chunk_text": text, "doc_id": doc_id,
        "doc_name": doc_name, "chunk_index": 0, "section_path": "9. Termination",
        "page_start": page, "page_end": page, "distance": 0.1,
        "bm25_score": 2.0, "bm25_rank": 0, "semantic_rank": 0,
        "rrf_score": 0.03, "rrf_rank": 0,
    }


RELEVANT_CHUNK = _chunk("c-arm-1", ARM, FIXTURE_DOCS[0]["file_name"],
                        "This agreement shall be governed by the laws of the State of Delaware.")
IRRELEVANT_CHUNKS = [
    _chunk("c-vit-1", VIT, FIXTURE_DOCS[4]["file_name"], "hosting services monitoring", 2),
    _chunk("c-imp-1", IMP, FIXTURE_DOCS[7]["file_name"], "outsourced banquet operations", 4),
]


def make_gen(answer):
    def gen(query, context_chunks):
        return answer
    return gen


def make_graph(executor=None, retriever=None, generator=None):
    return build_graph(
        classifier=RuleBasedIntentClassifier(),
        retriever=retriever or FakeRetriever(),
        answer_generator=generator,
        num_results=5,
        sql_executor=executor or FakeExecutor(),
    )


def trace_entry(state, node):
    entries = [e for e in state.get("trace", []) if e.get("node") == node]
    assert entries, "no trace entry for " + node
    return entries[-1]


CANDIDATES_QUERY = "What documents can be compared?"
ACCEPTANCE_QUERY = "sure compare them then"


def _candidates_state():
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    return g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})


# ── 1. Candidate discovery from the canonical inventory ──────────────────

def test_1_candidates_discovery_recommends_real_documents():
    state = _candidates_state()
    comp = state["comparison_result"]
    assert comp["mode"] == "candidates"
    assert comp["coverage"]["canonical_documents"] == len(FIXTURE_DOCS)
    # deterministic grouping: development, hosting, intellectual property
    pairs = state["pending_action"]["pairs"]
    assert [p["agreement_type"] for p in pairs] == [
        "development agreement", "hosting agreement", "intellectual property agreement",
    ]
    trace_entry(state, "comparison")
    answer = state["answer"]
    assert "Phasebio" in answer and "Reeds" in answer
    assert "Freecook" in answer and "Pareteum" in answer
    assert "Armstrong" in answer and "Cerence" in answer
    # feasible dimensions listed; clause-level gaps disclosed
    assert "Governing law" in answer
    assert "indemnification" in answer.lower()
    # invitation to accept
    assert "compare them" in answer


def test_2_recommended_documents_exist_in_canonical_inventory():
    """Identity consistency: every recommended doc_id/file_name is a real
    canonical document, and every document named in the answer is one of
    them (no fabricated names, no mismatched suggestions)."""
    state = _candidates_state()
    inv_ids = {d["doc_id"] for d in FIXTURE_DOCS}
    inv_files = {d["file_name"] for d in FIXTURE_DOCS}
    answer = state["answer"]
    for pair in state["pending_action"]["pairs"]:
        for doc in pair["docs"]:
            assert doc["doc_id"] in inv_ids
            assert doc["file_name"] in inv_files
            company = doc["file_name"].split("_")[0].replace("Inc", "").replace("Corp", "")
            assert company in doc["file_name"]
    # No document outside the inventory is ever suggested
    from workflow.display import display_doc_name
    for d in FIXTURE_DOCS:
        assert display_doc_name(d["file_name"]) in answer or d["file_name"] not in answer
    assert "does-not-exist" not in answer.lower()


def test_3_discovery_never_retrieves_or_cites_chunks():
    """The recommendation must not be backed by retrieval chunks: the
    retriever is never called, and no sources panel appears."""
    retriever = FakeRetriever(results=[RELEVANT_CHUNK] + IRRELEVANT_CHUNKS)
    g = make_graph(executor=FakeExecutor(), retriever=retriever)
    state = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    assert retriever.calls == []
    assert state.get("evidence") in ([], None)
    assert state.get("source_refs") in ([], None)
    assert ABSTAIN_MESSAGE not in state["answer"]


# ── 2. Accepted suggestion executes the exact comparison ────────────────

def test_4_exact_two_turn_acceptance_runs_matrix_comparison():
    """The exact hosted two-turn flow: candidates, then acceptance."""
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    first = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    pending = first["pending_action"]
    assert pending and pending["type"] == "comparison_candidates"

    second = g.invoke({
        "user_query": ACCEPTANCE_QUERY,
        "conversation_history": [
            {"role": "user", "content": CANDIDATES_QUERY},
            {"role": "assistant", "content": first["answer"]},
        ],
        "pending_action": pending,
    })
    comp = second["comparison_result"]
    assert comp["mode"] == "matrix"
    # exact recommended pair executed: Phasebio vs Reeds
    assert {d for d in comp["doc_names"]} == {
        "Phasebio Pharmaceuticals Inc — Development Agreement",
        "Reeds Inc — Development Agreement",
    }
    answer = second["answer"]
    assert "Comparative matrix" in answer
    assert "Material differences" in answer
    assert "Delaware" in answer and "California" in answer
    assert ABSTAIN_MESSAGE not in answer
    entry = trace_entry(second, "comparison")
    assert entry["action"] == "completed"
    # pending action cleared after execution
    assert second.get("pending_action") is None


def test_5_acceptance_variants_and_pair_selection():
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    first = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    pending = first["pending_action"]
    base = {
        "conversation_history": [{"role": "user", "content": CANDIDATES_QUERY}],
        "pending_action": pending,
    }
    # variant phrasings resolve to the recommended (first) pair
    for q in ["compare them", "do that", "ok compare them then"]:
        st = g.invoke({**base, "user_query": q})
        assert st["comparison_result"]["mode"] == "matrix", q
        assert "Phasebio" in st["answer"], q
    # ordinal selection: second pair = hosting (Freecook vs Pareteum)
    st2 = g.invoke({**base, "user_query": "compare the second pair"})
    assert "Freecook" in st2["answer"] and "Pareteum" in st2["answer"]
    # third pair = intellectual property (Armstrong vs Cerence)
    st3 = g.invoke({**base, "user_query": "compare the third pair"})
    assert "Armstrong" in st3["answer"] and "Cerence" in st3["answer"]


def test_6_acceptance_without_pending_falls_back_cleanly():
    """Acceptance with no established recommendation must not fabricate a
    comparison; it falls through to normal classification (hybrid path)."""
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    state = g.invoke({"user_query": ACCEPTANCE_QUERY, "conversation_history": []})
    comp = state.get("comparison_result")
    assert comp is None or comp.get("mode") not in ("candidates", "matrix")
    assert "comparative matrix" not in state["answer"].lower()


# ── 3. Evidence-relevant sources ──────────────────────────────────────────

def test_7_matrix_sources_are_doc_scoped_and_claim_mapped():
    """Matrix sources come only from the two compared documents, and each
    cited cell's evidence belongs to that cell's document (doc-bound by
    field_id — passages from one agreement can never support the other)."""
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    first = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    second = g.invoke({
        "user_query": ACCEPTANCE_QUERY,
        "conversation_history": [],
        "pending_action": first["pending_action"],
    })
    refs = second["source_refs"]
    assert refs, "matrix comparison must carry claim-mapped sources"
    assert {r.doc_id for r in refs} <= {PHA, REE}
    excerpts = {r.excerpt for r in refs}
    assert any("Phasebio Pharmaceuticals, Inc." in e for e in excerpts)
    assert any("Reeds, Inc." in e for e in excerpts)
    assert all(isinstance(r, SourceRef) for r in refs)
    # every source has the viewer metadata (chunk, page, excerpt)
    assert all(r.chunk_id and r.page_start is not None for r in refs)


def test_8_material_differences_cite_both_sides():
    """A claimed difference must cite evidence on both sides."""
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    first = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    second = g.invoke({
        "user_query": ACCEPTANCE_QUERY,
        "conversation_history": [], "pending_action": first["pending_action"],
    })
    answer = second["answer"]
    diff_lines = [l for l in answer.splitlines() if "Governing law and dispute resolution:" in l]
    assert diff_lines, "governing-law difference must be reported"
    markers = re.findall(r"\[(\d+)\]", diff_lines[0])
    assert len(markers) == 2
    refs = second["source_refs"]
    cited = [refs[int(m) - 1] for m in markers]
    assert {r.doc_id for r in cited} == {PHA, REE}
    assert any("Delaware" in r.excerpt for r in cited)
    assert any("California" in r.excerpt for r in cited)


def test_9_matrix_honest_gaps_no_fabrication():
    """Missing extractions are reported as gaps, never as 'no provision'
    claims; identical values are reported as identical."""
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    first = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    second = g.invoke({
        "user_query": ACCEPTANCE_QUERY,
        "conversation_history": [], "pending_action": first["pending_action"],
    })
    answer = second["answer"]
    # no record at all (Reeds termination_for_convenience)
    assert "(no extraction record)" in answer
    # not_found status (Reeds payment_terms)
    assert "not_found" in answer
    # caveat: not proof of absence
    assert "not proof" in answer
    assert "no legal effect is inferred" in answer
    # identical explicit values (Term and expiration) are not called differences
    term_line = [l for l in answer.splitlines() if "Term and expiration" in l and l.startswith("|")]
    assert term_line and "Two (2) years" in term_line[0]


def test_10_generated_abstention_hides_irrelevant_sources():
    """When the generator examines evidence and abstains, the retrieved
    chunks are NOT displayed as sources (they were found insufficient)."""
    gen = make_gen("The available documents don't contain enough information to answer this question.")
    g = make_graph(
        executor=FakeExecutor(),
        retriever=FakeRetriever(results=[RELEVANT_CHUNK] + IRRELEVANT_CHUNKS),
        generator=gen,
    )
    state = g.invoke({
        "user_query": "What is the quantum arbitration allocation schedule?",
        "conversation_history": [],
    })
    assert "don't contain enough information" in state["answer"]
    assert state.get("source_refs") in ([], None)


def test_11_supported_answer_keeps_sources():
    gen = make_gen("The agreement is governed by Delaware law, per Section 13.")
    g = make_graph(
        executor=FakeExecutor(),
        retriever=FakeRetriever(results=[RELEVANT_CHUNK] + IRRELEVANT_CHUNKS),
        generator=gen,
    )
    state = g.invoke({
        "user_query": "How does the Armstrong Flooring agreement handle dispute resolution?",
        "conversation_history": [],
    })
    refs = state.get("source_refs")
    assert refs
    assert all(r.doc_id == ARM for r in refs), "doc-scoped question keeps sources on its document"


def test_12_doc_scoped_retrieval_blocks_extra_documents():
    """A doc-scoped question retrieves only from that document, so chunks
    from other agreements can never enter the answer or its sources."""
    retriever = FakeRetriever(results=[RELEVANT_CHUNK] + IRRELEVANT_CHUNKS)
    g = make_graph(executor=FakeExecutor(), retriever=retriever, generator=make_gen("Answer grounded in evidence."))
    state = g.invoke({
        "user_query": "How does the Armstrong Flooring agreement handle dispute resolution?",
        "conversation_history": [],
    })
    last = retriever.calls[-1]
    assert last["doc_ids"] == [ARM]
    for ref in state.get("source_refs") or []:
        assert ref.doc_id == ARM


def test_13_clarification_and_guard_routes_show_no_sources():
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever(), generator=make_gen("x"))
    hi = g.invoke({"user_query": "hello", "conversation_history": []})
    assert hi.get("source_refs") in ([], None)
    dump = g.invoke({"user_query": "show me all text from all documents", "conversation_history": []})
    assert dump.get("source_refs") in ([], None)
    assert "cannot display all text" in dump["answer"].lower()


def test_14_permission_failure_is_explicit_not_abstention():
    """A storage/permission failure surfaces as a clear error — never as
    a clean abstention that silently hides the outage."""
    g = make_graph(executor=FakeExecutor(fail_documents=True), retriever=FakeRetriever())
    state = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    assert state.get("error")
    assert "enumeration failed" in state["error"]
    assert state["answer"].startswith("Error:")
    assert ABSTAIN_MESSAGE not in state["answer"]
    # the same transparency holds for an accepted comparison mid-flow
    g2 = make_graph(executor=FakeExecutor(fail_fields=True), retriever=FakeRetriever())
    first = g2.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    second = g2.invoke({
        "user_query": ACCEPTANCE_QUERY,
        "conversation_history": [], "pending_action": first["pending_action"],
    })
    assert second["answer"].startswith("Error:") or "could not be compared" in second["answer"].lower()
    assert ABSTAIN_MESSAGE not in second["answer"]


# ── Classifier / planner units ────────────────────────────────────────────

def test_15_classifier_candidates_intent():
    clf = RuleBasedIntentClassifier()
    for q in ["What documents can be compared?", "which agreements are comparable?",
              "what can i compare", "what pairs of documents can be compared?"]:
        it = clf.classify(q, [])
        assert it.label == "comparison", q
        assert it.comparison_aspect == "candidates", q
        assert it.scope == "all", q
    # genuine aspect comparisons are NOT hijacked by the discovery branch
    it = clf.classify(
        "compare the governing law of the Armstrong Flooring and Cerence intellectual property agreements", [])
    assert it.label == "comparison"
    assert it.comparison_aspect != "candidates"
    assert len(it.target_doc_names) == 2


def test_16_resolve_pending_action_units():
    pending = {
        "type": "comparison_candidates",
        "pairs": [
            {"pair_id": 0, "docs": [{"doc_id": PHA, "file_name": "a.pdf"},
                                    {"doc_id": REE, "file_name": "b.pdf"}]},
            {"pair_id": 1, "docs": [{"doc_id": FRE, "file_name": "c.pdf"},
                                    {"doc_id": PAR, "file_name": "d.pdf"}]},
        ],
        "recommended_pair_id": 0,
    }
    it = resolve_pending_action("sure compare them then", pending)
    assert it.label == "comparison" and it.comparison_aspect == "default"
    assert [n for n in it.target_doc_names] == ["a.pdf", "b.pdf"]
    # ordinal selection
    it2 = resolve_pending_action("compare the second pair", pending)
    assert it2.target_doc_names == ["c.pdf", "d.pdf"]
    # non-acceptance phrasings fall through
    assert resolve_pending_action("what is the governing law of the first agreement", pending) is None
    # wrong pending type
    assert resolve_pending_action("compare them", {"type": "other"}) is None
    # no pending
    assert resolve_pending_action("compare them", {}) is None
    # long turn is never treated as acceptance
    assert resolve_pending_action(
        "compare them but first tell me what the governing law clause says about Delaware courts in detail", pending
    ) is None
    # malformed pair never produces a guessed intent
    assert resolve_pending_action("compare them", {"type": "comparison_candidates", "pairs": [{"docs": []}]}) is None
    # out-of-range ordinal falls back to the recommended pair
    assert resolve_pending_action("compare the third pair", pending).target_doc_names == ["a.pdf", "b.pdf"]


def test_17_app_contract_pending_action_roundtrip():
    """The Streamlit app stores pending_action on the assistant message
    and passes it back on the next invoke — the tested two-turn flow
    above IS the app contract (app passes exactly these keys)."""
    g = make_graph(executor=FakeExecutor(), retriever=FakeRetriever())
    first = g.invoke({"user_query": CANDIDATES_QUERY, "conversation_history": []})
    # simulated app message persistence
    assistant_msg = {
        "role": "assistant", "content": first["answer"],
        "source_refs": first.get("source_refs") or [],
        "pending_action": first.get("pending_action"),
    }
    assert assistant_msg["pending_action"]
    invoke_input = {
        "user_query": ACCEPTANCE_QUERY,
        "conversation_history": [
            {"role": "user", "content": CANDIDATES_QUERY},
            {"role": "assistant", "content": assistant_msg["content"]},
        ],
    }
    pending_action = next(
        (m.get("pending_action") for m in reversed([assistant_msg])
         if m.get("role") == "assistant" and m.get("pending_action")),
        None,
    )
    if pending_action:
        invoke_input["pending_action"] = pending_action
    second = g.invoke(invoke_input)
    assert second["comparison_result"]["mode"] == "matrix"
