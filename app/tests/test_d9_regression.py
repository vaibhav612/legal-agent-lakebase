"""Regression tests for defect D9 (baseline run M4-BASE-001).

D9 covers intent-classification and document-name-resolution failures
observed in the first benchmark baseline.  Each test class maps to one
reviewed benchmark failure mode:

  - TestD01FullTitleFieldLookup: a field question naming a document by its
    FULL multi-word title ('the Armstrong Flooring intellectual property
    agreement') must route to structured_lookup with a name that resolves
    to exactly the intended document.  Pre-fix, only the last capitalized
    word ('property') was captured and matched unrelated documents.
  - TestD04LeadingTokenRetention: a multi-word company name ('West
    Pharmaceutical Services') must be kept whole; a bare trailing fragment
    ('Services') must never silently match unrelated 'Services' documents.
  - TestE01LeadingArticle: leading articles ('the Armstrong Flooring') must
    be stripped so comparison names resolve.
  - TestC02DiscoveryDocScope: doc-specific clause discovery must keep
    scope='specific_docs' with the named document (pre-fix the name was
    lost and the search silently broadened corpus-wide).
  - TestG01MultiStepPhrasing: 'and which ... governed by' must classify as
    multi_step with a per-clause plan (lookup + value_filter), not collapse
    into a single structured_lookup.
  - TestH02OversizedComparison: a corpus-wide comparison ('all 56 contracts
    on all 10 fields') must be rejected as unsupported with alternatives,
    never misrouted to hybrid_qa.
  - TestNoRegression: previously passing routing, ambiguity handling
    (multiple matches -> clarification, never arbitrary selection),
    unresolved-name reporting, and value-filter routing stay unchanged.

No LLM, database, Lakebase, or PDF access: the rule-based classifier and
stub nodes only.  DOCS file names mirror real EDGAR-style names from the
corpus so name cores resolve exactly as they do in production.

Reference: docs/evaluation/milestone4/benchmark_v1.jsonl (reviewed P1 cases
M4-D01, M4-D04, M4-E01, M4-C02, M4-G01, M4-H02) and
baseline_run_001_results.md (defect D9).
"""

from __future__ import annotations

import os
import sys

_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

import pytest

from workflow.classifier import (
    RuleBasedIntentClassifier,
    _generate_stub_plan,
    _split_multi_step_query,
)
from workflow.clause_discovery import _resolve_doc_names
from workflow.graph import build_graph
from workflow.planner import validate_plan


@pytest.fixture
def classifier():
    return RuleBasedIntentClassifier()


@pytest.fixture
def graph(classifier):
    return build_graph(classifier)


def _run(graph, query, history=None):
    return graph.invoke(
        {"user_query": query, "conversation_history": history or []}
    )


def _route(state) -> str:
    for entry in state.get("trace", []):
        route = entry.get("route")
        if route:
            return route
    return "unknown"


# Real EDGAR-style canonical file names (subset mirroring the corpus).
DOCS = [
    {"doc_id": "d_armstrong",
     "file_name": "ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual Property Agreement.pdf"},
    {"doc_id": "d_cerence",
     "file_name": "CerenceInc_20200330_10-K_EX-10.1_11965007_EX-10.1_IP Agreement.pdf"},
    {"doc_id": "d_west",
     "file_name": "WestPharmaceuticalServicesInc_20200116_8-K_EX-10.1_11947529_EX-10.1_Supply Agreement.pdf"},
    {"doc_id": "d_precheck",
     "file_name": "PrecheckHealthServicesInc_20220228_10-K_EX-10.20_16683923_EX-10.20_Master Services Agreement.pdf"},
    {"doc_id": "d_neurobo",
     "file_name": "NeuroboPharmaceuticalsInc_20220811_10-K_EX-10.1_19310883_EX-10.1_License Agreement.pdf"},
    {"doc_id": "d_bellring1",
     "file_name": "BellringBrandsInc_20190107_8-K_EX-10.1_11471795_EX-10.1_Manufacturing and Supply Agreement1.pdf"},
    {"doc_id": "d_bellring2",
     "file_name": "BellringBrandsInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Manufacturing and Supply Agreement2.pdf"},
]


G01_QUERY = (
    "What is the governing law of the Armstrong agreement, "
    "and which agreements are governed by Delaware law?"
)


# ── D9(a): M4-D01 — full-title field lookup ───────────────────────────────

class TestD01FullTitleFieldLookup:

    def test_routes_to_structured_lookup_with_full_title(self, classifier):
        """Full multi-word title keeps its leading name tokens (D9(a))."""
        intent = classifier.classify(
            "What is the governing law of the Armstrong Flooring "
            "intellectual property agreement?"
        )
        assert intent.label == "structured_lookup", (
            f"expected structured_lookup, got {intent.label}"
        )
        assert intent.target_field == "governing_law"
        assert intent.scope == "specific_docs"
        # The whole company-name core survives — not the trailing
        # 'property' fragment that pre-fix matched unrelated IP agreements.
        assert intent.target_doc_names == ["ArmstrongFlooring"]

    def test_extracted_name_resolves_to_unique_document(self):
        """The extracted core matches exactly one canonical document."""
        resolved, ambiguous, unresolved = _resolve_doc_names(None, DOCS, ["ArmstrongFlooring"])
        assert len(resolved) == 1
        assert resolved[0]["doc_id"] == "d_armstrong"
        assert ambiguous == [] and unresolved == []


# ── D9(b): M4-D04 — leading-token retention / fragment mismatch ───────────

class TestD04LeadingTokenRetention:

    def test_multiword_company_name_extracted_in_full(self, classifier):
        """'West Pharmaceutical Services' is kept whole (D9(c))."""
        intent = classifier.classify(
            "What is the governing law of the West Pharmaceutical Services agreement?"
        )
        assert intent.label == "structured_lookup"
        assert intent.target_field == "governing_law"
        assert intent.target_doc_names == ["WestPharmaceuticalServices"]

    def test_whole_name_resolves_and_fragment_is_ambiguous(self):
        """The whole name resolves uniquely; the bare 'Services' fragment
        stays ambiguous (clarification), never an arbitrary match."""
        resolved, ambiguous, unresolved = _resolve_doc_names(
            None, DOCS, ["WestPharmaceuticalServices"])
        assert len(resolved) == 1
        assert resolved[0]["doc_id"] == "d_west"

        # Pre-fix failure root: 'Services' substring-matched unrelated
        # documents (PrecheckHealthServices, West, ...).
        resolved2, ambiguous2, _ = _resolve_doc_names(None, DOCS, ["Services"])
        assert resolved2 == []
        assert len(ambiguous2) == 1
        assert len(ambiguous2[0]["matching_files"]) >= 2

    def test_west_core_excludes_unrelated_services_docs(self):
        """The West core must not resolve to Precheck/Neurobo documents."""
        resolved, _, _ = _resolve_doc_names(None, DOCS, ["WestPharmaceuticalServices"])
        ids = {d["doc_id"] for d in resolved}
        assert "d_precheck" not in ids
        assert "d_neurobo" not in ids


# ── D9(b): M4-E01 — leading article in comparison names ─────────────────

class TestE01LeadingArticle:

    def test_leading_the_stripped_from_comparison_names(self, classifier):
        """'the Armstrong Flooring' → 'ArmstrongFlooring' (D9(b))."""
        intent = classifier.classify(
            "Compare the indemnification provisions of the Armstrong "
            "Flooring and Cerence agreements"
        )
        assert intent.label == "comparison", (
            f"expected comparison, got {intent.label}"
        )
        assert intent.target_doc_names == ["ArmstrongFlooring", "Cerence"]
        assert "indemnification" in (intent.comparison_aspect or "")

    def test_both_comparison_names_resolve(self):
        resolved, ambiguous, unresolved = _resolve_doc_names(
            None, DOCS, ["ArmstrongFlooring", "Cerence"])
        assert len(resolved) == 2
        assert {d["doc_id"] for d in resolved} == {"d_armstrong", "d_cerence"}
        assert ambiguous == [] and unresolved == []

    def test_graph_routes_to_comparison(self, graph):
        state = _run(
            graph,
            "Compare the indemnification provisions of the Armstrong "
            "Flooring and Cerence agreements",
        )
        assert state["intent"].label == "comparison"
        assert _route(state) == "comparison"


# ── D9(a): M4-C02 — doc-specific discovery scope ──────────────────────────

class TestC02DiscoveryDocScope:

    def test_discovery_keeps_named_document(self, classifier):
        """Multi-word doc names must not be dropped to corpus-wide scope
        (silent broadening, defect D11 recurrence)."""
        intent = classifier.classify(
            "Find indemnification provisions in the Armstrong Flooring "
            "intellectual property agreement"
        )
        assert intent.label == "clause_discovery"
        assert intent.scope == "specific_docs"
        assert intent.target_doc_names == ["ArmstrongFlooring"]
        assert "indemnification" in (intent.clause_pattern or "")

    def test_discovery_name_resolves(self):
        resolved, ambiguous, unresolved = _resolve_doc_names(
            None, DOCS, ["ArmstrongFlooring"])
        assert len(resolved) == 1 and not ambiguous and not unresolved

    def test_selected_without_names_stays_corpus_wide(self, classifier):
        """'in these selected agreements' (no names) still yields [] names."""
        intent = classifier.classify(
            "Find indemnification provisions in these selected agreements"
        )
        assert intent.label == "clause_discovery"
        assert intent.target_doc_names == []
        assert intent.scope == "all"


# ── D9(d): M4-G01 — multi-step phrasing collapsed into one lookup ─────────

class TestG01MultiStepPhrasing:

    def test_not_collapsed_into_single_lookup(self, classifier):
        intent = classifier.classify(G01_QUERY)
        assert intent.label == "multi_step", (
            f"'and which ... governed by' must not collapse to a single "
            f"structured_lookup (got {intent.label})"
        )

    def test_query_splits_into_two_clauses(self):
        clauses = _split_multi_step_query(G01_QUERY)
        assert len(clauses) == 2
        assert "governing law of the Armstrong" in clauses[0]
        assert "governed by Delaware law" in clauses[1]

    def test_two_step_plan_lookup_then_value_filter(self, classifier):
        """Plan = structured_lookup(lookup) + structured_lookup(value_filter),
        matching benchmark M4-G01's expected route."""
        intent = classifier.classify(G01_QUERY)
        plan = _generate_stub_plan(intent, G01_QUERY)
        assert len(plan) == 2, f"expected 2 steps, got {len(plan)}"
        step1, step2 = plan
        assert step1["node"] == "structured_lookup"
        assert step1["target_field"] == "governing_law"
        assert step1["target_doc_names"] == ["Armstrong"]
        assert step1["target_value"] is None
        assert step2["node"] == "structured_lookup"
        assert step2["target_field"] == "governing_law"
        assert step2["target_doc_names"] == []
        assert step2["target_value"] == "Delaware law"
        # Plan validates against the allowlists
        validated = validate_plan(plan)
        assert len(validated) == 2

    def test_graph_routes_to_plan_validation_without_error(self, graph):
        state = _run(graph, G01_QUERY)
        assert state["intent"].label == "multi_step"
        assert state.get("error") is None, f"unexpected error: {state.get('error')}"
        assert _route(state) == "plan_validation"
        plan = state.get("plan", [])
        assert len(plan) == 2
        assert plan[0].node == "structured_lookup"
        assert plan[0].target_doc_names == ["Armstrong"]
        assert plan[1].node == "structured_lookup"
        assert plan[1].target_value == "Delaware law"

    def test_value_filter_without_marker_stays_single_lookup(self, classifier):
        """A plain governed-by question is still one structured_lookup."""
        intent = classifier.classify("Which agreements are governed by Delaware?")
        assert intent.label == "structured_lookup"
        assert intent.target_field == "governing_law"
        assert intent.target_value == "Delaware"
        assert intent.scope == "filtered"


# ── D9(e): M4-H02 — oversized comparison misrouted to hybrid_qa ────────────

class TestH02OversizedComparison:

    def test_corpus_wide_comparison_rejected_as_unsupported(self, classifier):
        intent = classifier.classify("Compare all 56 contracts on all 10 fields")
        assert intent.label == "unsupported", (
            f"oversized comparison must be rejected, got {intent.label}"
        )
        assert intent.clarification_needed is not None

    def test_graph_rejects_without_capability_execution(self, graph):
        state = _run(graph, "Compare all 56 contracts on all 10 fields")
        assert state["intent"].label == "unsupported"
        executed = {e["node"] for e in state.get("trace", [])}
        capability_nodes = {
            "hybrid_qa_stub", "comparison_stub", "structured_lookup_stub",
            "clause_discovery_stub", "corpus_overview_stub", "multi_step_stub",
        }
        assert not (executed & capability_nodes), (
            f"capability stubs executed for oversized comparison: "
            f"{executed & capability_nodes}"
        )
        answer = state.get("answer", "")
        assert answer, "rejection must produce an answer"
        assert "too large" in answer.lower() or "named documents" in answer.lower()

    def test_named_two_doc_comparison_still_routes(self, classifier):
        """The oversized guard must not reject ordinary 2-doc comparisons."""
        intent = classifier.classify(
            "Compare the governing law of the Armstrong and Cerence agreements"
        )
        assert intent.label == "comparison"
        assert len(intent.target_doc_names) == 2


# ── No-regression: routing, ambiguity, resolution safety ─────────────────

class TestNoRegression:

    ROUTE_CASES = [
        ("What is the governing law of the Bellring agreement?",
         "structured_lookup"),
        ("What are the termination conditions in the Cerence contract?",
         "hybrid_qa"),
        ("How many documents are in the corpus?", "corpus_overview"),
        ("Find all force majeure clauses across all contracts",
         "clause_discovery"),
        ("Compare termination clauses between Bellring and Cerence",
         "comparison"),
        ("Show me all text from all documents", "unsupported"),
        ("Compare termination clauses of Bellring and Cerence, "
         "and tell me which has better termination-for-convenience",
         "multi_step"),
    ]

    def test_previously_passing_routes_unchanged(self, classifier):
        for query, expected in self.ROUTE_CASES:
            intent = classifier.classify(query)
            assert intent.label == expected, (
                f"{query!r}: expected {expected}, got {intent.label}"
            )

    def test_single_word_name_still_extracted(self, classifier):
        intent = classifier.classify(
            "What is the governing law of the Bellring agreement?"
        )
        assert "Bellring" in intent.target_doc_names

    def test_ambiguous_name_clarifies_never_picks(self):
        """'Bellring' matches 2 files → ambiguous (clarification path),
        never an arbitrary single selection."""
        resolved, ambiguous, unresolved = _resolve_doc_names(None, DOCS, ["Bellring"])
        assert resolved == []
        assert len(ambiguous) == 1
        assert len(ambiguous[0]["matching_files"]) == 2
        assert unresolved == []

    def test_unresolved_name_reported_not_dropped(self):
        """Unresolvable names surface as unresolved — no silent broadening."""
        resolved, ambiguous, unresolved = _resolve_doc_names(
            None, DOCS, ["NonexistentCo"])
        assert resolved == [] and ambiguous == []
        assert unresolved == [{"requested_name": "NonexistentCo"}]

    def test_duplicate_named_docs_deduped(self):
        """Naming the same doc twice resolves once (dedup preserved)."""
        resolved, _, _ = _resolve_doc_names(
            None, DOCS, ["Cerence", "Cerence"])
        assert len(resolved) == 1
