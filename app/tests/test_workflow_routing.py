"""Unit tests for the LangGraph workflow routing foundation (Milestone 2A).

12 test cases covering routing, classification, plan validation, clarification,
follow-up context, safe failure, and determinism.

No LLM, database, Lakebase, or PDF access. Uses RuleBasedIntentClassifier
and stub nodes exclusively.
"""

from __future__ import annotations

import os
import sys

# Add the app folder to sys.path so 'workflow' is importable
_app_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_app_root))

import pytest
from pydantic import ValidationError

from workflow.state import (
    Intent,
    PlanStep,
    ALLOWED_FIELDS,
    ALLOWED_NODES,
    MAX_PLAN_STEPS,
    MAX_SOURCE_DOCS,
    MAX_COMPARISON_DOCS,
)
from workflow.classifier import (
    IntentClassifier,
    RuleBasedIntentClassifier,
    LLMIntentClassifier,
)
from workflow.planner import validate_plan, PlanValidationError
from workflow.guard import is_corpus_dump, is_oversized_request
from workflow.graph import build_graph, route_after_planner
from workflow.traces import format_trace


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
    return RuleBasedIntentClassifier()


@pytest.fixture
def graph(classifier):
    return build_graph(classifier)


def _run(graph, query, history=None):
    """Helper: invoke the graph and return the final state."""
    initial = {"user_query": query, "conversation_history": history or []}
    return graph.invoke(initial)


def _get_route(state) -> str:
    """Extract the route from the trace."""
    trace = state.get("trace", [])
    for entry in trace:
        route = entry.get("route")
        if route:
            return route
    return "unknown"


# ── Test 1: Corpus overview routing ───────────────────────────────────────

def test_1_corpus_overview_routing(graph):
    """Metadata count/distribution questions route to corpus_overview."""
    queries = [
        "How many documents are in the corpus?",
        "What types of documents do you have?",
        "How many contracts are in the corpus?",
    ]
    for q in queries:
        state = _run(graph, q)
        intent: Intent = state["intent"]
        assert intent.label == "corpus_overview", f"Query: {q!r} routed to {intent.label}"
        route = _get_route(state)
        assert route == "corpus_overview", f"Trace route mismatch for {q!r}: {route}"


# ── Test 2: Structured lookup routing ──────────────────────────────────────

def test_2_structured_lookup_routing(graph):
    """Questions about a specific extracted contract field route to structured_lookup."""
    state = _run(graph, "What is the governing law of the Bellring agreement?")
    intent: Intent = state["intent"]
    assert intent.label == "structured_lookup"
    assert intent.target_field == "governing_law"
    assert "Bellring" in intent.target_doc_names
    route = _get_route(state)
    assert route == "structured_lookup"


# ── Test 3: Hybrid Q&A routing ─────────────────────────────────────────────

def test_3_hybrid_qa_routing(graph):
    """Ordinary legal questions route to hybrid_qa."""
    state = _run(graph, "What are the termination conditions in the Cerence contract?")
    intent: Intent = state["intent"]
    assert intent.label == "hybrid_qa", f"Expected hybrid_qa, got {intent.label}"
    assert intent.needs_evidence is True
    route = _get_route(state)
    assert route == "hybrid_qa"


# ── Test 4: Clause discovery routing ──────────────────────────────────────

def test_4_clause_discovery_routing(graph):
    """Find clauses across all contracts routes to clause_discovery."""
    state = _run(graph, "Find all force majeure clauses across all contracts")
    intent: Intent = state["intent"]
    assert intent.label == "clause_discovery"
    assert intent.clause_pattern == "force majeure"
    assert intent.coverage_requirement == "exhaustive"
    route = _get_route(state)
    assert route == "clause_discovery"


# ── Test 5: Comparison routing ────────────────────────────────────────────

def test_5_comparison_routing(graph):
    """Comparing specified agreements routes to comparison."""
    state = _run(graph, "Compare termination clauses between Bellring and Cerence")
    intent: Intent = state["intent"]
    assert intent.label == "comparison"
    assert intent.comparison_aspect is not None
    assert len(intent.target_doc_names) >= 2
    route = _get_route(state)
    assert route == "comparison"


# ── Test 6: Multi-step plan ─────────────────────────────────────────────────

def test_6_multi_step_plan(graph):
    """A complex question generates a valid multi-step plan."""
    query = "Compare termination clauses of Bellring and Cerence, and tell me which has better termination-for-convenience"
    state = _run(graph, query)
    intent: Intent = state["intent"]
    assert intent.label == "multi_step"

    plan = state.get("plan", [])
    assert len(plan) >= 1, "Multi-step plan should have at least 1 step"
    assert len(plan) <= MAX_PLAN_STEPS, f"Plan exceeds max steps: {len(plan)}"

    # Validate the generated plan
    validated = validate_plan(plan)
    assert len(validated) >= 1

    route = _get_route(state)
    assert route == "plan_validation"


# ── Test 7: Unsupported corpus-dump rejection ──────────────────────────────

def test_7_unsupported_corpus_dump(graph):
    """Unsupported corpus-dump requests are rejected."""
    state = _run(graph, "Show me all text from all documents")
    intent: Intent = state["intent"]
    assert intent.label == "unsupported"
    assert intent.clarification_needed is not None
    # Should route to clarification (clarification_needed is set)
    answer = state.get("answer", "")
    assert "cannot" in answer.lower() or "rephrase" in answer.lower() or "can search" in answer.lower()


# ── Test 8: Ambiguous questions request clarification ───────────────────────

def test_8_ambiguous_clarification(graph):
    """Ambiguous questions request clarification."""
    state = _run(graph, "What about it?")
    intent: Intent = state["intent"]
    assert intent.clarification_needed is not None
    # Should route to clarification_node, not any capability node
    answer = state.get("answer", "")
    assert "rephrase" in answer.lower() or "can search" in answer.lower() or "can" in answer.lower()
    route = _get_route(state)
    assert route != "hybrid_qa", "Ambiguous query must not fall back to hybrid_qa"


# ── Test 9: Invalid intent output fails safely ──────────────────────────────

class _ExplodingClassifier(IntentClassifier):
    """Classifier that always raises an exception."""
    def classify(self, query, conversation_history=None):
        raise RuntimeError("Simulated classifier failure")


def test_9_invalid_intent_safe_failure():
    """Invalid intent output fails safely — does NOT silently fall back to hybrid_qa."""
    graph = build_graph(_ExplodingClassifier())
    state = _run(graph, "What is the governing law of the Bellring agreement?")
    intent: Intent = state["intent"]
    assert intent.label == "unsupported", f"Expected unsupported on failure, got {intent.label}"
    assert intent.clarification_needed is not None
    assert intent.label != "hybrid_qa", "Must not silently fall back to hybrid_qa"

    # Also test LLMIntentClassifier with invalid JSON
    def bad_llm(prompt):
        return "not valid json {{"

    llm_classifier = LLMIntentClassifier(bad_llm, max_retries=2)
    intent = llm_classifier.classify("test query")
    assert intent.label == "unsupported"
    assert intent.clarification_needed is not None
    assert intent.label != "hybrid_qa"


# ── Test 10: Plan validation rejections ─────────────────────────────────────

def test_10a_plan_unknown_node_rejected():
    """Plans with unknown nodes are rejected."""
    bad_plan = [{"step_id": 1, "node": "nonexistent_node", "description": "bad"}]
    with pytest.raises(PlanValidationError, match="nonexistent_node"):
        validate_plan(bad_plan)


def test_10b_plan_invalid_field_rejected():
    """Plans with invalid target_field are rejected."""
    bad_plan = [{
        "step_id": 1,
        "node": "structured_lookup",
        "description": "bad field",
        "target_field": "nonexistent_field",
    }]
    with pytest.raises((PlanValidationError, ValidationError)):
        validate_plan(bad_plan)


def test_10c_plan_too_many_steps_rejected():
    """Plans with more than 5 steps are rejected."""
    big_plan = [
        {"step_id": i, "node": "hybrid_qa", "description": f"step {i}"}
        for i in range(1, MAX_PLAN_STEPS + 2)
    ]
    with pytest.raises(PlanValidationError, match="maximum allowed"):
        validate_plan(big_plan)


def test_10d_valid_plan_accepted():
    """A valid plan passes validation."""
    good_plan = [
        {"step_id": 1, "node": "structured_lookup", "target_field": "governing_law",
         "target_doc_names": ["Bellring"]},
        {"step_id": 2, "node": "comparison", "comparison_aspect": "governing law",
         "target_doc_names": ["Bellring", "Cerence"]},
    ]
    validated = validate_plan(good_plan)
    assert len(validated) == 2
    assert validated[0].node == "structured_lookup"
    assert validated[1].node == "comparison"


# ── Test 11: Follow-up uses conversation context ────────────────────────────

def test_11_followup_uses_context(graph):
    """Follow-up questions use supplied conversation context where appropriate."""
    history = [
        {"role": "user", "content": "Compare termination clauses between Bellring and Cerence"},
        {"role": "assistant", "content": "[Comparison] stub result"},
    ]
    state = _run(graph, "What about Armstrong?", history=history)
    intent: Intent = state["intent"]
    assert intent.label == "comparison", f"Follow-up should inherit comparison, got {intent.label}"
    assert intent.comparison_aspect is not None, "Should inherit comparison_aspect from context"
    assert "Armstrong" in intent.target_doc_names, "Should include the new document"


# ── Test 12: Deterministic routing ─────────────────────────────────────────

def test_12_deterministic_routing(graph):
    """Routing is deterministic where the intent is unambiguous."""
    query = "What is the governing law of the Cerence agreement?"
    state1 = _run(graph, query)
    state2 = _run(graph, query)

    intent1: Intent = state1["intent"]
    intent2: Intent = state2["intent"]
    assert intent1.label == intent2.label
    assert intent1.target_field == intent2.target_field

    route1 = _get_route(state1)
    route2 = _get_route(state2)
    assert route1 == route2, f"Non-deterministic routing: {route1} vs {route2}"
    assert route1 == "structured_lookup"

    # Verify trace is consistent
    trace1 = state1.get("trace", [])
    trace2 = state2.get("trace", [])
    assert len(trace1) == len(trace2)
    # First trace entry should be the query planner classification
    assert trace1[0]["node"] == "query_planner"
    assert trace1[0]["route"] == trace2[0]["route"]


# ── Test 13: Invalid classification must not execute an unrelated capability ─

def test_13_invalid_classification_no_capability_execution(graph):
    """An invalid classification (unsupported) must not trigger any capability stub."""
    state = _run(graph, "Show me all text from all documents")
    trace = state.get("trace", [])
    capability_nodes = {
        "corpus_overview_stub", "hybrid_qa_stub", "clause_discovery_stub",
        "structured_lookup_stub", "comparison_stub", "multi_step_stub",
    }
    executed_nodes = {entry["node"] for entry in trace}
    # No capability stub should have been executed
    assert not (executed_nodes & capability_nodes), \
        f"Capability stubs executed for unsupported request: {executed_nodes & capability_nodes}"
    # Should route through clarification_node (unsupported with clarification_needed)
    # or guard_node (unsupported without clarification_needed) — not a capability node
    assert "guard_node" in executed_nodes or "clarification" in executed_nodes, \
        f"Expected guard_node or clarification in trace, got nodes: {executed_nodes}"


# ── Test 14: Incomplete/invalid multi-step plan must not fall back to hybrid_qa ─

class _EmptyMultiStepClassifier(IntentClassifier):
    """Classifier that returns multi_step with attributes causing an empty plan."""
    def classify(self, query, conversation_history=None):
        return Intent(label="multi_step", scope="all")


def test_14_invalid_plan_no_hybrid_qa_fallback():
    """An incomplete multi-step plan must not fall back to hybrid_qa."""
    graph = build_graph(_EmptyMultiStepClassifier())
    state = _run(graph, "complex multi-step query")

    intent: Intent = state["intent"]
    assert intent.label == "multi_step"

    # Should have an error (empty plan rejected by validation)
    assert state.get("error") is not None, \
        "Empty plan should produce an error"

    # No trace entry should reference hybrid_qa
    trace = state.get("trace", [])
    for entry in trace:
        assert entry.get("route") != "hybrid_qa", \
            "Invalid plan must not route to hybrid_qa"
        assert entry.get("node") != "hybrid_qa_stub", \
            "hybrid_qa_stub must not execute on invalid plan"

    # Should route to final_answer with error
    fa_entries = [e for e in trace if e["node"] == "final_answer"]
    assert len(fa_entries) >= 1, "final_answer must execute for invalid plan"
    answer = state.get("answer", "")
    assert "error" in answer.lower(), \
        f"Answer should contain error message, got: {answer!r}"


# ── Test 15: Source-popup limit (3) separate from comparison execution limit (10) ─

def test_15_source_popup_vs_comparison_limits():
    """MAX_SOURCE_DOCS (3) must remain separate from MAX_COMPARISON_DOCS (10)."""
    assert MAX_SOURCE_DOCS == 3, f"MAX_SOURCE_DOCS should be 3, got {MAX_SOURCE_DOCS}"
    assert MAX_COMPARISON_DOCS == 10, f"MAX_COMPARISON_DOCS should be 10, got {MAX_COMPARISON_DOCS}"
    assert MAX_SOURCE_DOCS != MAX_COMPARISON_DOCS, \
        "Source-popup and comparison limits must be distinct values"

    # 4 docs: above source-popup limit (3) but below comparison limit (10)
    intent_4 = Intent(
        label="comparison", scope="specific_docs",
        target_doc_names=["doc1", "doc2", "doc3", "doc4"],
        comparison_aspect="termination",
    )
    assert not is_oversized_request(intent_4), \
        "4 docs should NOT be oversized (below MAX_COMPARISON_DOCS=10)"

    # 11 docs: above comparison limit (10)
    intent_11 = Intent(
        label="comparison", scope="specific_docs",
        target_doc_names=[f"doc{i}" for i in range(11)],
        comparison_aspect="termination",
    )
    assert is_oversized_request(intent_11), \
        "11 docs should be oversized (above MAX_COMPARISON_DOCS=10)"


# ── Test 16: Each of the seven intents routes to the expected graph branch ─

def test_16_all_seven_intents_route_correctly(graph):
    """Each of the 7 intent labels routes to its expected graph branch."""
    # (query, expected_intent_label, expected_route_from_planner)
    cases = [
        ("How many documents are in the corpus?",
         "corpus_overview", "corpus_overview"),
        ("What are the termination conditions in the Cerence contract?",
         "hybrid_qa", "hybrid_qa"),
        ("Find all force majeure clauses across all contracts",
         "clause_discovery", "clause_discovery"),
        ("What is the governing law of the Bellring agreement?",
         "structured_lookup", "structured_lookup"),
        ("Compare termination clauses between Bellring and Cerence",
         "comparison", "comparison"),
        ("Show me all text from all documents",
         "unsupported", "clarification"),
    ]
    for query, expected_label, expected_route in cases:
        state = _run(graph, query)
        intent: Intent = state["intent"]
        assert intent.label == expected_label, \
            f"Query {query!r}: expected {expected_label}, got {intent.label}"
        route = _get_route(state)
        assert route == expected_route, \
            f"Query {query!r}: expected route {expected_route}, got {route}"

    # 7th intent: multi_step routes to plan_validation
    multi_query = (
        "Compare termination clauses of Bellring and Cerence, "
        "and tell me which has better termination-for-convenience"
    )
    state = _run(graph, multi_query)
    assert state["intent"].label == "multi_step"
    route = _get_route(state)
    assert route == "plan_validation", \
        f"Multi-step should route to plan_validation, got {route}"
    assert state.get("error") is None, \
        "Valid multi-step plan should not produce errors"


# ── Test 17: Execution traces record classification, validation, and failure ─

def test_17a_trace_records_classification_and_success(graph):
    """Execution traces accurately record classification and successful plan validation."""
    multi_query = (
        "Compare termination clauses of Bellring and Cerence, "
        "and tell me which has better termination-for-convenience"
    )
    state = _run(graph, multi_query)
    trace = state.get("trace", [])

    # query_planner entry with classification
    qp = [e for e in trace if e["node"] == "query_planner"]
    assert len(qp) >= 1, "query_planner trace entry missing"
    assert qp[0]["action"] == "classified"
    assert qp[0]["route"] == "plan_validation"

    # plan_validation entry with success
    pv = [e for e in trace if e["node"] == "plan_validation"]
    assert len(pv) >= 1, "plan_validation trace entry missing"
    assert pv[0]["action"] == "validated"

    # multi_step_stub executed
    ms = [e for e in trace if e["node"] == "multi_step_stub"]
    assert len(ms) >= 1, "multi_step_stub trace entry missing"

    # final_answer generated
    fa = [e for e in trace if e["node"] == "final_answer"]
    assert len(fa) >= 1, "final_answer trace entry missing"


def test_17b_trace_records_validation_failure():
    """Execution traces accurately record plan validation failure."""
    graph = build_graph(_EmptyMultiStepClassifier())
    state = _run(graph, "complex query")
    trace = state.get("trace", [])

    # query_planner classification recorded
    qp = [e for e in trace if e["node"] == "query_planner"]
    assert len(qp) >= 1
    assert qp[0]["action"] == "classified"

    # plan_validation rejection recorded
    pv = [e for e in trace if e["node"] == "plan_validation"]
    assert len(pv) >= 1, "plan_validation trace entry missing"
    assert pv[0]["action"] == "rejected"
    assert pv[0]["route"] == "final_answer"

    # final_answer with error flag
    fa = [e for e in trace if e["node"] == "final_answer"]
    assert len(fa) >= 1
    assert fa[0].get("details", {}).get("has_error") is True, \
        "final_answer trace should record has_error=True"


# ── Test 18: Valid plans still execute the intended foundation stub ─

def test_18_valid_plan_executes_stub(graph):
    """A valid multi-step plan executes the multi_step_stub foundation node."""
    multi_query = (
        "Compare termination clauses of Bellring and Cerence, "
        "and tell me which has better termination-for-convenience"
    )
    state = _run(graph, multi_query)

    # No error
    assert state.get("error") is None

    # Plan has at least 2 steps
    plan = state.get("plan", [])
    assert len(plan) >= 2, f"Expected >=2 plan steps, got {len(plan)}"

    # multi_step_stub trace entries (one per step)
    trace = state.get("trace", [])
    ms = [e for e in trace if e["node"] == "multi_step_stub"]
    assert len(ms) >= 2, \
        f"Expected >=2 multi_step_stub entries (one per step), got {len(ms)}"

    # evidence_validation ran
    ev = [e for e in trace if e["node"] == "evidence_validation"]
    assert len(ev) >= 1, "evidence_validation should run after multi_step_stub"

    # final_answer generated
    fa = [e for e in trace if e["node"] == "final_answer"]
    assert len(fa) >= 1, "final_answer should execute"


# ── Hosted smoke finding (2026-10-29): name phrase without a trailing
# generic noun must still resolve and route to structured_lookup ──────────

class TestNameFallbackWithoutAgreementNoun:
    """'What is the governing law of the Armstrong Flooring intellectual
    property' (no 'agreement' noun — incl. the truncated hosted phrasing)
    previously extracted NO document name and misrouted to corpus_overview
    field coverage. The fallback in _extract_single_doc reduces the
    prepositional phrase to its leading capitalized run, so these route to
    structured_lookup scoped to the resolved name."""

    @pytest.mark.parametrize("query", [
        "What is the governing law of the Armstrong Flooring intellectual property agreement?",  # noun present (unchanged)
        "What is the governing law of the Armstrong Flooring intellectual property",
        "What is the governing law of the Armstrong Flooring intellectual proper",   # truncated hosted phrasing
        "What is the governing law of the Armstrong Flooring?",
        "What is the effective date of the Upjohn Manufacturing",   # no noun, multi-word name
    ])
    def test_field_question_without_trailing_noun_routes_to_lookup(self, query):
        clf = RuleBasedIntentClassifier()
        it = clf.classify(query, conversation_history=[])
        assert it.label == "structured_lookup", (
            f"expected structured_lookup, got {it.label}")
        assert it.scope == "specific_docs"
        assert it.target_doc_names, "a name must be resolved"
        assert it.target_doc_names[0].startswith(("ArmstrongFlooring", "Upjohn")), (
            f"unexpected name core {it.target_doc_names[0]!r}")

    @pytest.mark.parametrize("query", [
        "What is the governing law of the agreement?",      # generic-only phrase: no name
        "What is the governing law of the contracts?",
        "What is the governing law?",                        # no prepositional phrase
        "What are the payment terms for software maintenance?",  # lowercase phrase: no name
    ])
    def test_no_name_is_ever_guessed(self, query):
        clf = RuleBasedIntentClassifier()
        it = clf.classify(query, conversation_history=[])
        assert not it.target_doc_names, (
            f"no document name may be invented for {query!r}")

    def test_value_filter_and_discovery_routes_unchanged(self):
        clf = RuleBasedIntentClassifier()
        vf = clf.classify("Which agreements are governed by Delaware law?",
                          conversation_history=[])
        assert vf.label == "structured_lookup" and vf.scope == "filtered"
        assert not vf.target_doc_names
        disc = clf.classify(
            "Find all indemnification clauses in the Armstrong Flooring intellectual property agreement",
            conversation_history=[])
        assert disc.label == "clause_discovery"


# ── Fix 008 (hosted smoke finding 2026-10-29): 'can you repeat the answer
# again?' must re-display the previous answer, not run a legal retrieval ──

class TestRepeatPreviousAnswer:
    """The hosted app treated 'i couldn't see the pdf, can you repeat the
    answer again?' as a legal question: hybrid_qa retrieved 4 irrelevant
    chunks and abstained. It is a CONVERSATIONAL repeat request: the last
    assistant answer must be re-displayed verbatim through the
    clarification route (no retrieval, no SQL, no LLM), like the greeting
    fix 002. The frozen 7-label taxonomy is unchanged (label stays
    'unsupported'; only the message differs)."""

    HISTORY = [
        {"role": "user", "content": "What is the governing law of the Armstrong Flooring agreement?"},
        {"role": "assistant", "content": "The governing law is Delaware."},
    ]

    @pytest.mark.parametrize("query", [
        "i couldnt see the pdf can you repeat the answer again?",  # exact hosted phrasing
        "can you repeat the answer again?",
        "I couldn't see that — say that again",
        "please show the answer again",
    ])
    def test_repeat_re_displays_previous_answer(self, query):
        clf = RuleBasedIntentClassifier()
        it = clf.classify(query, conversation_history=self.HISTORY)
        assert it.clarification_needed, (
            f"repeat request must use the clarification route, got {it.label}")
        assert "Delaware" in it.clarification_needed, (
            "the previous answer must be re-displayed verbatim")
        assert it.label == "unsupported"  # frozen taxonomy unchanged

    def test_first_turn_repeat_has_nothing_to_repeat(self):
        clf = RuleBasedIntentClassifier()
        it = clf.classify("can you repeat the answer again?", conversation_history=[])
        # falls through to normal classification — never fabricates an answer
        assert "Delaware" not in (it.clarification_needed or "")

    @pytest.mark.parametrize("query", [
        "What is the governing law of the Armstrong Flooring agreement?",
        "Find all indemnification clauses in the Upjohn agreement",
    ])
    def test_legal_questions_never_hijacked(self, query):
        clf = RuleBasedIntentClassifier()
        it = clf.classify(query, conversation_history=self.HISTORY)
        assert it.clarification_needed is None
        assert it.label in ("structured_lookup", "clause_discovery", "hybrid_qa",
                            "comparison", "corpus_overview")
