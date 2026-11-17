"""LangGraph StateGraph with conditional routing for the Legal Assistant.

Graph structure (spec Section 2.1-2.4):

  START → query_planner → route_after_planner (conditional)
    ├── clarification_node → END
    ├── plan_validation → route_after_plan_validation (conditional)
    │     ├── multi_step_stub → evidence_validation → final_answer → END
    │     └── final_answer → END  (on plan validation error)
    ├── guard_node → final_answer → END
    ├── corpus_overview → final_answer → END
    ├── hybrid_qa → evidence_validation → final_answer → END
    ├── clause_discovery → evidence_validation → final_answer → END
    ├── structured_lookup → final_answer → END
    └── comparison → evidence_validation → final_answer → END

Dependency injection (Milestone 2B.1 + Milestone 3):
  - query_planner is injected with an IntentClassifier (RuleBased for
    tests/fallback, LLM-based for production).
  - hybrid_qa is the REAL retrieval node when a retriever is passed to
    build_graph (wrapping LakebaseRetriever.search_hybrid_v2); without a
    retriever the deterministic foundation stub is used so routing tests
    need no Lakebase connection.
  - structured_lookup is the REAL SQL node when a sql_executor is passed
    to build_graph (parameterized, allowlisted queries over
    extracted_fields/documents/evidence); without an executor the
    deterministic foundation stub is used (Milestone 3).
  - corpus_overview is the REAL SQL node when a sql_executor is passed
    to build_graph (aggregate counts and distributions over
    documents/extracted_fields); without an executor the deterministic
    foundation stub is used (Milestone 3).
  - clause_discovery is the REAL per-document retrieval node when BOTH a
    retriever and a sql_executor are passed to build_graph (retriever for
    document-filtered hybrid retrieval, executor for canonical-doc
    enumeration and name resolution); without both the deterministic
    foundation stub is used (Milestone 3).
  - comparison is the REAL per-document retrieval + comparison node when
    BOTH a retriever and a sql_executor are passed to build_graph (retriever
    for document-filtered evidence retrieval, executor for canonical-doc
    enumeration and name resolution, optional answer_generator for grounded
    comparison narrative); without both the deterministic foundation stub
    is used (Milestone 3).
  - evidence_validation is the real validation node (empty-text rejection,
    chunk dedup, 24k context budget, abstention).
  - final_answer is injected with the existing response generator
    (LakebaseRetriever.generate_response) for evidence-grounded answers;
    without one it formats results exactly like the foundation stub.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, START, END

from .state import WorkflowState, Intent
from .classifier import IntentClassifier, make_query_planner_node, RuleBasedIntentClassifier
from .planner import plan_validation_node
from .guard import guard_node
from .hybrid_qa import make_hybrid_qa_node, DEFAULT_NUM_RESULTS
from .structured_lookup import make_structured_lookup_node
from .corpus_overview import make_corpus_overview_node
from .clause_discovery import make_clause_discovery_node
from .comparison import make_comparison_node
from .validation import make_evidence_validation_node
from .evidence_refinement import make_evidence_refinement_node
from .answers import make_final_answer_node
from .stubs import (
    corpus_overview_stub,
    hybrid_qa_stub,
    clause_discovery_stub,
    structured_lookup_stub,
    comparison_stub,
    multi_step_stub,
    clarification_node,
)


# ── Routing functions ─────────────────────────────────────────────────────

def route_after_planner(state: dict) -> str:
    """Conditional routing after the query planner classifies intent.

    Returns the name of the next node to execute.
    """
    intent: Optional[Intent] = state.get("intent")

    if intent is None:
        return "guard_node"

    # Clarification route: do not execute any capability node
    if intent.clarification_needed:
        return "clarification_node"

    # Route based on intent label
    label = intent.label
    if label == "multi_step":
        return "plan_validation"
    if label == "unsupported":
        return "guard_node"
    if label == "corpus_overview":
        return "corpus_overview"
    if label == "hybrid_qa":
        return "hybrid_qa"
    if label == "clause_discovery":
        return "clause_discovery"
    if label == "structured_lookup":
        return "structured_lookup"
    if label == "comparison":
        return "comparison"

    # Safe fallback: unknown label → guard
    return "guard_node"


def route_after_plan_validation(state: dict) -> str:
    """Conditional routing after plan validation.

    If validation succeeded (no error), proceed to multi_step_stub.
    If validation failed (error set), go directly to final_answer.
    """
    if state.get("error"):
        return "final_answer"
    return "multi_step_stub"


def route_after_comparison(state: dict) -> str:
    """Conditional routing after the comparison node (fix 010).

    The metadata-backed comparison modes — candidate discovery ('what
    documents can be compared?') and the default-dimension matrix — carry
    NO chunk evidence by design.  Routing them through evidence_validation
    would mark the state abstaining (rule: no usable evidence = abstain)
    and mask the pre-formatted answer with the generic abstention message.
    They carry their own claim-mapped sources in comparison_result and go
    straight to final_answer.  Chunk-evidence comparisons (explicit
    aspects like 'governing law') behave exactly as before.
    """
    comp = state.get("comparison_result")
    if isinstance(comp, dict) and comp.get("mode") in ("candidates", "matrix"):
        return "final_answer"
    return "evidence_validation"


# ── Graph builder ─────────────────────────────────────────────────────────

def build_graph(
    classifier: Optional[IntentClassifier] = None,
    retriever=None,
    answer_generator=None,
    num_results: int = DEFAULT_NUM_RESULTS,
    sql_executor=None,
):
    """Build and compile the LangGraph StateGraph.

    Args:
        classifier: IntentClassifier instance.  If None, uses RuleBasedIntentClassifier
                    (deterministic, no LLM).  Tests should pass a RuleBasedIntentClassifier
                    or a custom stub.  Production should pass LLMIntentClassifier.
        retriever: object exposing search(query, num_results) — the real
                   LakebaseRetriever in production, a fake in unit tests.
                   If None, the deterministic hybrid_qa foundation stub is
                   used (no retrieval backend wired).
        answer_generator: callable(query, context_chunks) -> str matching
                          LakebaseRetriever.generate_response.  If None, the
                          final_answer node formats results without LLM
                          generation (foundation behavior).
        num_results: top-k for hybrid retrieval (default 5).
        sql_executor: object exposing execute(sql, params) -> list[dict] —
                      the real SQL-warehouse executor in production
                      (StatementExecutionSqlExecutor or an approved
                      equivalent), a fake in unit tests.  If None, the
                      deterministic structured_lookup foundation stub is
                      used (no database wired).

    Returns:
        A compiled LangGraph that can be invoked with:
            graph.invoke({"user_query": "...", "conversation_history": [...]})
    """
    if classifier is None:
        classifier = RuleBasedIntentClassifier()

    query_planner = make_query_planner_node(classifier)

    # Milestone 2B.1: real hybrid Q&A node when a retriever is injected;
    # foundation stub otherwise (keeps routing tests Lakebase-free).
    if retriever is not None:
        hybrid_qa_node = make_hybrid_qa_node(retriever, num_results=num_results, sql_executor=sql_executor)
    else:
        hybrid_qa_node = hybrid_qa_stub

    # Milestone 3: real structured-lookup node when a SQL executor is
    # injected; foundation stub otherwise (keeps routing tests DB-free).
    if sql_executor is not None:
        structured_lookup_node = make_structured_lookup_node(sql_executor)
    else:
        structured_lookup_node = structured_lookup_stub

    # Milestone 3: real corpus-overview node when a SQL executor is
    # injected; foundation stub otherwise (keeps routing tests DB-free).
    if sql_executor is not None:
        corpus_overview_node = make_corpus_overview_node(sql_executor)
    else:
        corpus_overview_node = corpus_overview_stub

    # Milestone 3: real clause-discovery node when BOTH a retriever and a
    # SQL executor are injected (retriever for per-document hybrid retrieval,
    # executor for canonical-doc enumeration and name resolution); foundation
    # stub otherwise (keeps routing tests Lakebase/DB-free).
    if retriever is not None and sql_executor is not None:
        clause_discovery_node = make_clause_discovery_node(
            retriever, sql_executor, num_results=num_results,
        )
    else:
        clause_discovery_node = clause_discovery_stub

    # Milestone 3: real comparison node when BOTH a retriever and a
    # SQL executor are injected (retriever for per-document evidence,
    # executor for canonical-doc enumeration and name resolution); the
    # optional answer_generator is also passed through for grounded
    # comparison narrative; foundation stub otherwise (keeps routing
    # tests Lakebase/DB-free).
    if retriever is not None and sql_executor is not None:
        comparison_node = make_comparison_node(
            retriever, sql_executor, answer_generator,
            num_results=num_results,
        )
    else:
        comparison_node = comparison_stub

    # Real evidence validation (empty-text, dedup, context budget, abstention)
    evidence_validation = make_evidence_validation_node()

    # Fix 011 (2026-10-30, Stage A): evidence refinement between hybrid_qa
    # and evidence_validation - topic-term relevance filtering plus
    # same-section/adjacent-chunk expansion so complete provisions enter
    # the evidence.  The chunk fetcher comes from the real retriever
    # (get_chunks_for_expansion); without a retriever the node still
    # filters but cannot expand (tests pass fixtures directly).
    chunk_fetcher = getattr(retriever, "get_chunks_for_expansion", None)
    evidence_refinement = make_evidence_refinement_node(chunk_fetcher)

    # Final answer: LLM-grounded generation when a generator is injected
    final_answer = make_final_answer_node(answer_generator)

    graph_builder = StateGraph(WorkflowState)

    # ── Add nodes ──
    graph_builder.add_node("query_planner", query_planner)
    graph_builder.add_node("plan_validation", plan_validation_node)
    graph_builder.add_node("guard_node", guard_node)
    graph_builder.add_node("corpus_overview", corpus_overview_node)
    graph_builder.add_node("hybrid_qa", hybrid_qa_node)
    graph_builder.add_node("clause_discovery", clause_discovery_node)
    graph_builder.add_node("structured_lookup", structured_lookup_node)
    graph_builder.add_node("comparison", comparison_node)
    graph_builder.add_node("multi_step_stub", multi_step_stub)
    graph_builder.add_node("evidence_refinement", evidence_refinement)
    graph_builder.add_node("evidence_validation", evidence_validation)
    graph_builder.add_node("final_answer", final_answer)
    graph_builder.add_node("clarification_node", clarification_node)

    # ── Add edges ──
    graph_builder.add_edge(START, "query_planner")

    # Conditional routing after query planner
    graph_builder.add_conditional_edges(
        "query_planner",
        route_after_planner,
        {
            "clarification_node": "clarification_node",
            "plan_validation": "plan_validation",
            "guard_node": "guard_node",
            "corpus_overview": "corpus_overview",
            "hybrid_qa": "hybrid_qa",
            "clause_discovery": "clause_discovery",
            "structured_lookup": "structured_lookup",
            "comparison": "comparison",
        },
    )

    # Clarification → END
    graph_builder.add_edge("clarification_node", END)

    # Plan validation → conditional (multi_step_stub or final_answer on error)
    graph_builder.add_conditional_edges(
        "plan_validation",
        route_after_plan_validation,
        {
            "multi_step_stub": "multi_step_stub",
            "final_answer": "final_answer",
        },
    )

    # Guard → final_answer
    graph_builder.add_edge("guard_node", "final_answer")

    # Capability stubs → their respective next nodes
    graph_builder.add_edge("corpus_overview", "final_answer")
    graph_builder.add_edge("structured_lookup", "final_answer")
    # Fix 011 (Stage A): hybrid retrieval passes through evidence refinement
    # (topic filter + section expansion) BEFORE technical validation.  The
    # other evidence-producing routes (clause_discovery, multi_step, chunk
    # comparisons) validate as before — their retrieval semantics already
    # scope per document/pattern.
    graph_builder.add_edge("hybrid_qa", "evidence_refinement")
    graph_builder.add_edge("evidence_refinement", "evidence_validation")
    graph_builder.add_edge("clause_discovery", "evidence_validation")
    # Fix 010: metadata-backed comparison modes (candidates / matrix) skip
    # evidence_validation (they have no chunk evidence and would be marked
    # abstaining); chunk-evidence comparisons validate as before.
    graph_builder.add_conditional_edges(
        "comparison",
        route_after_comparison,
        {
            "evidence_validation": "evidence_validation",
            "final_answer": "final_answer",
        },
    )
    graph_builder.add_edge("multi_step_stub", "evidence_validation")

    # Evidence validation → final_answer
    graph_builder.add_edge("evidence_validation", "final_answer")

    # Final answer → END
    graph_builder.add_edge("final_answer", END)

    return graph_builder.compile()
