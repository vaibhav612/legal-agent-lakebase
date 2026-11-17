"""Typed shared workflow state and data models for the Legal Assistant LangGraph.

Matches behavior specification Section 2.2-2.3:
  - WorkflowState: 15 fields as defined in the spec
  - Intent: 7 fixed labels with structured output and cross-field validation
  - PlanStep: bounded execution plan steps
  - EvidenceItem: uniform evidence structure
  - SourceRef: source reference for citations

All models use Pydantic v2 for validation.  WorkflowState is a TypedDict
so LangGraph can merge partial node updates.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict
from operator import add

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Constants from behavior spec ──────────────────────────────────────────

ALLOWED_FIELDS: list[str] = [
    "agreement_date",
    "contract_title",
    "effective_date",
    "expiration_or_term",
    "governing_law",
    "notice_to_terminate_renewal",
    "parties",
    "payment_terms",
    "renewal_term",
    "termination_for_convenience",
]

ALLOWED_NODES: list[str] = [
    "corpus_overview",
    "hybrid_qa",
    "clause_discovery",
    "structured_lookup",
    "comparison",
    "source_reference",
]

INTENT_LABELS: list[str] = [
    "corpus_overview",
    "hybrid_qa",
    "clause_discovery",
    "structured_lookup",
    "comparison",
    "multi_step",
    "unsupported",
]

SCOPE_VALUES: list[str] = ["all", "specific_docs", "filtered"]
COVERAGE_VALUES: list[str] = ["exhaustive", "best_effort", "top_k"]

# Status filters for structured_lookup value-quality questions (Milestone 3).
# Maps to actual value_status values (see workflow/structured_lookup.py):
#   missing -> not_found, plus documents with no extraction record
#   uncertain -> inferred, conflicting
STATUS_FILTER_VALUES: list[str] = ["missing", "uncertain", "missing_or_uncertain"]

MAX_PLAN_STEPS: int = 5
MAX_DOC_LISTING: int = 50
MAX_SOURCE_DOCS: int = 3  # Source-popup display limit (spec Section 1.7)
MAX_COMPARISON_DOCS: int = 10  # Execution limit for cross-document comparison

# Evidence context budget (spec Section 2.6, rule 4).  Mirrors
# MAX_CONTEXT_CHARS = 24000 in lakebase_retriever.py.  The authoritative
# context packing (headers + separators) remains _assemble_context in the
# retriever; this constant gates the workflow-level evidence budget on the
# cumulative supporting_text length.
MAX_EVIDENCE_CONTEXT_CHARS: int = 24000


# ── Pydantic models ──────────────────────────────────────────────────────

class Intent(BaseModel):
    """Classified intent from the query planner (spec Section 2.3).

    Uses fixed labels and structured output.  Cross-field validation ensures
    label-specific requirements are met (e.g. structured_lookup needs target_field).
    """

    label: Literal[
        "corpus_overview",
        "hybrid_qa",
        "clause_discovery",
        "structured_lookup",
        "comparison",
        "multi_step",
        "unsupported",
    ]
    scope: str = Field(default="all")
    target_doc_names: list[str] = Field(default_factory=list)
    target_field: Optional[str] = None
    target_value: Optional[str] = None  # structured_lookup: value fragment (bound SQL parameter, never SQL text)
    status_filter: Optional[str] = None  # structured_lookup: missing | uncertain | missing_or_uncertain
    clause_pattern: Optional[str] = None
    comparison_aspect: Optional[str] = None
    coverage_requirement: str = Field(default="best_effort")
    needs_evidence: bool = False
    clarification_needed: Optional[str] = None
    # Fix 011 (2026-10-30, additive): resolved search text for follow-up
    # turns.  When a follow-up ('what about the limits?') reuses the previous
    # question's document scope, the planner combines the previous query
    # with the follow-up phrasing here so retrieval sees the full intent —
    # the raw follow-up text alone ('the limits') is not a usable search
    # query.  None (default) preserves the existing behavior of searching
    # with the raw user_query.
    search_query: Optional[str] = None
    # Fix 012 (2026-10-31, additive, Defect D): set by _resolve_followup when
    # this intent was produced by resolving a follow-up against the previous
    # turn.  The planner uses it (together with prior_answer_docs from the
    # app) to scope the search to the document that actually backed the
    # previous ANSWER when the follow-up itself names no agreement.
    resolved_from_followup: bool = False

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        if v not in SCOPE_VALUES:
            raise ValueError(f"scope must be one of {SCOPE_VALUES}, got '{v}'")
        return v

    @field_validator("coverage_requirement")
    @classmethod
    def _validate_coverage(cls, v: str) -> str:
        if v not in COVERAGE_VALUES:
            raise ValueError(
                f"coverage_requirement must be one of {COVERAGE_VALUES}, got '{v}'"
            )
        return v

    @field_validator("target_field")
    @classmethod
    def _validate_target_field(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALLOWED_FIELDS:
            raise ValueError(
                f"target_field must be one of {ALLOWED_FIELDS} or None, got '{v}'"
            )
        return v

    @field_validator("status_filter")
    @classmethod
    def _validate_status_filter(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in STATUS_FILTER_VALUES:
            raise ValueError(
                f"status_filter must be one of {STATUS_FILTER_VALUES} or None, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def _validate_label_consistency(self) -> "Intent":
        """Cross-field validation: label-specific requirements."""
        if self.label == "structured_lookup" and not self.target_field:
            raise ValueError("structured_lookup intent requires target_field")
        if self.label == "clause_discovery" and not self.clause_pattern:
            raise ValueError("clause_discovery intent requires clause_pattern")
        if self.label == "comparison":
            if not self.comparison_aspect:
                raise ValueError("comparison intent requires comparison_aspect")
            # Fix 010 (2026-10-29, additive): aspect='candidates' is the
            # document-discovery request 'what documents can be compared?'
            # — it carries NO target documents by design (the node discovers
            # them from the canonical inventory). All other comparison
            # aspects still require at least 2 resolved document names.
            if (
                self.comparison_aspect != "candidates"
                and len(self.target_doc_names) < 2
            ):
                raise ValueError(
                    "comparison intent requires at least 2 target_doc_names"
                )
        return self


class PlanStep(BaseModel):
    """A single step in a multi-step execution plan (spec Section 2.10).

    Each step targets a known capability node with valid parameters.
    """

    step_id: int
    node: str
    description: str = ""
    target_doc_names: list[str] = Field(default_factory=list)
    target_field: Optional[str] = None
    clause_pattern: Optional[str] = None
    comparison_aspect: Optional[str] = None
    # structured_lookup steps (D9(d)/G01): carry the value fragment and
    # status filter of the clause intent so value_filter plan steps keep
    # their semantics.  target_value is bound as a SQL parameter by the
    # lookup node downstream — never SQL text.
    target_value: Optional[str] = None
    status_filter: Optional[str] = None

    @field_validator("node")
    @classmethod
    def _validate_node(cls, v: str) -> str:
        if v not in ALLOWED_NODES:
            raise ValueError(
                f"node must be one of {ALLOWED_NODES}, got '{v}'"
            )
        return v

    @field_validator("target_field")
    @classmethod
    def _validate_target_field(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALLOWED_FIELDS:
            raise ValueError(
                f"target_field must be one of {ALLOWED_FIELDS} or None, got '{v}'"
            )
        return v


class EvidenceItem(BaseModel):
    """Uniform evidence structure (spec Section 2.6).

    Every evidence-producing node must populate this structure.
    """

    doc_id: str
    doc_name: str
    chunk_id: str
    section_path: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    supporting_text: str
    retrieval_score: Optional[float] = None
    retrieval_method: str = Field(default="hybrid")
    coverage_status: Optional[str] = None
    # Fix 011 (2026-10-30, additive, Stage C): table structure preserved
    # end-to-end.  table_html carries the exact <table> markup (cell
    # boundaries, column headers, row order) extracted from the source PDF
    # and synced from UC chunks_v2 into Lakebase document_chunks_v2.
    # has_table mirrors the UC flag.  Both optional so existing producers
    # and tests stay compatible.
    table_html: Optional[str] = None
    has_table: Optional[bool] = None
    # Fix 011 (2026-10-30, additive, Stage A): provenance of this evidence
    # item in the refinement lifecycle — 'retrieved' (top-k fusion result)
    # or 'section_expansion' (same-section/adjacent chunk added to complete
    # a provision).  Display priority uses retrieval first.
    evidence_role: Optional[str] = None


class SourceRef(BaseModel):
    """Source reference for citations (spec Section 1.7).

    Max 3 documents shown at a time.

    Stage 2/3 fix (2026-10-29): optional citation-enrichment fields for the
    source viewer -- the linked chunk (chunk_id), the exact evidence
    rectangles (rects, [{page_id, coord}], stored 0-indexed page ids), the
    rectangle provenance (rect_source), and the canonical PDF location
    (file_path) + page_count for page navigation. All are optional and
    default to None/[] so existing producers stay compatible. Canonical
    stored values are never modified here; display formatting happens in
    the presentation layer (workflow/display.py).
    """

    doc_id: str
    doc_name: str
    section_path: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    excerpt: str = ""
    chunk_id: Optional[str] = None
    rects: list = []
    rect_source: Optional[str] = None
    file_path: Optional[str] = None
    page_count: Optional[int] = None
    # Fix 012 (2026-10-31, additive, Defect B): the 1-based evidence
    # position this reference corresponds to in the generation context
    # ('[Document n]' headers), so an inline [n] citation in the answer
    # resolves to exactly this passage.  None = not a context-cited
    # passage (e.g. a table source), rendered without a [n] badge.
    evidence_ref: Optional[int] = None


class SourceGroup(BaseModel):
    """Fix 012 (2026-10-31, Defect B): document-level source grouping.

    ONE top-level entry per canonical document (grouped by doc_id, never
    by display name, section, page, or chunk id).  Each supporting
    evidence passage from that document stays individually addressable in
    `evidence` (each keeps its own chunk_id, section, pages, excerpt, and
    [n] evidence_ref).  The three identifier spaces stay distinct:

    * group identity  == doc_id (canonical document)
    * evidence ref    == evidence_ref (context position / inline [n])
    * retriever chunk == chunk_id (Lakebase/UC chunk, bbox key)

    doc_name is the canonical stored file name; the readable display
    title is derived at render time (display.py) with an explicit
    fallback -- never a section number in place of a document title.
    """

    doc_id: str
    doc_name: str
    section_path: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence: list[SourceRef] = []


# ── Workflow State (TypedDict for LangGraph) ─────────────────────────────

class WorkflowState(TypedDict, total=False):
    """Shared state for the LangGraph workflow (spec Section 2.2).

    Uses total=False so nodes can return partial updates.
    The 'trace' field uses Annotated[list, add] so each node appends
    trace entries without overwriting previous ones.
    """

    user_query: str
    conversation_history: list[dict]
    intent: Intent
    plan: list[PlanStep]
    target_doc_ids: list[str]
    target_doc_names: list[str]
    evidence: list[EvidenceItem]
    metadata_result: Optional[dict]
    lookup_result: Optional[dict]
    comparison_result: Optional[dict]
    # Fix 010 (2026-10-29, additive): cross-turn action context.  The app
    # persists the pending_action returned by the graph on the assistant
    # history message and passes it back on the next invoke so the planner
    # can resolve acceptance phrasings ('sure compare them then') into a
    # concrete comparison of the exact documents previously recommended.
    # Never carried into SQL directly; document identities are re-resolved
    # against the canonical inventory by the comparison node.
    pending_action: Optional[dict]
    # Fix 012 (2026-10-31, additive, Defect D): canonical document names
    # (stored file_name values) backing the PREVIOUS assistant answer, or
    # None.  The app derives them from the last assistant history message
    # (source_groups / table_result / source_refs) and passes them back so
    # the planner can scope a follow-up ('what about the limits?') to the
    # actual preceding agreement when the follow-up itself names none.
    prior_answer_docs: Optional[list[str]]
    # Fix 012 (2026-10-31, additive, Defect B): document-grouped source
    # references rendered by the UI.  Parallel to source_refs (kept for
    # compatibility); every inline [n] citation resolves via evidence_ref.
    source_groups: Optional[list] = None
    # Fix 011 (2026-10-30, additive, Stage C): structured table answer
    # payload for table requests — {columns, rows, doc_name, section_path,
    # page_start, page_end, chunk_ids, row_count, continuation_note}.
    # Rendered by the UI as an actual table (st.dataframe); never routed
    # through the prose-only LLM answer template.
    table_result: Optional[dict]
    answer: Optional[str]
    source_refs: list[SourceRef]
    error: Optional[str]
    abstain: bool
    abstain_reason: Optional[str]
    trace: Annotated[list[dict], add]
