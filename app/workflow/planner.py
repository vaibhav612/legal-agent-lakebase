"""Plan validation for multi-step queries (spec Section 2.10).

Validates that a multi-step plan:
  1. Contains only known capability nodes.
  2. Does not exceed MAX_PLAN_STEPS (5).
  3. Has valid parameters (e.g. target_field is allowlisted).

If validation fails, returns an error describing the problem.
"""

from __future__ import annotations

from pydantic import ValidationError

from .state import (
    ALLOWED_FIELDS,
    ALLOWED_NODES,
    MAX_PLAN_STEPS,
    PlanStep,
)
from .traces import make_trace


class PlanValidationError(Exception):
    """Raised when a plan fails validation."""

    def __init__(self, message: str, step_id: int | None = None):
        self.step_id = step_id
        super().__init__(message)


def validate_plan(plan: list[dict] | list[PlanStep]) -> list[PlanStep]:
    """Validate a multi-step plan and return a list of PlanStep objects.

    Raises PlanValidationError if the plan is invalid.
    """
    if not plan:
        raise PlanValidationError("Plan is empty.")

    if len(plan) > MAX_PLAN_STEPS:
        raise PlanValidationError(
            f"Plan has {len(plan)} steps; maximum allowed is {MAX_PLAN_STEPS}. "
            "Consider breaking the question into smaller parts."
        )

    steps: list[PlanStep] = []
    for i, raw_step in enumerate(plan):
        try:
            if isinstance(raw_step, PlanStep):
                step = raw_step
            else:
                step = PlanStep(**raw_step)
        except ValidationError as e:
            raise PlanValidationError(
                f"Step {i + 1} is invalid: {e}", step_id=i + 1
            ) from e

        if step.node not in ALLOWED_NODES:
            raise PlanValidationError(
                f"Step {i + 1} targets unknown node '{step.node}'. "
                f"Allowed nodes: {ALLOWED_NODES}",
                step_id=i + 1,
            )

        if step.target_field is not None and step.target_field not in ALLOWED_FIELDS:
            raise PlanValidationError(
                f"Step {i + 1} has invalid target_field '{step.target_field}'. "
                f"Allowed fields: {ALLOWED_FIELDS}",
                step_id=i + 1,
            )

        steps.append(step)

    return steps


def plan_validation_node(state: dict) -> dict:
    """LangGraph node that validates a multi-step plan.

    If valid, stores the validated plan.  If invalid, sets an error.
    """
    raw_plan = state.get("plan", [])
    try:
        validated = validate_plan(raw_plan)
        return {
            "plan": validated,
            "error": None,
            "trace": [
                make_trace(
                    "plan_validation",
                    action="validated",
                    route="multi_step_stub",
                    reason=f"{len(validated)} steps approved",
                    num_steps=len(validated),
                )
            ],
        }
    except PlanValidationError as e:
        return {
            "error": str(e),
            "plan": [],
            "trace": [
                make_trace(
                    "plan_validation",
                    action="rejected",
                    route="final_answer",
                    reason=str(e),
                    num_steps=len(raw_plan),
                )
            ],
        }
