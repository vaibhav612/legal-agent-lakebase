"""Structured execution traces for the LangGraph workflow.

Each node appends a TraceEntry to the shared state's 'trace' field.
This provides sufficient information to inspect the selected route
and why it was selected, without requiring external logging.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from .state import Intent


class TraceEntry(BaseModel):
    """A single execution trace entry recorded by a graph node."""

    step: int = 0
    node: str
    action: str = ""
    route: Optional[str] = None
    reason: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return self.model_dump()


def make_trace(
    node: str,
    action: str = "",
    route: str | None = None,
    reason: str | None = None,
    **details: Any,
) -> dict:
    """Create a trace dict for appending to the state's 'trace' field.

    Usage inside a node:
        return {"trace": [make_trace("query_planner", "classified", route="hybrid_qa")]}
    """
    entry = TraceEntry(node=node, action=action, route=route, reason=reason, details=details)
    return entry.to_dict()


def format_trace(trace: list[dict]) -> str:
    """Format the execution trace as a human-readable string for debugging."""
    if not trace:
        return "(no trace entries)"
    lines = []
    for i, entry in enumerate(trace, 1):
        node = entry.get("node", "?")
        action = entry.get("action", "")
        route = entry.get("route", "")
        reason = entry.get("reason", "")
        parts = [f"  {i}. [{node}]"]
        if action:
            parts.append(f" {action}")
        if route:
            parts.append(f" → route={route}")
        if reason:
            parts.append(f" ({reason})")
        lines.append("".join(parts))
    return "\n".join(lines)
