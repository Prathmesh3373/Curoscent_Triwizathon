"""Structured Investigation Decision & Tool Call Trace Models for RosterLens.

Provides concrete schemas for recording and tracking:
- Why tools were called (evidence gaps, goals).
- Exact tool inputs and outputs.
- Real contribution/impact on evaluation.
- Execution status, execution timing, errors.
- Conditional branching decisions (why another tool was called vs skipped).
- Evidence grounding: distinguishing OpenDota data, Python calculations, LLM evaluations, and uncertainties.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolCallExplanation(BaseModel):
    tool_name: str = Field(description="Name of tool or API invoked (e.g. OpenDota /players/{id}/recentMatches)")
    step_number: int = Field(default=1, description="Sequential step index")
    step_title: str = Field(description="High-level operation name")
    reason: str = Field(description="Task objective or evidence gap justifying the invocation")
    input_params: dict[str, Any] = Field(default_factory=dict, description="Parameters sent to tool (e.g. account_id, filters)")
    result_summary: str = Field(description="Concise summary of actual returned data")
    impact: str = Field(description="Contribution to evaluation or why further investigation was required")
    status: Literal["completed", "failed", "skipped", "running"] = Field(default="completed")
    latency_ms: int = Field(default=0, description="Execution time in milliseconds")
    error: str | None = Field(default=None, description="Error message if failed")
    data_source_type: Literal["opendota_api", "python_metric", "web_research", "llm_reasoning"] = Field(
        default="opendota_api",
        description="Origin of data"
    )


class InvestigationTraceEvent(BaseModel):
    investigation_id: str
    step_id: str
    step_order: int
    title: str
    node_name: str
    decision_summary: str
    evidence_gap_or_objective: str
    status: Literal["completed", "failed", "skipped", "waiting_for_user"] = "completed"
    tool_calls: list[ToolCallExplanation] = Field(default_factory=list)
    conditional_decision: str | None = Field(default=None, description="Explanation of why an alternative branch/additional retrieval occurred or was skipped")
    duration_ms: int = 0
    timestamp: str
    error_details: str | None = None
