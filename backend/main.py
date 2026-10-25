"""FastAPI REST & Streaming Server for RoasterLens Esports Intelligence Platform.

Provides backend API endpoints connecting the conversational UI with the LangGraph state machine:
- POST /api/chat/stream: Server-Sent Events (SSE) streaming real-time step progress, status bar updates,
  and audit events to power the UI left sidebar, plus final report streaming to the right screen.
- POST /api/chat: Standard REST endpoint returning full report, telemetry, and audit trail.
- GET  /api/pipeline/steps: Metadata describing all execution steps for rendering the left sidebar checklist.
- GET  /api/sessions: Lists active conversation sessions and threads.
- GET  /api/session/{session_id}: Retrieves state, chat history, and audit trail for a specific session.
- DELETE /api/session/{session_id}: Clears a session.
- GET  /api/health: Service and LLM configuration health check.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.agent_graph import build_graph, create_resilient_gemini_model, normalize_model_name
from backend.intelligence_api import router as intelligence_router, register_investigation, _investigations
from backend.text_utils import extract_clean_text

import pathlib
env_path = pathlib.Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="RoasterLens — Esports Scouting & Multi-Roster Intelligence API",
    description="Conversational backend connecting LangGraph AI Brain with real-time UI step-streaming.",
    version="1.1.0"
)

# Enable CORS for frontend development servers (React, Next.js, Vite, etc.)
app.include_router(intelligence_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

session_checkpointer = MemorySaver()

# In-memory session catalog for UI session listing and history
_session_registry: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -------------------------------------------------------------------------
# Request & Response Models
# -------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="Natural language scouting query or conversational response")
    session_id: str | None = Field(default=None, description="Optional conversation session ID (thread)")


class StepMetadata(BaseModel):
    id: str
    title: str
    description: str
    order: int
    category: str


# Static pipeline steps specification for the left sidebar
PIPELINE_STEPS: list[dict[str, Any]] = [
    {
        "id": "extract_and_validate_intent",
        "title": "Intent Extraction & Validation",
        "description": "LLM Brain extracts target player, role, and requirements; checks context completeness.",
        "order": 1,
        "category": "intent"
    },
    {
        "id": "ask_user",
        "title": "Conversational Clarification",
        "description": "Engages user if initial requirements are missing or ambiguous before consuming API calls.",
        "order": 2,
        "category": "conversational"
    },
    {
        "id": "collect_opendota_evidence",
        "title": "OpenDota Evidence Collection",
        "description": "Consults live OpenAPI docs and fetches real match, hero, and career telemetry from OpenDota API.",
        "order": 3,
        "category": "data_collection"
    },
    {
        "id": "collect_contextual_evidence",
        "title": "Contextual Web Research",
        "description": "Dispatches live web search for recent tournament results, patch meta, and team roster context.",
        "order": 4,
        "category": "data_collection"
    },
    {
        "id": "synthesize_evidence",
        "title": "Telemetry Synthesis & Metric Aggregation",
        "description": "Aggregates winrates, KDA ratios, GPM/XPM averages, and hero pool depth into structured evidence.",
        "order": 5,
        "category": "synthesis"
    },
    {
        "id": "evaluate_candidate",
        "title": "Candidate Evaluation",
        "description": "LLM Brain performs dynamic feature selection, situational analysis, and role-specific impact.",
        "order": 6,
        "category": "evaluation"
    },
    {
        "id": "counter_scout_challenge",
        "title": "Counter-Scout Challenge",
        "description": "Independent Red-Team critique challenges preliminary evaluation, bias, and tilt vulnerabilities.",
        "order": 7,
        "category": "audit"
    },
    {
        "id": "final_evaluation",
        "title": "Final Intelligence Dossier",
        "description": "Reconciles evaluation and critique into definitive scouting recommendation with full Audit Trail.",
        "order": 8,
        "category": "reporting"
    }
]

STEP_PROGRESS_WEIGHTS: dict[str, int] = {
    "extract_and_validate_intent": 15,
    "ask_user": 100,
    "collect_opendota_evidence": 35,
    "collect_contextual_evidence": 50,
    "synthesize_evidence": 65,
    "evaluate_candidate": 80,
    "counter_scout_challenge": 90,
    "final_evaluation": 100
}


def _get_or_init_session(session_id: str | None) -> str:
    s_id = session_id or str(uuid4())
    if s_id not in _session_registry:
        _session_registry[s_id] = {
            "session_id": s_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "messages": [],
            "latest_phase": "initialized",
            "audit_trail": []
        }
    return s_id


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return {
        "status": "healthy",
        "service": "RoasterLens Esports Intelligence Platform",
        "version": "1.1.0",
        "llm_configured": bool(api_key),
        "tavily_search_configured": bool(os.getenv("TAVILY_API_KEY")),
        "model": normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
        "active_sessions_count": len(_session_registry)
    }


@app.get("/api/pipeline/steps")
def get_pipeline_steps() -> dict[str, Any]:
    """Returns the metadata of all pipeline steps for rendering the left sidebar checklist and status bar."""
    return {
        "status": "success",
        "total_steps": len(PIPELINE_STEPS),
        "steps": PIPELINE_STEPS
    }


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    """Lists all active scouting sessions/threads for the conversation history sidebar."""
    sessions = []
    for s_id, s_data in _session_registry.items():
        msgs = s_data.get("messages", [])
        last_msg = msgs[-1].get("content", "")[:120] if msgs else "New session"
        sessions.append({
            "session_id": s_id,
            "created_at": s_data.get("created_at"),
            "updated_at": s_data.get("updated_at"),
            "message_count": len(msgs),
            "preview": last_msg,
            "latest_phase": s_data.get("latest_phase")
        })
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return {"status": "success", "sessions": sessions}


@app.get("/api/session/{session_id}")
def get_session(session_id: str = Path(..., description="The session ID to retrieve")) -> dict[str, Any]:
    """Retrieves conversation history, state, and audit trail for a specific session."""
    if session_id not in _session_registry:
        raise HTTPException(404, f"Session {session_id} not found.")
    return {
        "status": "success",
        "session": _session_registry[session_id]
    }


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str = Path(..., description="The session ID to delete")) -> dict[str, Any]:
    """Clears a session and its checkpointer state."""
    if session_id in _session_registry:
        del _session_registry[session_id]
    return {"status": "success", "message": f"Session {session_id} deleted."}


@app.post("/api/chat")
async def chat_sync(body: ChatRequest) -> dict[str, Any]:
    """Standard REST endpoint: executes pipeline and returns full report and audit trail."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(503, "Set GEMINI_API_KEY in .env to enable the scouting agent.")

    session_id = _get_or_init_session(body.session_id)
    turn_id = str(uuid4())
    model_name = normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    model = create_resilient_gemini_model(model_name, api_key)
    graph = build_graph(model, checkpointer=session_checkpointer)

    # Record user message in session
    _session_registry[session_id]["messages"].append({
        "role": "user",
        "content": body.message,
        "timestamp": _utc_now()
    })
    _session_registry[session_id]["updated_at"] = _utc_now()

    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=body.message)],
                "user_request": body.message,
                "audit_trail": [],
                "turn_id": turn_id,
            },
            config={"recursion_limit": 50, "configurable": {"thread_id": session_id}},
        )
    except Exception as exc:
        raise HTTPException(502, f"Scouting Pipeline execution failed: {exc}") from exc

    final_report = extract_clean_text(result.get("final_report", ""))
    audit_trail = result.get("audit_trail", [])
    investigation_trace = result.get("investigation_trace", [])
    phase = result.get("phase", "completed")

    # Record assistant message in session
    _session_registry[session_id]["messages"].append({
        "role": "assistant",
        "content": final_report,
        "timestamp": _utc_now(),
        "phase": phase
    })
    _session_registry[session_id]["latest_phase"] = phase
    _session_registry[session_id]["audit_trail"].extend(audit_trail)

    inv_record = register_investigation(
        session_id=session_id,
        query=body.message,
        intent=result.get("intent", {}),
        final_report=final_report,
        audit_trail=audit_trail,
        investigation_trace=investigation_trace
    )

    return {
        "status": "success",
        "session_id": session_id,
        "turn_id": turn_id,
        "phase": phase,
        "needs_clarification": result.get("needs_clarification", False),
        "intent": result.get("intent", {}),
        "feature_selection": result.get("feature_selection", {}),
        "synthesized_evidence": result.get("synthesized_evidence", {}),
        "preliminary_evaluation": result.get("preliminary_evaluation"),
        "counter_scout_challenge": result.get("counter_scout_challenge"),
        "final_report": final_report,
        "audit_trail": audit_trail,
        "investigation_trace": investigation_trace,
        "candidates": inv_record.get("candidates", []),
        "model": model_name
    }


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint.
    
    Streams real-time step progress, status bar updates, and audit events to power
    the UI left sidebar, plus final result streaming for the main right screen.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(503, "Set GEMINI_API_KEY in .env to enable the scouting agent.")

    session_id = _get_or_init_session(body.session_id)
    turn_id = str(uuid4())
    model_name = normalize_model_name(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    model = create_resilient_gemini_model(model_name, api_key)
    graph = build_graph(model, checkpointer=session_checkpointer)

    # Record user message in session
    _session_registry[session_id]["messages"].append({
        "role": "user",
        "content": body.message,
        "timestamp": _utc_now()
    })
    _session_registry[session_id]["updated_at"] = _utc_now()

    async def sse_generator() -> AsyncGenerator[dict[str, str], None]:
        # 1. Pipeline Start Event
        yield {
            "event": "pipeline_init",
            "data": json.dumps({
                "session_id": session_id,
                "turn_id": turn_id,
                "user_message": body.message,
                "started_at": _utc_now()
            })
        }

        final_report = ""
        full_audit_trail: list[dict[str, Any]] = []
        full_investigation_trace: list[dict[str, Any]] = []
        last_phase = "initialized"
        accumulated_state: dict[str, Any] = {}

        try:
            # Stream LangGraph state transitions as nodes complete
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content=body.message)],
                    "user_request": body.message,
                    "audit_trail": [],
                    "turn_id": turn_id,
                },
                config={"recursion_limit": 50, "configurable": {"thread_id": session_id}},
                stream_mode="updates"
            ):
                for node_name, node_output in chunk.items():
                    if not isinstance(node_output, dict):
                        continue
                    accumulated_state.update(node_output)
                    progress = STEP_PROGRESS_WEIGHTS.get(node_name, 50)
                    last_phase = node_output.get("phase", node_name)
                    audit_items = node_output.get("audit_trail", [])
                    full_audit_trail.extend(audit_items)
                    trace_items = node_output.get("investigation_trace", [])
                    full_investigation_trace.extend(trace_items)

                    # Extract node-specific details for UI display
                    node_summary: dict[str, Any] = {}
                    if node_name == "extract_and_validate_intent":
                        node_summary = {"intent": node_output.get("intent"), "needs_clarification": node_output.get("needs_clarification")}
                    elif node_name == "collect_opendota_evidence":
                        node_summary = {"endpoints_fetched": [e.get("endpoint") for e in node_output.get("opendota_evidence", [])]}
                    elif node_name == "collect_contextual_evidence":
                        node_summary = {"results_count": len(node_output.get("contextual_evidence", []))}
                    elif node_name == "synthesize_evidence":
                        node_summary = node_output.get("synthesized_evidence", {})
                    elif node_name == "evaluate_candidate":
                        node_summary = {"preview": (node_output.get("preliminary_evaluation") or "")[:200] + "..."}
                    elif node_name == "counter_scout_challenge":
                        node_summary = {"preview": (node_output.get("counter_scout_challenge") or "")[:200] + "..."}
                    elif node_name in {"final_evaluation", "ask_user"}:
                        final_report = extract_clean_text(node_output.get("final_report", ""))

                    # 2. Emit Step Update with real tool explanations & decision trace
                    yield {
                        "event": "step_update",
                        "data": json.dumps({
                            "step_id": node_name,
                            "status": "completed" if node_name != "ask_user" else "waiting_for_user",
                            "progress": progress,
                            "timestamp": _utc_now(),
                            "summary": node_summary,
                            "audit_events": audit_items,
                            "trace_events": trace_items
                        })
                    }

            # Register investigation into persistent shared model with real shortlisted candidates and decision trace
            inv_record = register_investigation(
                session_id=session_id,
                query=body.message,
                intent=accumulated_state.get("intent", {}),
                final_report=extract_clean_text(final_report or accumulated_state.get("final_report", "")),
                audit_trail=full_audit_trail,
                investigation_trace=full_investigation_trace
            )

            # 3. Final Result Event for Right Screen
            yield {
                "event": "pipeline_complete",
                "data": json.dumps({
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "final_report": extract_clean_text(final_report or accumulated_state.get("final_report", "")),
                    "phase": last_phase,
                    "needs_clarification": accumulated_state.get("needs_clarification", False),
                    "audit_trail": full_audit_trail,
                    "investigation_trace": full_investigation_trace,
                    "candidates": inv_record.get("candidates", []),
                    "completed_at": _utc_now()
                })
            }

            # Store in session registry
            _session_registry[session_id]["messages"].append({
                "role": "assistant",
                "content": extract_clean_text(final_report or accumulated_state.get("final_report", "")),
                "timestamp": _utc_now(),
                "phase": last_phase
            })
            _session_registry[session_id]["latest_phase"] = last_phase
            _session_registry[session_id]["audit_trail"].extend(full_audit_trail)

        except Exception as exc:
            yield {
                "event": "pipeline_error",
                "data": json.dumps({
                    "error": str(exc),
                    "phase": last_phase,
                    "timestamp": _utc_now()
                })
            }

    return EventSourceResponse(sse_generator())
