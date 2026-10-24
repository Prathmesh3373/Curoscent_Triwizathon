"""LangGraph CLI / Studio entrypoint for RoasterLens Esports Intelligence Agent."""

from __future__ import annotations

import os
from dotenv import load_dotenv

from backend.agent_graph import build_graph, create_resilient_gemini_model, normalize_model_name

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not _api_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env before starting LangGraph Studio.")

_raw_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_model_name = normalize_model_name(_raw_model)

# Create resilient model with multi-tier fallback chain and rate-limit recovery
_model = create_resilient_gemini_model(_model_name, _api_key)

# Compile LangGraph state machine with full LLM Brain orchestration
graph = build_graph(_model)
