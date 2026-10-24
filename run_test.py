"""Interactive CLI Test Runner for RoasterLens Esports Intelligence Pipeline.

Usage:
    python run_test.py
    python run_test.py --prompt "Your custom prompt here"
"""

import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage
from backend.agent_graph import build_graph, create_resilient_gemini_model


def run_pipeline(prompt: str):
    print("\n" + "=" * 70)
    print(" ROASTERLENS — ESPORTS INTELLIGENCE PIPELINE TEST")
    print("=" * 70)
    print(f"\n[USER INPUT]:\n  \"{prompt}\"\n")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set in .env! Please set your Gemini API key.")
        return

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if "3.5" in model_name:
        model_name = "gemini-2.5-flash"

    print(f"[INITIALIZING] Resilient Gemini Brain ({model_name} with fallback chain)...")
    model = create_resilient_gemini_model(model_name, api_key)

    print("[INITIALIZING] Compiling LangGraph State Machine...")
    graph = build_graph(model)

    print("\n[EXECUTING] LangGraph States...")
    try:
        result = graph.invoke({
            "messages": [HumanMessage(content=prompt)],
            "user_request": prompt,
            "audit_trail": [],
            "turn_id": "cli_test_turn"
        })
    except Exception as e:
        print(f"\n[PIPELINE ERROR]: {e}")
        return

    # Check state results
    print("\n" + "-" * 70)
    print(" PIPELINE EXECUTION SUMMARY")
    print("-" * 70)
    print(f"End Phase: {result.get('phase')}")
    print(f"Needs Clarification: {result.get('needs_clarification')}")

    intent = result.get("intent", {})
    print(f"\n1. EXTRACTED INTENT:")
    print(f"   - User Goal: {intent.get('user_goal')}")
    print(f"   - Target Player: {intent.get('player_target')}")
    print(f"   - Target Role: {intent.get('target_role')}")
    print(f"   - Strategic Priorities: {intent.get('strategic_priorities')}")
    print(f"   - Sufficiency Decision: {intent.get('is_sufficient_to_proceed')}")

    if result.get("needs_clarification"):
        print(f"\n2. CONVERSATIONAL CLARIFICATION (State: ask_user):")
        print(f"   {result.get('final_report')}")
    else:
        opendota_evidence = result.get("opendota_evidence", [])
        print(f"\n2. OPENDOTA EVIDENCE COLLECTION:")
        print(f"   - Total Endpoints Queried: {len(opendota_evidence)}")
        for ep in opendota_evidence:
            endpoint_name = ep.get("endpoint")
            source = ep.get("result", {}).get("source", "OpenDota")
            print(f"     * {endpoint_name} (Source: {source})")

        features = result.get("feature_selection", {})
        print(f"\n3. DYNAMIC FEATURE SELECTION (No Hardcoding):")
        print(f"   - Selected Metrics: {features.get('selected_metrics')}")
        print(f"   - Role Indicators: {features.get('role_indicators')}")
        print(f"   - Rationale: {features.get('feature_selection_rationale')}")

        print(f"\n4. PRELIMINARY EVALUATION PREVIEW:")
        prelim = result.get("preliminary_evaluation", "")
        print(f"   {prelim[:300]}...\n")

        print(f"\n5. COUNTER-SCOUT CHALLENGE PREVIEW:")
        counter = result.get("counter_scout_challenge", "")
        print(f"   {counter[:300]}...\n")

        print("\n" + "=" * 70)
        print(" FINAL REASONED SCOUTING INTELLIGENCE REPORT")
        print("=" * 70)
        print(result.get("final_report"))

    print("\n" + "=" * 70)
    print(" STRUCTURED AUDIT TRAIL LOG")
    print("=" * 70)
    for i, event in enumerate(result.get("audit_trail", []), 1):
        print(f"[{i}] {event.get('timestamp')} | Phase: {event.get('phase')} | Agent: {event.get('agent')}")
        print(f"    Action: {event.get('action')}")
        print(f"    Rationale: {event.get('decision_rationale')}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoasterLens Test Runner")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt to test")
    args = parser.parse_args()

    if args.prompt:
        run_pipeline(args.prompt)
    else:
        print("\nSelect a test case to execute:")
        print("1. Full Scouting Workflow: 'Scout Puppey for an IGL role with strong vision control and mid-game decision making.'")
        print("2. Core Carry Workflow: 'Evaluate Yatoro for our Pos 1 Carry slot focusing on late-game farm and stability.'")
        print("3. Conversational Clarification: 'Hello, I want to scout a player for my team.'")
        print("4. Enter custom prompt")
        
        choice = input("\nEnter choice [1-4] (default 1): ").strip()
        if choice == "2":
            p = "Evaluate Yatoro for our Pos 1 Carry slot focusing on late-game farm and stability."
        elif choice == "3":
            p = "Hello, I want to scout a player for my team."
        elif choice == "4":
            p = input("Enter your custom prompt: ").strip()
        else:
            p = "Scout Puppey for an IGL role with strong vision control and mid-game decision making."

        run_pipeline(p)
