"""Hermetic Unit & Integration Tests for RoasterLens Pipeline (Unittest compatible)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, HumanMessage

from backend.opendota_service import (
    search_opendota_docs,
    call_opendota_endpoint,
    search_contextual_intel,
)
from backend.agent_graph import build_graph


class MockScoutingBrain:
    """Mock LLM Brain returning realistic esports domain responses for fast hermetic verification."""
    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        prompt_text = " ".join(str(m.content) for m in messages).lower()

        # 1. Intent validation node
        user_content = str(messages[-1].content).lower() if messages else ""
        if "extract the user's intent" in prompt_text or "intent validator" in prompt_text:
            if "hello" in user_content or "help me" in user_content:
                return AIMessage(content=json.dumps({
                    "user_goal": "General inquiry",
                    "player_target": None,
                    "target_role": None,
                    "strategic_priorities": [],
                    "competitive_tier": None,
                    "is_sufficient_to_proceed": False,
                    "clarification_question": "Welcome to RoasterLens! Which player handle, competitive role (e.g., IGL, Carry, Support), or strategic roster need would you like to scout?",
                    "rationale": "Prompt lacks role, player, and competitive criteria."
                }))
            return AIMessage(content=json.dumps({
                "user_goal": "Scout an experienced In-Game Leader (IGL) with strong shot-calling",
                "player_target": "Puppey",
                "target_role": "In-Game Leader (IGL) / Pos 5 Hard Support",
                "strategic_priorities": ["Mid-game shot calling", "Vision control", "Hero pool adaptability"],
                "competitive_tier": "Tier-1 International",
                "is_sufficient_to_proceed": True,
                "clarification_question": None,
                "rationale": "Clear candidate name, role, and explicit strategic priorities provided."
            }))

        # 2. OpenDota docs planning
        if "opendota intelligence specialist" in prompt_text:
            return AIMessage(content=json.dumps({
                "docs_query": "search recent matches totals heroes",
                "selected_endpoints": ["/search", "/players/{account_id}/recentMatches", "/players/{account_id}/heroes", "/players/{account_id}/totals"],
                "reasoning": "Selected endpoints give full visibility into situational stats, career vision, and hero versatility."
            }))

        # 3. OpenDota sufficiency check
        if "opendota data quality officer" in prompt_text:
            return AIMessage(content=json.dumps({
                "is_sufficient": True,
                "data_quality_assessment": "Retrieved 4 endpoints covering profile, 20 recent matches, hero pool, and totals.",
                "alternative_endpoints_needed": [],
                "reasoning": "High data integrity. Sufficient to evaluate warding, teamfight participation, and winrates."
            }))

        # 4. Contextual planning
        if "esports context & competitive intel" in prompt_text:
            return AIMessage(content=json.dumps({
                "search_queries": ["Puppey Dota 2 tournament results 2024", "Dota 2 IGL meta patch 7.37"],
                "target_features": ["LAN tournament resilience", "Shot calling consistency", "Draft versatility"],
                "reasoning": "Context needed to corroborate statistical trends with meta patch demands."
            }))

        # 6. Preliminary evaluation
        if "principal esports recruitment director" in prompt_text:
            return AIMessage(content=(
                "PRELIMINARY SCOUTING EVALUATION: Clement 'Puppey' Ivanov\n"
                "- Role: In-Game Leader (IGL) / Pos 5 Support\n"
                "- Situational Metrics: 73.4% teamfight participation, 14.4 observer wards/match.\n"
                "- Shot-Calling Impact: Elite micro-hero control translates to early map tempo.\n"
                "- Preliminary Fit: Tier-1 Captain grade (A-)."
            ))

        # 5. Evidence synthesis & feature selection
        if "chief data scientist" in prompt_text:
            return AIMessage(content=json.dumps({
                "selected_metrics": ["Observer Wards Placed/Min (0.42)", "Teamfight Participation Rate (73.4%)", "Hero Pool Versatility (Chen 70% WR, Enchantress 65.8% WR)"],
                "role_indicators": ["Mid-Game Objective Shot-Calling (Roshan contest rate)", "Deficit Recovery Composure"],
                "chemistry_factors": ["Vocal Captaincy Protocol", "Cross-Roster Integration"],
                "feature_selection_rationale": "Selected vision control and teamfight participation as key quantitative proxies for IGL impact."
            }))

        # 7. Counter-Scout Challenge
        if "adversarial counter-scout" in prompt_text or "red-team analyst" in prompt_text:
            return AIMessage(content=(
                "COUNTER-SCOUT CHALLENGE & RED-TEAM AUDIT:\n"
                "1. Sample Bias: High win rate on Chen/Enchantress masks vulnerability when draft bans force standard lane supports.\n"
                "2. Proxy Limitation: High ward volume does not guarantee superior late-game teamfight communication under high tilt pressure.\n"
                "3. Tilt Risk: Documented struggles in recent Tier-1 qualifiers when playing from 10k gold deficits.\n"
                "4. Alternative Contender: Consider Seleri (Gaimin Gladiators) or Fly (Nouns) who offer higher early lane aggression."
            ))

        # 8. Final Report
        return AIMessage(content=(
            "# EXECUTIVE SCOUTING INTELLIGENCE DOSSIER\n\n"
            "## 1. EXECUTIVE SCOUTING RECOMMENDATION & ROSTER FIT\n"
            "**Candidate:** Clement 'Puppey' Ivanov | **Role:** Pos 5 IGL | **Recommendation:** Sign (Grade A-)\n\n"
            "## 2. SITUATIONAL & ROLE-SPECIFIC METRICS BREAKDOWN\n"
            "- **Teamfight Participation:** 73.4% (Elite situational benchmark)\n"
            "- **Vision Control:** 14.4 Observer / 18.0 Sentry per game (0.41/min)\n"
            "- **Formula:** IGL_Impact = 0.4*(TF_Part) + 0.3*(Vision_Ratio) + 0.3*(Pool_Depth)\n\n"
            "## 3. COUNTER-SCOUT RECONCILIATION & RISK MITIGATION\n"
            "The Counter-Scout rightly identified micro-hero ban dependency. We mitigate this by establishing a secondary draft pool (Oracle, Treant).\n\n"
            "## 4. MULTI-ROSTER & LONG-TERM SYNERGY TRAJECTORY\n"
            "Proven leadership across a decade of tier-1 rosters ensures immediate operational stability.\n\n"
            "## 5. ACTIONABLE RECRUITMENT TRIAL PLAN\n"
            "Phase 1 (14 days scrim trial) -> Phase 2 (Tier-2 qualifier event) -> Phase 3 (Contract execution).\n\n"
            "## 6. DATA AUDIT TRAIL SUMMARY\n"
            "Data verified from OpenDota API (/players/87278707/recentMatches, /totals, /heroes) and Competitive Intel."
        ))


class TestRoasterLensPipeline(unittest.TestCase):

    def test_01_opendota_docs_search(self):
        res = search_opendota_docs("recent matches player totals")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["matches_count"], 0)
        endpoints = [e["path"] for e in res["endpoints"]]
        self.assertTrue(any("/players/{account_id}/recentMatches" in ep for ep in endpoints))

    def test_02_opendota_endpoint_dispatch(self):
        search_res = call_opendota_endpoint("/search", {"q": "Puppey"})
        self.assertIn("status", search_res)
        self.assertEqual(search_res["endpoint"], "/search")

    def test_03_contextual_intel(self):
        intel = search_contextual_intel("Puppey IGL shot-calling Dota 2")
        self.assertIn("status", intel)
        self.assertEqual(intel["query"], "Puppey IGL shot-calling Dota 2")

    def test_04_langgraph_pipeline_conversational_clarification(self):
        mock_brain = MockScoutingBrain()
        graph = build_graph(mock_brain)

        result = graph.invoke({
            "messages": [HumanMessage(content="hello, help me")],
            "user_request": "hello, help me",
            "audit_trail": [],
            "turn_id": "test_turn_1"
        })

        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["phase"], "waiting_for_user_input")
        self.assertIn("Welcome to RoasterLens", result["final_report"])
        self.assertEqual(len(result["audit_trail"]), 2)
        self.assertEqual(result["audit_trail"][0]["action"], "extract_and_validate_intent")
        self.assertEqual(result["audit_trail"][1]["action"], "ask_user")

    def test_05_langgraph_pipeline_full_scouting_execution(self):
        mock_brain = MockScoutingBrain()
        graph = build_graph(mock_brain)

        prompt = "Scout Puppey for our Dota 2 roster as an In-Game Leader (IGL) with strong mid-round shot-calling and vision control."
        result = graph.invoke({
            "messages": [HumanMessage(content=prompt)],
            "user_request": prompt,
            "audit_trail": [],
            "turn_id": "test_turn_2"
        })

        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["phase"], "completed")

        audit_actions = [e["action"] for e in result["audit_trail"]]
        expected_sequence = [
            "extract_and_validate_intent",
            "collect_opendota_evidence",
            "collect_contextual_evidence",
            "synthesize_evidence",
            "evaluate_candidate",
            "counter_scout_challenge",
            "final_evaluation"
        ]
        for action in expected_sequence:
            self.assertIn(action, audit_actions)

        self.assertGreaterEqual(len(result["opendota_evidence"]), 1)
        self.assertGreaterEqual(len(result["contextual_evidence"]), 1)
        self.assertIn("selected_metrics", result["feature_selection"])
        self.assertIn("PRELIMINARY SCOUTING EVALUATION", result["preliminary_evaluation"])
        self.assertIn("COUNTER-SCOUT CHALLENGE", result["counter_scout_challenge"])
        self.assertIn("EXECUTIVE SCOUTING INTELLIGENCE DOSSIER", result["final_report"])


if __name__ == "__main__":
    unittest.main()
