"""FastAPI endpoint verification test suite for RoasterLens conversational backend."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import (
    health,
    get_pipeline_steps,
    list_sessions,
    get_session,
    delete_session,
    _get_or_init_session,
)


class TestFastAPI(unittest.TestCase):
    def test_01_health_endpoint(self):
        data = health()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "RoasterLens Esports Intelligence Platform")
        self.assertIn("llm_configured", data)
        self.assertIn("model", data)

    def test_02_pipeline_steps_endpoint(self):
        res = get_pipeline_steps()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_steps"], 8)
        step_ids = [s["id"] for s in res["steps"]]
        self.assertIn("extract_and_validate_intent", step_ids)
        self.assertIn("collect_opendota_evidence", step_ids)
        self.assertIn("collect_contextual_evidence", step_ids)
        self.assertIn("synthesize_evidence", step_ids)
        self.assertIn("evaluate_candidate", step_ids)
        self.assertIn("counter_scout_challenge", step_ids)
        self.assertIn("final_evaluation", step_ids)
        self.assertIn("ask_user", step_ids)

    def test_03_session_lifecycle(self):
        # Initialize a test session
        session_id = _get_or_init_session("test_session_123")
        self.assertEqual(session_id, "test_session_123")

        # Retrieve session
        session_data = get_session("test_session_123")
        self.assertEqual(session_data["status"], "success")
        self.assertEqual(session_data["session"]["session_id"], "test_session_123")

        # List sessions
        sessions_list = list_sessions()
        self.assertGreaterEqual(len(sessions_list["sessions"]), 1)
        found = any(s["session_id"] == "test_session_123" for s in sessions_list["sessions"])
        self.assertTrue(found)

        # Delete session
        del_res = delete_session("test_session_123")
        self.assertEqual(del_res["status"], "success")


if __name__ == "__main__":
    unittest.main()
