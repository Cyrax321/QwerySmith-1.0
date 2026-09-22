#!/usr/bin/env python3
"""
tests/test_dialogue_state.py -- Unit tests for DialogueStateTracker
"""

import unittest
from harness.conversation import DialogueStateTracker


class TestDialogueStateTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = DialogueStateTracker(max_context_turns=3)
        self.session_id = "test_user_session"
        self.db_name = "store.db"

    def test_followup_detection(self):
        # Initial turn is NOT a follow up
        self.assertFalse(self.tracker.is_followup("Which products are in stock?", self.session_id))

        # Record turn 1
        self.tracker.record_turn(
            session_id=self.session_id,
            db_name=self.db_name,
            question="Which products are in stock?",
            sql="SELECT name FROM products WHERE stock > 0;",
            columns=["name"],
            rows=[("MacBook Pro",), ("Headphones",)],
        )

        # Follow-ups should be detected
        self.assertTrue(self.tracker.is_followup("and which of them cost over $500?", self.session_id))
        self.assertTrue(self.tracker.is_followup("what about in Germany?", self.session_id))
        self.assertTrue(self.tracker.is_followup("only show those from Japan", self.session_id))

        # Brand new independent question is NOT detected as follow-up
        self.assertFalse(self.tracker.is_followup("List all customer names and emails.", self.session_id))

    def test_context_prompt_construction(self):
        self.tracker.record_turn(
            session_id=self.session_id,
            db_name=self.db_name,
            question="Show top platinum customers",
            sql="SELECT name FROM customers WHERE tier = 'Platinum';",
            columns=["name"],
            rows=[("Sophia",), ("Kenji",)],
        )

        ctx = self.tracker.build_context_prompt(self.session_id, "and how much did they spend?")
        self.assertIn("Show top platinum customers", ctx)
        self.assertIn("Sophia", ctx)
        self.assertIn("anaphoric follow-up", ctx)


if __name__ == "__main__":
    unittest.main()
