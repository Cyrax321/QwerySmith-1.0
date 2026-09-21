#!/usr/bin/env python3
import unittest
from harness.conversation import DialogueStateTracker


class TestConversationSerialization(unittest.TestCase):
    def test_export_and_restore_session(self):
        tracker = DialogueStateTracker()
        tracker.record_turn(
            session_id="sess_1",
            db_name="store.db",
            question="List products",
            sql="SELECT * FROM products;",
            columns=["id", "name"],
            rows=[(1, "Laptop")],
        )
        exported = tracker.export_session("sess_1")
        self.assertIsNotNone(exported)
        self.assertEqual(len(exported["turns"]), 1)

        tracker_restored = DialogueStateTracker()
        tracker_restored.restore_session(exported)
        self.assertIn("sess_1", tracker_restored.sessions)
        self.assertEqual(len(tracker_restored.sessions["sess_1"].turns), 1)


if __name__ == "__main__":
    unittest.main()
