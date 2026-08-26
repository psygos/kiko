import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fable_behavior_trace as trace


class FableBehaviorTraceTests(unittest.TestCase):
    def test_checked_in_trace_is_exactly_current(self):
        with trace.DEFAULT_FIXTURE.open("rb") as source:
            self.assertEqual(source.read(), trace.canonical_bytes())

    def test_trace_has_one_record_for_every_distinct_fable_act(self):
        generated = trace.build_trace()
        names = [record["name"] for record in generated["acts"]]
        self.assertEqual(len(names), 24)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            {record["pet_reaction"] for record in generated["acts"]
             if record["pet_reaction"] is not None},
            {"boop", "play", "affection"},
        )

    def test_fixture_has_no_duplicate_json_keys(self):
        with trace.DEFAULT_FIXTURE.open(encoding="utf-8") as source:
            parsed = json.load(source, object_pairs_hook=trace._unique_object)
        self.assertEqual(parsed, trace.build_trace())


if __name__ == "__main__":
    unittest.main()
