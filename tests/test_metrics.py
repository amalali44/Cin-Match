import unittest

from recsys.metrics import ranking_metrics


class RankingMetricTests(unittest.TestCase):
    def test_hand_calculated_macro_metrics(self):
        result = ranking_metrics({1: [10, 20], 2: [40]}, {1: {10, 30}, 2: {40}}, 2)
        self.assertEqual(result, {"precision_at_k": 0.5, "recall_at_k": 0.75, "evaluated_users": 2})

    def test_fixed_k_penalizes_short_lists(self):
        result = ranking_metrics({1: [10]}, {1: {10}}, 5)
        self.assertEqual(result["precision_at_k"], 0.2)
        self.assertEqual(result["recall_at_k"], 1.0)

    def test_unreachable_positives_remain_in_recall_denominator(self):
        result = ranking_metrics({1: [10]}, {1: {10, 999}}, 1)
        self.assertEqual(result["recall_at_k"], 0.5)

    def test_missing_predictions_count_as_zero(self):
        self.assertEqual(ranking_metrics({}, {1: {10}}, 2)["recall_at_k"], 0)

    def test_no_relevant_users_is_explicitly_undefined(self):
        result = ranking_metrics({1: [10]}, {1: set()}, 2)
        self.assertEqual(result, {"precision_at_k": None, "recall_at_k": None, "evaluated_users": 0})

    def test_rejects_duplicates_and_invalid_k(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ranking_metrics({1: [10, 10]}, {1: {10}}, 2)
        with self.assertRaises(ValueError):
            ranking_metrics({}, {}, 0)
