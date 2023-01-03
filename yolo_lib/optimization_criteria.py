from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest import TestCase


@dataclass
class OptimizationCriteria:
    """
    References an entry in a performance metric dictionary that is a "main performance metric".
    Epochs that maximize a main performance metric are stored.
    dAP is typically used as a main performance metric
    """

    short_name: str
    path: List[str]
    minimize: bool = False

    def get_score(self, epoch_log_object: Dict[str, Any]) -> Optional[float]:
        # Recursively traverse path
        node = epoch_log_object
        for key in self.path:
            if key not in node:
                return None
            node = node[key]

        # Check that the end node is a number
        return float(node)

    def get_best_in_list(self, training_log: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        best_score = None
        best_epoch_log_object = None
        none_count = 0
        for crt_epoch_log_obj in training_log:
            # Check if best_score should be updated
            # 1: best_score is None and crt has a score.
            # 2: best_score is something and crt has a score and crt score is better
            crt_score = self.get_score(crt_epoch_log_obj)
            none_count += int(crt_score is None)

            # Update best score
            is_crt_better = self.compare_scores(crt_score, best_score)
            if is_crt_better:
                best_score = crt_score
                best_epoch_log_object = crt_epoch_log_obj

        # Print none_count if nonzero
        if none_count > 0:
            print(f"WARNING: OptimizationCriteria.get_best_in_list(): training_log contained {none_count} items with missing {self.short_name}.")

        # Return best log object
        return best_epoch_log_object

    def compare_scores(self, score_a: Optional[float], score_b: Optional[float]) -> Optional[bool]:
        """Returns true if score_a is best, false if score_b is best, and none if neither is better"""
        if score_a is None and score_b is None:
            return None
        elif score_a is None:
            return False
        elif score_b is None:
            return True
        else:
            if score_a == score_b:
                return None
            elif self.minimize:
                return score_a < score_b
            else:
                return score_a > score_b

    def is_crt_best(self, training_log: List[Dict[str, Any]], crt_epoch_log_object: Dict[str, Any]) -> Optional[bool]:
        best_in_list = self.get_best_in_list(training_log)
        best_score_in_list = self.get_score(best_in_list) if best_in_list is not None else None
        crt_score = self.get_score(crt_epoch_log_object)
        is_crt_better = self.compare_scores(crt_score, best_score_in_list)
        return is_crt_better


class OptimizationCriteriaTests(TestCase):
    def test_maximizer_score_comparison(self):
        maximizer = OptimizationCriteria([], "name", minimize=False)
        self.assertEqual(maximizer.compare_scores(0.0, 0.0), None)
        self.assertEqual(maximizer.compare_scores(0.0, 2.0), False)
        self.assertEqual(maximizer.compare_scores(0.0, None), True)
        self.assertEqual(maximizer.compare_scores(1.0, 0.0), True)
        self.assertEqual(maximizer.compare_scores(1.0, 2.0), False)
        self.assertEqual(maximizer.compare_scores(1.0, None), True)
        self.assertEqual(maximizer.compare_scores(2.0, 0.0), True)
        self.assertEqual(maximizer.compare_scores(2.0, 2.0), None)
        self.assertEqual(maximizer.compare_scores(2.0, None), True)
        self.assertEqual(maximizer.compare_scores(None, 0.0), False)
        self.assertEqual(maximizer.compare_scores(None, 2.0), False)
        self.assertEqual(maximizer.compare_scores(None, None), None)

    def test_minimizer_score_comparison(self):
        minimizer = OptimizationCriteria([], "name", minimize=True)
        self.assertEqual(minimizer.compare_scores(0.0, 0.0), None)
        self.assertEqual(minimizer.compare_scores(0.0, 2.0), True)
        self.assertEqual(minimizer.compare_scores(0.0, None), True)
        self.assertEqual(minimizer.compare_scores(1.0, 0.0), False)
        self.assertEqual(minimizer.compare_scores(1.0, 2.0), True)
        self.assertEqual(minimizer.compare_scores(1.0, None), True)
        self.assertEqual(minimizer.compare_scores(2.0, 0.0), False)
        self.assertEqual(minimizer.compare_scores(2.0, 2.0), None)
        self.assertEqual(minimizer.compare_scores(2.0, None), True)
        self.assertEqual(minimizer.compare_scores(None, 0.0), False)
        self.assertEqual(minimizer.compare_scores(None, 2.0), False)
        self.assertEqual(minimizer.compare_scores(None, None), None)
