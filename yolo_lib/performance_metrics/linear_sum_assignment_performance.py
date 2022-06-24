import unittest
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from yolo_lib.data.dataclasses import DetectionBlock, AnnotationBlock, Annotation
from yolo_lib.performance_metrics.base_performance_metric import BasePerformanceMetric
from .matching import Matching, get_num_clusters



class LinSumPerformance:
    """
    Performance metrics where the detection-to-target
    matching is based on minimizing a linear sum assignment
    """
    def __reset_hook__(self, sub_state: Dict[str, Any]):
        pass

    def __increment_hook__(
        self,
        matched_detections: DetectionBlock,
        matched_annotations: AnnotationBlock,
        matching: Matching,
        sub_state: Dict[str, Any]
    ):
        pass

    def __finalize_hook__(self, sub_state: Dict[str, Any]):
        pass

    def __init__(
        self,
        match_distance_threshold: float,
        duplicate_distance_threshold: float,
        num_confidence_thresholds: int,
    ):
        self.match_distance_threshold = torch.tensor(match_distance_threshold, requires_grad=False)
        self.duplicate_distance_threshold = torch.tensor(duplicate_distance_threshold, requires_grad=False)
        self.confidence_thresholds = [i / (num_confidence_thresholds+1) for i in range(1, num_confidence_thresholds + 1)]
        self.reset()

    def reset(self):
        self.sub_states = []
        for _ in self.confidence_thresholds:
            crt_substate = {
                "tp_count": 0,
                "fp_count": 0,
                "fn_count": 0,
                "relaxed_tp_count": 0,
                "relaxed_fp_count": 0,
                "match_distance_sum": 0.0,
            }
            self.__reset_hook__(crt_substate)
            self.sub_states.append(crt_substate)

    def increment_substate(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
        sub_state: Dict[str, Any],
    ):
        assert isinstance(detections, DetectionBlock)
        assert isinstance(annotations, AnnotationBlock)
        assert isinstance(sub_state, dict)

        # No detections
        if detections.size == 0:
            sub_state["fn_count"] += annotations.size
            return

        # No annotations
        if annotations.size == 0:
            sub_state["fp_count"] += detections.size
            sub_state["relaxed_fp_count"] += get_num_clusters(detections.center_yx, self.duplicate_distance_threshold)
            return

        # Get annotation-detection matching
        matching = Matching.get_matching(
            detections,
            annotations,
            self.match_distance_threshold,
            self.duplicate_distance_threshold
        )

        # Increment position based performance metric counters
        sub_state["tp_count"] += int(matching.num_matchings)
        sub_state["fp_count"] += int(detections.size - matching.num_matchings)
        sub_state["fn_count"] += int(annotations.size - matching.num_matchings)
        sub_state["relaxed_tp_count"] += int(matching.relaxed_tp_count)
        sub_state["relaxed_fp_count"] += int(matching.num_clusters)
        sub_state["match_distance_sum"] += float(matching.matched_distances.sum())

        # Custom performance metric hooks
        matched_detections = detections.extract_index_tensor(matching.matched_d_idxs)
        matched_annotations = annotations.extract_index_tensor(matching.matched_a_idxs)
        assert matched_annotations.size == matched_detections.size
        self.__increment_hook__(matched_detections, matched_annotations, matching, sub_state)

    def increment(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ):
        assert isinstance(detections, DetectionBlock)
        assert isinstance(annotations, AnnotationBlock)
        for confidence_threshold, sub_state in zip(self.confidence_thresholds, self.sub_states):
            crt_detections = detections.filter_min_positivity(confidence_threshold)
            self.increment_substate(crt_detections, annotations, sub_state)

    def divide(self, upper, lower):
        if lower == 0:
            return None
        else:
            return float(upper / lower)

    def finalize_substate(self, sub_state: Dict[str, Any]) -> Dict[str, Any]:
        num_predictions = sub_state["tp_count"] + sub_state["fp_count"]
        num_annotations = sub_state["tp_count"] + sub_state["fn_count"]
        precision = self.divide(sub_state["tp_count"], num_predictions)
        recall = self.divide(sub_state["tp_count"], num_annotations)
        if precision is None or recall is None:
            f2 = 0.0
        elif precision == 0.0 and recall == 0.0:
            f2 = 0.0
        else:
            f2 = ((1 + 4) * precision * recall) / (4 * precision + recall)

        output = {
            "precision": precision,
            "recall": recall,
            "f2": f2,
            "relaxed_precision": self.divide(sub_state["tp_count"], (sub_state["tp_count"] + sub_state["relaxed_fp_count"])),
            "relaxed_recall": self.divide(sub_state["relaxed_tp_count"], num_annotations),

            "tp_count": sub_state["tp_count"],
            "fp_count": sub_state["fp_count"],
            "fn_count": sub_state["fn_count"],
            "relaxed_tp_count": sub_state["relaxed_tp_count"],
            "relaxed_fp_count": sub_state["relaxed_fp_count"],
            "num_predictions": num_predictions,
            "num_annotations": num_annotations,

            "avg_distance": self.divide(sub_state["match_distance_sum"], sub_state["tp_count"]),
        }

        hook = self.__finalize_hook__(sub_state)
        if hook is not None:
            for key, val in hook.items():
                output[key] = val

        return output

    def auc_helper(self, precision: List[float], recall: List[float]) -> float:
        n = len(precision)
        assert len(recall) == n

        # Remove None values
        precision_not_none = []
        recall_not_none = []
        for p, r, in zip(precision, recall):
            if p is not None and r is not None:
                precision_not_none.append(p)
                recall_not_none.append(r)

        precision = precision_not_none
        recall = recall_not_none
        if len(precision) == 0:
            return 0.0

        # Sort elements in precision and recall by the ordering of recall
        recall_idx_tuples = [(i, r) for (i, r) in enumerate(recall)]
        recall_idx_tuples.sort(key=lambda i_r_tuple: i_r_tuple[1])
        new_order = [i for (i, r) in recall_idx_tuples]
        ordered_recall = [recall[i] for i in new_order]
        ordered_precision = [precision[i] for i in new_order]

        # Enforce a "stair shape" of the curve. If a sub-state with higher recall
        # also has higher precicion, we use the higher precicion
        precision_staired = [
            max((ordered_precision[j] for j in range(i,len(ordered_precision))))
            for i in range(len(ordered_precision))
        ]

        # Add a recall=0 and recall=1 entry, to include the entire graph
        # For recall=0, we use precicion=precicion_staired[0] as the corresponding precision
        # For recall=1, we use precicion=0
        precicion_padded = [precision_staired[0]] + precision_staired + [0.0]
        recall_padded = [0.0] + ordered_recall + [1.0]

        # We now have a complete (from 0 to 1) and monotonic (stair shaped) precision-recall curve
        # Do a pair-wise iteration to integrate the AUC
        num_entries = len(precicion_padded)
        assert num_entries == len(recall_padded)
        auc = 0
        for i in range(num_entries-1):
            # Extract parallelogram data for current line segment
            recall_prev = recall_padded[i]
            recall_next = recall_padded[i+1]
            precicion_prev = precicion_padded[i]
            precicion_next = precicion_padded[i+1]
            auc += (precicion_next + precicion_prev) * (recall_next - recall_prev) / 2

        return auc

    def finalize(self) -> Dict[str, Any]:
        # Performance metrics for each confidence threshold
        sub_finalizes = []
        for sub_state in self.sub_states:
            sub_finalizes.append(self.finalize_substate(sub_state))

        # Area under curve (AUC). aka mAP. aka AP.
        precision = [sub_finalize["precision"] for sub_finalize in sub_finalizes]
        recall = [sub_finalize["recall"] for sub_finalize in sub_finalizes]
        auc = self.auc_helper(precision, recall)

        # Relaxed AUC
        relaxed_precision = [sub_finalize["relaxed_precision"] for sub_finalize in sub_finalizes]
        relaxed_recall = [sub_finalize["relaxed_recall"] for sub_finalize in sub_finalizes]
        relaxed_auc = self.auc_helper(relaxed_precision, relaxed_recall)

        # Output
        output = {
            "sub_results": {
                f"{threshold:.2f}": sub_finalize for (threshold, sub_finalize) in zip(self.confidence_thresholds, sub_finalizes)
            },
            "auc": auc,
            "relaxed_auc": relaxed_auc,
        }
        return output



class TestPointPerformanceMetrics(unittest.TestCase):
    def list_to_tensor(self, values: List[Tuple[float, float]]) -> Tuple[int, torch.Tensor]:
        size = len(values)
        if size == 0:
            tensor = torch.zeros((0, 2), dtype=torch.float32)
        else:
            torch_list = [torch.tensor(x, dtype=torch.float32) for x in values]
            tensor = torch.stack(torch_list)

        assert tensor.shape == (size, 2)
        return size, tensor

    def helper(
        self,
        annotations: List[Tuple[float, float]],
        detections: List[Tuple[float, float]],
        expected_tp_count,
        expected_fp_count,
        expected_fn_count,
        expected_relaxed_tp_count,
        expected_relaxed_fp_count,
        expected_match_distance_sum,
        match_thresh: float = 100.0,
        duplicate_thresh: float = 40.0,
    ):
        # Annotation block
        a_block = AnnotationBlock.from_annotation_list([
            Annotation(center_yx=np.array(a), is_high_confidence=True)
            for a in annotations
        ])

        # Detection block
        d_size, d_tensor = self.list_to_tensor(detections)
        d_block = DetectionBlock(
            size=d_size,
            center_yx=d_tensor,
            objectness=torch.tensor(0.8)[None].expand(d_size)
        )

        # Get performance metrics
        performance_metrics = BasePerformanceMetric(match_thresh, duplicate_thresh)
        performance_metrics.reset()
        performance_metrics.increment(d_block, a_block)

        # Compare performance metrics
        self.assertEqual(performance_metrics.tp_count, expected_tp_count)
        self.assertEqual(performance_metrics.fp_count, expected_fp_count)
        self.assertEqual(performance_metrics.fn_count, expected_fn_count)
        self.assertEqual(performance_metrics.relaxed_tp_count, expected_relaxed_tp_count)
        self.assertEqual(performance_metrics.relaxed_fp_count, expected_relaxed_fp_count)
        self.assertAlmostEqual(performance_metrics.match_distance_sum, expected_match_distance_sum)


    def test_no_detections_many_annotations(self):
        self.helper(
            annotations=[(0, 1), (10, 50), (1000, 900)],
            detections=[],
            expected_tp_count=0,
            expected_fp_count=0,
            expected_fn_count=3,
            expected_relaxed_tp_count=0,
            expected_relaxed_fp_count=0,
            expected_match_distance_sum=0.0,
        )


    def test_no_detections_no_annotations(self):
        self.helper(
            annotations=[],
            detections=[],
            expected_tp_count=0,
            expected_fp_count=0,
            expected_fn_count=0,
            expected_relaxed_tp_count=0,
            expected_relaxed_fp_count=0,
            expected_match_distance_sum=0.0,
        )


    def test_one_detection_no_annotations(self):
        self.helper(
            annotations=[],
            detections=[(100, 100)],
            expected_tp_count=0,
            expected_fp_count=1,
            expected_fn_count=0,
            expected_relaxed_tp_count=0,
            expected_relaxed_fp_count=1,
            expected_match_distance_sum=0.0,
        )


    def test_duplicate_false_detection_no_annotations(self):
        self.helper(
            annotations=[],
            detections=[(100, 100), (101, 101)],
            expected_tp_count=0,
            expected_fp_count=2,
            expected_fn_count=0,
            expected_relaxed_tp_count=0,
            expected_relaxed_fp_count=1,
            expected_match_distance_sum=0.0,
        )


    def test_non_duplicate_false_detections_no_annotations(self):
        self.helper(
            annotations=[],
            detections=[(100, 100), (1000, 11)],
            expected_tp_count=0,
            expected_fp_count=2,
            expected_fn_count=0,
            expected_relaxed_tp_count=0,
            expected_relaxed_fp_count=2,
            expected_match_distance_sum=0.0,
        )

    def extract_thing(self, objects, type):
        return [x["VALUE"] for x in objects if x["TYPE"] == type]

    def extract_sum(self, objects, type):
        return sum(self.extract_thing(objects, type))

    def test_more_detections_than_annotations(self):
        TP = "TP"
        FP = "FP"
        FN = "FN"
        RELAXED_TP = "RELAXED_TP"
        RELAXED_FP = "RELAXED_FP"
        DISTANCE = "DISTANCE"
        ANNOTATION = "ANNOTATION"
        DETECTION = "DETECTION"
        duplicate_thresh = 30.0 / 2
        match_thresh = 100.0
        objects = [
            # Detection and annotation                  (1000, 1000)
            # Barely within match range
            {"TYPE": TP, "VALUE": 1},
            {"TYPE": RELAXED_TP, "VALUE": 1},
            {"TYPE": DISTANCE, "VALUE": 99},
            {"TYPE": ANNOTATION, "VALUE": (1001, 1000)},
            {"TYPE": DETECTION, "VALUE": (1001, 1099)},

            # One annotation, duplicate detection       (1000, 2000)
            {"TYPE": TP, "VALUE": 1},
            {"TYPE": RELAXED_TP, "VALUE": 1},
            {"TYPE": FP, "VALUE": 1},
            {"TYPE": DISTANCE, "VALUE": 1},
            {"TYPE": ANNOTATION, "VALUE": (1000, 2000)},
            {"TYPE": DETECTION, "VALUE": (1001, 2004)},
            {"TYPE": DETECTION, "VALUE": (1000, 1999)},

            # Two annotations, two detections           (1000, 3000)
            {"TYPE": TP, "VALUE": 2},
            {"TYPE": RELAXED_TP, "VALUE": 2},
            {"TYPE": DISTANCE, "VALUE": 2},
            {"TYPE": ANNOTATION, "VALUE": (1001, 3000)},
            {"TYPE": DETECTION, "VALUE": (1001, 3001)},
            {"TYPE": ANNOTATION, "VALUE": (999, 3000)},
            {"TYPE": DETECTION, "VALUE": (999, 3001)},

            # Lone annotation                           (1000, 4000)
            {"TYPE": FN, "VALUE": 1},
            {"TYPE": ANNOTATION, "VALUE": (999, 4000)},

            # Duplicate false detection                 (2000, 1000)
            {"TYPE": FP, "VALUE": 2},
            {"TYPE": RELAXED_FP, "VALUE": 1},
            {"TYPE": DETECTION, "VALUE": (1999, 1000)},
            {"TYPE": DETECTION, "VALUE": (1999, 1029)},

            # Close false detections (not duplicate)    (2000, 2000)
            {"TYPE": FP, "VALUE": 2},
            {"TYPE": RELAXED_FP, "VALUE": 2},
            {"TYPE": DETECTION, "VALUE": (1999, 2000)},
            {"TYPE": DETECTION, "VALUE": (1999, 2031)},

            # Detection close to annotation,            (2000, 4000)
            # barely outside match distance
            {"TYPE": FP, "VALUE": 1},
            {"TYPE": FN, "VALUE": 1},
            {"TYPE": RELAXED_FP, "VALUE": 1},
            {"TYPE": ANNOTATION, "VALUE": (1999, 4000)},
            {"TYPE": DETECTION, "VALUE": (1999, 4101)},

            # Single detection, two annotations         (3000, 1000)
            {"TYPE": TP, "VALUE": 1},
            {"TYPE": FN, "VALUE": 1},
            {"TYPE": RELAXED_TP, "VALUE": 2},
            {"TYPE": DISTANCE, "VALUE": 2},
            {"TYPE": DETECTION, "VALUE": (2999, 1000)},
            {"TYPE": ANNOTATION, "VALUE": (3001, 1000)},
            {"TYPE": ANNOTATION, "VALUE": (3001, 999)},
        ]


        TP = self.extract_sum(objects, TP)
        FP = self.extract_sum(objects, FP)
        FN = self.extract_sum(objects, FN)
        RELAXED_TP = self.extract_sum(objects, RELAXED_TP)
        RELAXED_FP = self.extract_sum(objects, RELAXED_FP)
        DISTANCE = self.extract_sum(objects, DISTANCE)
        ANNOTATION = self.extract_thing(objects, ANNOTATION)
        DETECTION = self.extract_thing(objects, DETECTION)
        self.helper(
            ANNOTATION,
            DETECTION,
            TP,
            FP,
            FN,
            RELAXED_TP,
            RELAXED_FP,
            DISTANCE,
            match_thresh,
            duplicate_thresh
        )

