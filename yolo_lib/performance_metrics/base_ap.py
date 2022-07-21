import unittest
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from yolo_lib.data.dataclasses import DetectionBlock, AnnotationBlock, Annotation
from yolo_lib.util.iou import get_iou
from yolo_lib.performance_metrics import regression_metrics
from yolo_lib.performance_metrics.base_performance_metric import BasePerformanceMetric
from yolo_lib.performance_metrics.regression_metrics import RegressionMetric
from yolo_lib.util.check_tensor import check_tensor
from .matching import Matching, get_num_clusters


class BaseAveragePrecision(BasePerformanceMetric):
    def __init__(self, max_detections: int, regression_metrics: List[RegressionMetric], include_f2: bool=False):
        assert isinstance(max_detections, int)
        self.max_detections = max_detections
        self.regression_metrics = regression_metrics
        self.include_f2 = include_f2
        self.reset()

    def reset(self):
        # List of resuts for each image
        self.results = []
        for regression_metric in self.regression_metrics:
            regression_metric.reset()

    def __get_similarity_matrix__(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ) -> torch.Tensor:
        raise NotImplementedError

    def __threshold_similarity_matrix__(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def increment(
        self,
        detections: DetectionBlock,
        annotations: AnnotationBlock,
    ):
        # Sort detections by positivity, and check edge cases
        detections = detections.order_by_objectness(self.max_detections)
        if detections.size == 0:
            self.results.append({
                "tp_positivity": torch.tensor([], dtype=torch.float32),
                "fp_positivity": torch.tensor([], dtype=torch.float32),
                "num_fn": annotations.size,
            })
            return
        elif annotations.size == 0:
            self.results.append({
                "tp_positivity": torch.tensor([], dtype=torch.float32),
                "fp_positivity": detections.objectness.cpu(),
                "num_fn": 0,
            })
            return

        # Sort detection block by positivity, and get similarity matrix.
        # Ensure that matrices are on the CPU for efficiency
        similarity_matrix = self.__get_similarity_matrix__(detections, annotations)
        thresholded_similarity_matrix = self.__threshold_similarity_matrix__(similarity_matrix)
        similarity_matrix = similarity_matrix.cpu()
        thresholded_similarity_matrix = thresholded_similarity_matrix.cpu()
        check_tensor(similarity_matrix, (detections.size, annotations.size))
        check_tensor(thresholded_similarity_matrix, (detections.size, annotations.size), torch.bool)

        # Get indices of FP, TP_a, TP_p, and FN.
        unassigned_annotation_b = torch.empty((annotations.size), dtype=torch.bool)
        unassigned_annotation_b[:] = True
        num_unassigned_annotations = int(annotations.size)
        fp_idxs = []
        tp_idxs_a = []
        tp_idxs_p = []
        for i in range(detections.size):
            # Find the best (currently unassigned) annotation for prediction i.
            # If there are no alternatives, register i as false positive
            # If there are one or more alternatives, register i and j(best candidate)
            # as true positives, and also register j as assigned.
            candidate_annotation_idxs = (unassigned_annotation_b & thresholded_similarity_matrix[i]).nonzero()[:, 0]
            num_candidates = candidate_annotation_idxs.shape[0]
            check_tensor(candidate_annotation_idxs, (num_candidates, ), torch.int64)
            if num_candidates == 0:
                fp_idxs.append(i)
            else:
                similarities = similarity_matrix[i][candidate_annotation_idxs]
                check_tensor(similarities, (num_candidates, ))
                j = candidate_annotation_idxs[similarities.argmax()]
                check_tensor(j, (), torch.int64)
                tp_idxs_p.append(i)
                tp_idxs_a.append(j)
                unassigned_annotation_b[j] = False
                num_unassigned_annotations -= 1

        # We now have:
        # - Unassigned annotation count
        # - Pair-wise indices of matched annotations and predictions
        # - Indices of unassigned predictions
        # We store the positivity scores of false positives and true positives
        # in index tensors, and we store the number of false negatives (missed targets)
        fp_positivity = detections.objectness[fp_idxs].cpu()
        tp_positivity = detections.objectness[tp_idxs_p].cpu()
        self.results.append({
            "tp_positivity": tp_positivity,
            "fp_positivity": fp_positivity,
            "num_fn": num_unassigned_annotations,
        })

        # Increment regression metrics, if any
        if self.regression_metrics:
            tp_idxs_p = torch.tensor(tp_idxs_p, dtype=torch.int64)
            tp_idxs_a = torch.tensor(tp_idxs_a, dtype=torch.int64)
            matched_detections = detections.extract_index_tensor(tp_idxs_p)
            matched_annotations = annotations.extract_index_tensor(tp_idxs_a)
            for regression_metric in self.regression_metrics:
                regression_metric.increment(matched_detections, matched_annotations)

    def finalize(self) -> Dict[str, Any]:

        # Positivity of all true positives and false positives
        # Number of false negatives
        tp_positivity = torch.cat([result["tp_positivity"] for result in self.results])
        fp_positivity = torch.cat([result["fp_positivity"] for result in self.results])
        num_fn = torch.tensor([result["num_fn"] for result in self.results]).sum()

        # Create positivity matrix: [NUM_PREDICTIONS, 3]
        # One row for positivity score
        # Two rows for "is_tp" and "is_fp" flags.
        num_tp = tp_positivity.shape[0]
        num_fp = fp_positivity.shape[0]
        num_p = num_tp + num_fp
        tp_matrix = torch.empty((num_tp, 3), dtype=torch.float32)
        tp_matrix[:, 0] = tp_positivity
        tp_matrix[:, 1] = 0.0
        tp_matrix[:, 2] = 1.0
        fp_matrix = torch.empty((num_fp, 3), dtype=torch.float32)
        fp_matrix[:, 0] = fp_positivity
        fp_matrix[:, 1] = 1.0
        fp_matrix[:, 2] = 0.0
        positivity_matrix_unordered = torch.cat([tp_matrix, fp_matrix], dim=0)
        check_tensor(positivity_matrix_unordered, (num_p, 3), torch.float32)

        # Sort the positivity matrix in descending order
        positivity_matrix_order = positivity_matrix_unordered[:, 0].argsort(dim=0, descending=True)
        positivity_matrix = positivity_matrix_unordered[positivity_matrix_order]
        check_tensor(positivity_matrix_order, (num_p, ), torch.int64)
        check_tensor(positivity_matrix, (num_p, 3), torch.float32)

        # Get vectors representing number of TP, FN, and FP at each threshold level
        tp_vec: torch.Tensor = positivity_matrix[:, 2].cumsum(dim=0)
        fp_vec: torch.Tensor = positivity_matrix[:, 1].cumsum(dim=0)
        fn_vec: torch.Tensor = (num_fn + num_tp) - tp_vec

        # Compute precision and recall at each threshold level
        precision: torch.Tensor = tp_vec / (tp_vec + fp_vec)
        recall: torch.Tensor = tp_vec / (tp_vec + fn_vec)
        check_tensor(precision, (num_p, ))
        check_tensor(recall, (num_p, ))

        # Order precision and recall pairs by recall
        precision_recall_order = recall.argsort(descending=True)
        check_tensor(precision_recall_order, (num_p, ), torch.int64)
        precision_desc = precision[precision_recall_order]
        recall_desc = recall[precision_recall_order]
        check_tensor(precision_desc, (num_p, ))
        check_tensor(recall_desc, (num_p, ))

        # Interpolate precision (not necessary for recall)
        precision_interpolated_desc = precision_desc.cummax(dim=0).values

        # Compute the area under the precision-recall curve
        line_segment_widths = recall_desc[:-1] - recall_desc[1:]
        line_segment_heights = precision_interpolated_desc[1:]
        line_segment_areas = line_segment_widths * line_segment_heights
        check_tensor(line_segment_areas, (num_p - 1, ))
        max_precision = precision_interpolated_desc[-1]
        min_recall = recall_desc[-1]
        ap = line_segment_areas.sum() + max_precision * min_recall

        # Find highest F2 score achieved, along with the corresponding precision, recall, and positivity
        if self.include_f2:
            f2_vec_nan = ((1 + 4) * precision * recall) / (4 * precision + recall)
            f2_vec = torch.where(f2_vec_nan.isnan(), 0.0, f2_vec_nan.double())
            max_f2_idx = f2_vec.argmax()
            check_tensor(f2_vec, (num_p, ))
            check_tensor(max_f2_idx, ())
            max_f2 = {
                "F2": float(f2_vec[max_f2_idx]),
                "precision": float(precision[max_f2_idx]),
                "recall": float(recall[max_f2_idx]),
                "positivity": float(positivity_matrix[max_f2_idx, 0]),
            }

        # Build performance meric dict, and finalize regression metrics
        performance = {"AP": float(ap)}
        for regression_metric in self.regression_metrics:
            for key, val in regression_metric.finalize().items():
                performance[key] = val

        if self.include_f2:
            performance["F2"] = max_f2

        # Return performance metrics
        return performance

