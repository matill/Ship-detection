import torch
from typing import Any, Dict, List
from yolo_lib.util.iou import get_centered_iou
from yolo_lib.data.dataclasses import DetectionBlock, AnnotationBlock
from yolo_lib.util.check_tensor import check_tensor
from .matching import Matching


class RegressionMetric:
    def reset(self):
        pass

    def increment(
        self,
        matched_detections: DetectionBlock,
        matched_annotations: AnnotationBlock,
    ):
        raise NotImplementedError

    def finalize(self) -> Dict[str, float]:
        raise NotImplementedError

    def divide(self, upper, lower):
        if lower == 0:
            return None
        else:
            return float(upper / lower)


class CenteredIoUMetric(RegressionMetric):
    def reset(self):
        self.annotations_with_hw_count = 0
        self.iou_sum = 0

    def increment(
        self,
        matched_detections: DetectionBlock,
        matched_annotations: AnnotationBlock,
    ):
        assert isinstance(matched_detections, DetectionBlock)
        assert isinstance(matched_annotations, AnnotationBlock)
        assert matched_detections.size == matched_annotations.size
        num_matchings_total = matched_detections.size

        # Extract HW values where the Annotation/target is known, and compute CenteredIoU
        with_hw_bitmap = matched_annotations.has_size_hw
        check_tensor(with_hw_bitmap, (num_matchings_total, ), torch.bool)
        true_hw = matched_annotations.size_hw[with_hw_bitmap]
        predicted_hw = matched_detections.size_hw[with_hw_bitmap]
        num_with_hw = predicted_hw.shape[0]
        assert predicted_hw.shape == (num_with_hw, 2)
        assert true_hw.shape == (num_with_hw, 2)
        ious = get_centered_iou(
            predicted_hw[:, None, :],
            true_hw[:, None, :],
            num_with_hw,
            1
        )

        # Check output shape and increment counters
        assert ious.shape == (num_with_hw, 1)
        self.iou_sum += float(ious.sum())
        self.annotations_with_hw_count += num_with_hw

    def finalize(self) -> Dict[str, float]:
        return {
            "avg_iou": self.divide(self.iou_sum, self.annotations_with_hw_count),
        }

class RotationMetric(RegressionMetric):
    def reset(self):
        self.angle_mod_360_sum = 0
        self.annotations_with_360_angle_count = 0
        self.angle_mod_180_sum = 0
        self.annotations_with_180_angle_count = 0
        self.correct_direction_count = 0

    def increment(
        self,
        matched_detections: DetectionBlock,
        matched_annotations: AnnotationBlock,
    ):
        # Subset with known orientation, WITHOUT distinguishing front and back
        # Compute mod-180 angle differences and increment counters
        with_rotation_bitmap = matched_annotations.has_rotation
        matched_annotations_180 = matched_annotations.extract_bitmap(with_rotation_bitmap)
        matched_detections_180 = matched_detections.extract_bitmap(with_rotation_bitmap)
        num_180 = matched_annotations_180.size
        mod_180_diffs = self.get_mod_180_differences(
            num_180,
            matched_annotations_180.rotation,
            matched_detections_180.rotation,
        )

        assert mod_180_diffs.shape == (num_180, )
        self.angle_mod_180_sum += float(mod_180_diffs.sum())
        self.annotations_with_180_angle_count += num_180

        # Subset with known orientation, WITH distinguishing front and back
        with_360_bitmap = matched_annotations_180.has_rotation
        matched_annotations_360 = matched_annotations_180.extract_bitmap(with_360_bitmap)
        matched_detections_360 = matched_detections_180.extract_bitmap(with_360_bitmap)
        num_360 = matched_annotations_360.size
        mod_360_diffs = self.get_mod_360_differences(
            num_360,
            matched_annotations_360.rotation,
            matched_detections_360.rotation,
        )

        assert mod_360_diffs.shape == (num_360, )
        self.angle_mod_360_sum += float(mod_360_diffs.sum())
        self.annotations_with_360_angle_count += num_360
        self.correct_direction_count += float((0.25 > mod_360_diffs).sum())

    def get_mod_180_differences(
        self,
        num_matchings: int,
        true_rotations: torch.Tensor,
        predicted_rotations: torch.Tensor,
    ) -> torch.Tensor:
        shape = (num_matchings, )
        assert isinstance(num_matchings, int)
        assert isinstance(true_rotations, torch.Tensor) and true_rotations.shape == shape
        assert isinstance(predicted_rotations, torch.Tensor) and predicted_rotations.shape == shape

        true_rotations_mod = true_rotations % 0.5
        predicted_rotations_mod = predicted_rotations % 0.5
        differences = (true_rotations_mod - predicted_rotations_mod).abs()
        reversed_idxs = differences > 0.25
        reversed_vals = (0.5 - differences) * reversed_idxs
        nonreversed_vals = differences * (~reversed_idxs)
        corrected_differences = reversed_vals + nonreversed_vals
        assert corrected_differences.shape == shape
        return corrected_differences

    def get_mod_360_differences(
        self,
        num_matchings: int,
        true_rotations: torch.Tensor,
        predicted_rotations: torch.Tensor,
    ) -> torch.Tensor:
        shape = (num_matchings, )
        assert isinstance(num_matchings, int)
        assert isinstance(true_rotations, torch.Tensor) and true_rotations.shape == shape
        assert isinstance(predicted_rotations, torch.Tensor) and predicted_rotations.shape == shape

        diffs = (true_rotations - predicted_rotations).abs()
        over_180_bitmap = diffs > 0.5
        under_180_bitmap = ~over_180_bitmap
        under_180_values = diffs * under_180_bitmap
        over_180_values = (1.0 - diffs) * over_180_bitmap
        corrected_diffs = under_180_values + over_180_values
        assert corrected_diffs.shape == shape
        return corrected_diffs

    def finalize(self) -> Dict[str, float]:
        return {
            "avg_angle_mod_360_degrees": self.divide(self.angle_mod_360_sum * 360, self.annotations_with_360_angle_count),
            "avg_angle_mod_180_degrees": self.divide(self.angle_mod_180_sum * 360, self.annotations_with_180_angle_count),
            "correct_direction_rate": self.divide(self.correct_direction_count, self.annotations_with_360_angle_count),
            "annotations_with_180_angle_count": int(self.annotations_with_180_angle_count),
        }

class RegressionMetric:
    def reset(self):
        pass

    def increment(
        self,
        matched_detections: DetectionBlock,
        matched_annotations: AnnotationBlock,
    ):
        raise NotImplementedError

    def finalize(self) -> Dict[str, float]:
        raise NotImplementedError

    def divide(self, upper, lower):
        if lower == 0:
            return None
        else:
            return float(upper / lower)

class CenterDistaneMetric(RegressionMetric):
    def reset(self):
        self.distances: List[torch.Tensor] = []
        self.num_matchings = 0

    def increment(
        self,
        matched_detections: DetectionBlock,
        matched_annotations: AnnotationBlock,
    ):
        distance = (matched_detections.center_yx - matched_annotations.center_yx).norm(dim=1)
        check_tensor(distance, (matched_detections.size, ))
        self.distances.append(distance)
        self.num_matchings += matched_detections.size

    def finalize(self) -> Dict[str, float]:
        distances = torch.cat(self.distances)
        check_tensor(distances, (self.num_matchings, ))
        return {
            "mean_center_distance": float(distances.mean()),
            "median_center_distance": float(distances.median())
        }

class SubclassificationMetrics(RegressionMetric):
    def __init__(self, num_classes: int):
        assert isinstance(num_classes, int)
        assert num_classes >= 2
        self.num_classes = num_classes

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))

    def increment(
        self,
        matched_detections: DetectionBlock,
        matched_annotations: AnnotationBlock,
    ):
        assert isinstance(matched_detections, DetectionBlock)
        assert isinstance(matched_annotations, AnnotationBlock)
        assert matched_detections.size == matched_annotations.size
        num_matchings_total = matched_detections.size

        # Extract class values where the Annotation/target is known
        with_class_bitmap = matched_annotations.has_max_class
        check_tensor(with_class_bitmap, (num_matchings_total, ), torch.bool)
        true_class = matched_annotations.max_class[with_class_bitmap]
        predicted_class = matched_detections.get_max_class()[with_class_bitmap]
        num_with_hw = predicted_class.shape[0]
        check_tensor(true_class, (num_with_hw, ), torch.int64)
        check_tensor(predicted_class, (num_with_hw, ), torch.int64)

        # Increment confusion matrix
        self.confusion_matrix[true_class, predicted_class] += 1

    def finalize(self) -> Dict[str, float]:
        diagonal_vec = self.confusion_matrix.diag()
        diagonal_sum = diagonal_vec.sum()
        matrix_sum = self.confusion_matrix.sum()
        check_tensor(diagonal_vec, (self.num_classes, ), torch.int64)
        check_tensor(diagonal_sum, (), torch.int64)
        check_tensor(matrix_sum, (), torch.int64)
        return {
            "mean_accuracy": self.divide(diagonal_sum.float(), matrix_sum.float()),
            "confusion_matrix": [[int(elem)for elem in row] for row in self.confusion_matrix],
            "info": r"confusion_matrix[i][j] represents true class i and predicted class j",
        }



