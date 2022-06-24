
from __future__ import annotations
from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
import unittest

from dataclasses import dataclass
from typing import Tuple
import unittest
import torch
from yolo_lib.cfg import DEVICE, SAFE_MODE
from yolo_lib.data.dataclasses import Annotation, DetectionBlock, AnnotationBlock
from scipy.optimize import linear_sum_assignment


__all__ = ["matching", "get_num_clusters", "TestMatching"]


@dataclass
class Matching:
    # Distances from detections to annotations, and bitmap of thresholded entries
    da_distance_matrix: torch.Tensor
    thresholded_distance_matrix: torch.Tensor

    # Number of annotations that are close to any detections
    relaxed_tp_count: torch.Tensor

    # Bitmap of detections that are false positives
    # (1 => False positive, 0 => True positive of duplicate true positive)
    # For each detection
    false_detection_bitmap: torch.Tensor

    # Number of clusters in the false detection set
    num_clusters: int

    # Hungarian matching indexes
    num_matchings: int
    matched_d_idxs: torch.Tensor
    matched_a_idxs: torch.Tensor

    # Distances between matched annotations and detections
    matched_distances: torch.Tensor

    def get_matching(
        detections: DetectionBlock,
        annotations: AnnotationBlock,
        match_distance_threshold: torch.Tensor,
        duplicate_distance_threshold: torch.Tensor
    ) -> Matching:

        # This function should only be called with actual content
        assert detections.size > 0
        assert annotations.size > 0

        # Distance matrix between detections and annotations
        # (d, a) shape
        da_distance_matrix = _get_distance_matrix(detections.center_yx, annotations.center_yx)

        # Bitmap corresponding to which detections were close to which annotations
        thresholded_distance_matrix = da_distance_matrix < match_distance_threshold
        assert thresholded_distance_matrix.shape == (detections.size, annotations.size)

        # Relaxed tp count: Which annotations were close to any detections
        relaxed_tp_bitmap = torch.any(thresholded_distance_matrix, dim=0)
        assert relaxed_tp_bitmap.shape == (annotations.size, )
        relaxed_tp_count = relaxed_tp_bitmap.sum()

        # False detection bitmap: Bitmap corresponding to the set of detections that are not
        # close to any annotations, excluding duplicates.
        # (1 => False positive. 0 => True positive or duplicate true positive)
        false_detection_bitmap = ~torch.any(thresholded_distance_matrix, dim=1)
        assert false_detection_bitmap.shape == (detections.size, )

        # Relaxed (reduced) fp count: The number of clusters in
        # the set of false detections
        false_detections_yx = detections.center_yx[false_detection_bitmap]
        num_clusters = get_num_clusters(false_detections_yx, duplicate_distance_threshold)

        # Get Hungarian matching between detections and annotations
        detection_matching = _get_hungarian_matching(da_distance_matrix, match_distance_threshold)
        (matched_d_idxs, matched_a_idxs) = detection_matching
        num_matchings = matched_d_idxs.shape[0]
        assert matched_d_idxs.shape == (num_matchings, )
        assert matched_a_idxs.shape == (num_matchings, )

        # Matched distances
        matched_distances = da_distance_matrix[matched_d_idxs, matched_a_idxs]
        assert matched_distances.shape == (num_matchings, )

        # Return matching
        return Matching(
            da_distance_matrix,
            thresholded_distance_matrix,
            relaxed_tp_count,
            false_detection_bitmap,
            num_clusters,
            num_matchings,
            matched_d_idxs,
            matched_a_idxs,
            matched_distances,
        )


def _get_distance_matrix(a: torch.Tensor, b: torch.Tensor):
    A = a.shape[0]
    B = b.shape[0]

    # a.shape = (A, 2)
    # b.shape = (B, 2)
    # Add repeating axis to a and b to get common (A, B, 2) shape
    a_repeating = a[:, None, :].expand(-1, B, -1)
    b_repeating = b[None, :, :].expand(A, -1, -1)
    if SAFE_MODE:
        assert a_repeating.shape == b_repeating.shape

    # Norm of distance between a_repeating and b_repeating
    distance = a_repeating - b_repeating
    distance_norm = torch.norm(distance, dim=2)
    if SAFE_MODE:
        assert distance_norm.shape == (A, B)

    return distance_norm


def _get_hungarian_matching(
    distance_matrix: torch.Tensor,
    distance_threshold: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(distance_matrix, torch.Tensor)
    assert isinstance(distance_threshold, torch.Tensor)

    # Number of things :)
    num_detections, num_annotations = distance_matrix.shape

    # Set all disallowed entries (above threshold) to a fixed value
    allowed_bitmap = distance_matrix < distance_threshold
    not_allowed_vals = torch.zeros(num_detections, num_annotations, device=DEVICE) + (2*distance_threshold)
    not_allowed_masked = not_allowed_vals * (~allowed_bitmap)
    allowed_masked = distance_matrix * allowed_bitmap
    augmented_distance_matrix = allowed_masked + not_allowed_masked

    # Get matching (while still allowing both over and under threshold)
    distance_matrix_np = augmented_distance_matrix.cpu().detach().numpy()
    match = linear_sum_assignment(distance_matrix_np, maximize=False)
    d_idxs_np, a_idxs_np = match
    d_idxs = torch.tensor(d_idxs_np)
    a_idxs = torch.tensor(a_idxs_np)
    num_matchings = d_idxs.shape[0]
    assert d_idxs.shape == (num_matchings, )
    assert a_idxs.shape == (num_matchings, )
    assert num_matchings == min(num_detections, num_annotations)

    # Get cost of each assignment, and get a reduced set of matches that are
    # below the distance threshold
    costs = distance_matrix[d_idxs, a_idxs]
    allowed_match_bitmap = costs < distance_threshold
    d_idxs_allowed = d_idxs[allowed_match_bitmap]
    a_idxs_allowed = a_idxs[allowed_match_bitmap]
    num_allowed_matchings = d_idxs_allowed.shape[0]
    assert costs.shape == (num_matchings, )
    assert allowed_match_bitmap.shape == (num_matchings, )
    assert d_idxs_allowed.shape == (num_allowed_matchings, )
    assert a_idxs_allowed.shape == (num_allowed_matchings, )
    assert num_allowed_matchings <= num_matchings

    # Return final matching
    return (d_idxs_allowed, a_idxs_allowed)


def get_num_clusters(
    yx_positions: torch.Tensor,
    distance_threshold: torch.Tensor
) -> torch.Tensor:
    distance_threshold_np = distance_threshold.cpu().detach().numpy()

    # Edge case: Clustering fails when it gets 0 or 1 samples.
    # Correct behaviour is to return the number of samples.
    num_samples = yx_positions.shape[0]
    if num_samples <= 1:
        return num_samples

    # Get distance matrix
    num_samples = yx_positions.shape[0]
    distance_matrix = _get_distance_matrix(yx_positions, yx_positions)
    assert distance_matrix.shape == (num_samples, num_samples)
    distance_matrix_np = distance_matrix.cpu().detach().numpy()

    # Get clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        linkage="complete",
        distance_threshold=2*distance_threshold_np
    )
    return clustering.fit(distance_matrix_np).n_clusters_


class TestMatching(unittest.TestCase):
    def assert_tensors_equal(self, a, b, approx=None):
        self.assertEqual(a.shape, b.shape)
        if approx:
            equal = (a - b).abs() < approx
        else:
            equal = a == b

        self.assertTrue(equal.all(), f"\na:\n{a}\nb:\n{b}")

    def test_more_detections_than_annotations(self):
        detections = DetectionBlock(
            size=5,
            center_yx=torch.tensor([
                [0, 0],
                [0, 1],
                [100, 0],
                [101, 1],
                [300, 300]
            ], dtype=torch.float64),
            objectness=torch.tensor(.9).expand(5)
        )
        annotations = AnnotationBlock.from_annotation_list([
            Annotation([2, 2], True),
            Annotation([99, 99], True),
            Annotation([200, 200], True),
            Annotation([400, 400], True),
        ])

        duplicate_threshold = torch.tensor(50)
        match_threshold = torch.tensor(100)
        matching = Matching.get_matching(detections, annotations, match_threshold, duplicate_threshold)
        expected_matching = Matching(
            da_distance_matrix=torch.tensor([
                [  2.8284,   2.2361,  98.0204,  99.0051, 421.4356],
                [140.0071, 139.3018,  99.0051,  98.0204, 284.2569],
                [282.8427, 282.1365, 223.6068, 222.2656, 141.4214],
                [565.6854, 564.9788, 500.0000, 498.6000, 141.4214],
            ]).T,
            thresholded_distance_matrix=torch.tensor([
                [ True,  True,  True,  True, False],
                [False, False,  True,  True, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]).T,
            relaxed_tp_count=2,
            false_detection_bitmap=torch.tensor([False, False, False, False, True]),
            num_clusters=1,
            num_matchings=2,
            matched_a_idxs=None,
            matched_d_idxs=None,
            matched_distances=None
        )

        # Check all outputs except matching tensors
        self.assert_tensors_equal(expected_matching.da_distance_matrix, matching.da_distance_matrix, approx=0.01)
        self.assert_tensors_equal(expected_matching.thresholded_distance_matrix, matching.thresholded_distance_matrix)
        self.assert_tensors_equal(expected_matching.false_detection_bitmap, matching.false_detection_bitmap)
        self.assertEqual(expected_matching.num_clusters, matching.num_clusters)
        self.assertEqual(expected_matching.num_matchings, matching.num_matchings)

        # Get unordered sets of matching lists
        expected_matching = set([
            (1, 0, 2.23606797749979),
            (3, 1, 98.02040603874276),
        ])

        matched_d_idxs = matching.matched_d_idxs
        matched_a_idxs = matching.matched_a_idxs
        distances = matching.da_distance_matrix[matched_d_idxs, matched_a_idxs]
        received_matching = set([
            (int(d_idx), int(a_idx), float(distance))
            for (d_idx, a_idx, distance)
            in zip(matched_d_idxs, matched_a_idxs, distances)
        ])

        self.assertEqual(received_matching, expected_matching)


