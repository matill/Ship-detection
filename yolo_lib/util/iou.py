from typing import Tuple
import unittest
import torch
from torch import Tensor
from yolo_lib.cfg import DEVICE, SAFE_MODE
from yolo_lib.torch_consts import DEVICE_4_OVER_PI_SQUARED, DEVICE_0
from yolo_lib.util.check_tensor import check_tensor

"""
Implements the core mathematics involving IoU and related metrics
"""

__all__ = ["get_centered_iou", "get_ciou_grid_loss", "get_iou"]


def get_centered_iou(
    predicted_hw: torch.Tensor, # shape: [n, m, {hw}]
    true_hw: torch.Tensor,      # shape: [n, m, {hw}]
    n: int,
    m: int,
) -> torch.Tensor:
    assert isinstance(n, int)
    assert isinstance(m, int)
    assert predicted_hw.shape == (n, m, 2)
    assert true_hw.shape == (n, m, 2)

    # Bottom right (br)
    predicted_br = predicted_hw * 0.5
    true_br = true_hw * 0.5

    # Top left (tl)
    predicted_tl = -predicted_br
    true_tl = -true_br

    # Use IoU core
    return _iou_core(
        predicted_hw,
        predicted_tl,
        predicted_br,
        true_hw,
        true_tl,
        true_br,
        n,
        m,
    )


def get_diou_grid_loss(
    predicted_yx: torch.Tensor, # shape: [n, m, {yx}]
    predicted_hw: torch.Tensor, # shape: [n, m, {hw}]
    true_yx: torch.Tensor,      # shape: [n, m, {yx}]
    true_hw: torch.Tensor,      # shape: [n, m, {hw}]
    n: int,
    m: int,
    do_detach: bool,
) -> Tensor:
    # Get top-left and bottom-rigth corners of true and predicted bounding boxes
    # [n, m, 2]
    predicted_top_left, predicted_bottom_right = _get_top_left_and_bottom_right(predicted_yx, predicted_hw)
    true_top_left, true_bottom_right = _get_top_left_and_bottom_right(true_yx, true_hw)

    # Compute IoU
    # [n, m]
    iou = _iou_core(
        predicted_hw,
        predicted_top_left,
        predicted_bottom_right,
        true_hw,
        true_top_left,
        true_bottom_right,
        n,
        m
    )

    # Compute DIoU
    diou = _diou_core(
        iou,
        predicted_yx,
        predicted_top_left,
        predicted_bottom_right,
        true_yx,
        true_top_left,
        true_bottom_right,
        n,
        m,
        do_detach,
    )

    check_tensor(iou, (n, m))
    check_tensor(diou, (n, m))
    return diou


def get_ciou_grid_loss(
    predicted_yx: torch.Tensor, # shape: [n, m, {yx}]
    predicted_hw: torch.Tensor, # shape: [n, m, {hw}]
    true_yx: torch.Tensor,      # shape: [n, m, {yx}]
    true_hw: torch.Tensor,      # shape: [n, m, {hw}]
    n: int,
    m: int,
    stable: bool,
) -> torch.Tensor:
    # Get top-left and bottom-rigth corners of true and predicted bounding boxes
    # [n, m, 2]
    predicted_top_left, predicted_bottom_right = _get_top_left_and_bottom_right(predicted_yx, predicted_hw)
    true_top_left, true_bottom_right = _get_top_left_and_bottom_right(true_yx, true_hw)

    # Compute IoU
    # [n, m]
    iou = _iou_core(
        predicted_hw,
        predicted_top_left,
        predicted_bottom_right,
        true_hw,
        true_top_left,
        true_bottom_right,
        n,
        m
    )

    # Compute DIoU
    diou = _diou_core(
        iou,
        predicted_yx,
        predicted_top_left,
        predicted_bottom_right,
        true_yx,
        true_top_left,
        true_bottom_right,
        n,
        m
    )

    # Compute CIoU
    ciou = _ciou_core(
        iou,
        diou,
        predicted_hw,
        true_hw,
        n,
        m,
        stable,
    )

    check_tensor(iou, (n, m))
    check_tensor(diou, (n, m))
    check_tensor(ciou, (n, m))
    return ciou


def get_iou(
    predicted_yx: torch.Tensor,
    predicted_hw: torch.Tensor,

    true_yx: torch.Tensor,
    true_hw: torch.Tensor,
    n: int,
    m: int,
) -> torch.Tensor:
    predicted_top_left, predicted_bottom_right = _get_top_left_and_bottom_right(predicted_yx, predicted_hw)
    true_top_left, true_bottom_right = _get_top_left_and_bottom_right(true_yx, true_hw)
    check_tensor(predicted_top_left, (n, m, 2))
    check_tensor(predicted_bottom_right, (n, m, 2))
    check_tensor(true_top_left, (n, m, 2))
    check_tensor(true_bottom_right, (n, m, 2))
    return _iou_core(
        predicted_hw,
        predicted_top_left,
        predicted_bottom_right,
        true_hw,
        true_top_left,
        true_bottom_right,
        n,
        m
    )


def _iou_core(
    predicted_hw,
    predicted_top_left,
    predicted_bottom_right,
    true_hw,
    true_top_left,
    true_bottom_right,
    n,
    m
):
    # For each match, get the intersection area by multiplying intersection height
    # and width.
    smallest_bottom_right = torch.min(predicted_bottom_right, true_bottom_right)
    largest_top_left = torch.max(predicted_top_left, true_top_left)
    height_widths = torch.max(smallest_bottom_right - largest_top_left, DEVICE_0)
    heights = height_widths[:, :, 0]
    widths = height_widths[:, :, 1]
    intersections = heights * widths

    # Compute union area of each annotation-prediction pair using |Union(A, B)| = |A| + |B| - |Intersection(A, B)|
    true_areas = true_hw[:, :, 0] * true_hw[:, :, 1]
    predicted_areas = predicted_hw[:, :, 0] * predicted_hw[:, :, 1]
    unions = true_areas + predicted_areas - intersections

    # Compute IoU
    iou = intersections / unions

    # Assertions
    if SAFE_MODE:
        check_tensor(intersections, (n, m))
        check_tensor(true_areas, (n, m))
        check_tensor(predicted_areas, (n, m))
        check_tensor(unions, (n, m))
        check_tensor(iou, (n, m))

    return iou


def _diou_core(
    iou: Tensor,
    predicted_yx: Tensor,
    predicted_top_left: Tensor,
    predicted_bottom_right: Tensor,
    true_yx: Tensor,
    true_top_left: Tensor,
    true_bottom_right: Tensor,
    n: int,
    m: int,
    do_detach: bool=True,
) -> Tensor:
    # Get the square of the length of the diagonal of the smallest bounding box
    # that contains both the annotation and the predicted box
    largest_bottom_right = torch.max(predicted_bottom_right, true_bottom_right)
    smallest_top_left = torch.min(predicted_top_left, true_top_left)
    big_box_diagonal = largest_bottom_right - smallest_top_left
    big_box_diagonal_len_sqrd = big_box_diagonal.square().sum(dim=2)
    check_tensor(largest_bottom_right, (n, m, 2))
    check_tensor(smallest_top_left, (n, m, 2))
    check_tensor(big_box_diagonal, (n, m, 2))
    check_tensor(big_box_diagonal_len_sqrd, (n, m))

    # NOTE: DETACHING TEST
    # TODO: Check if this should be done
    if do_detach:
        big_box_diagonal_len_sqrd = big_box_diagonal_len_sqrd.detach()

    # Get the square of the distance between the center points of the annotation and
    # the predicted box
    center_differences = predicted_yx - true_yx
    center_differences_len_sqrd = center_differences.square().sum(dim=2)
    check_tensor(center_differences, (n, m, 2))
    check_tensor(center_differences_len_sqrd, (n, m))

    # Compute normalized L2 difference between predicted and true positions
    normalized_l2 = center_differences_len_sqrd / big_box_diagonal_len_sqrd
    check_tensor(normalized_l2, (n, m))

    # Compute DIoU loss
    diou_loss = normalized_l2 - iou + 1
    check_tensor(diou_loss, (n, m))
    return diou_loss


def _aspect_ratio_arctan(hw):
    heights = hw[:, :, 0]
    widths = hw[:, :, 1]
    return torch.arctan(widths / heights)


class StableAspectRatioDiff(torch.autograd.Function):
    """
    Computes the v term in CIoU loss. v = (arctan(wt/ht) - arctan(wp/hp))^2 * 4/pi^2
    Numerically stabilized backwards pass as described in "Distance-IoU Loss: Faster and..."
    """

    @staticmethod
    def forward(ctx, predicted_hw: Tensor, true_hw: Tensor, n: int, m: int) -> Tensor:
        check_tensor(predicted_hw, (n, m, 2))
        check_tensor(true_hw, (n, m, 2))
        
        # Get the aspect ratio arctan of predicted and true bounding boxes
        predicted_arctan = _aspect_ratio_arctan(predicted_hw)
        true_arctan = _aspect_ratio_arctan(true_hw)
        arctan_diff = true_arctan - predicted_arctan
        v = arctan_diff.square() * (4 / 3.14159 ** 2)
        check_tensor(predicted_arctan, (n, m))
        check_tensor(true_arctan, (n, m))
        check_tensor(arctan_diff, (n, m))
        check_tensor(v, (n, m))

        # Save arctan_diff, and return v
        ctx.save_for_backward(arctan_diff, predicted_hw)
        return v

    @staticmethod
    def backward(ctx, grad_loss_v: Tensor) -> Tuple[Tensor, None, None, None]:
        (arctan_diff, predicted_hw) = ctx.saved_tensors
        (n, m) = arctan_diff.shape
        check_tensor(grad_loss_v, (n, m))

        # Compute d(v)/d(hw)
        # NOTE: h gradients are proportional to w, while w gradients are proportional to h.
        # Hence, grad_v_predicted_hw[:, :, 0] use predicted_hw[:, :, 1], and vice versa. 
        # This is not an error
        scaled_arctan_diff = arctan_diff * 8 / (3.14159 ** 2)
        grad_v_predicted_hw = torch.empty(n, m, 2, dtype=torch.float, device=DEVICE)
        grad_v_predicted_hw[:, :, 0] = +scaled_arctan_diff * predicted_hw[:, :, 1]
        grad_v_predicted_hw[:, :, 1] = -scaled_arctan_diff * predicted_hw[:, :, 0]
        check_tensor(grad_v_predicted_hw, (n, m, 2))

        # Compute d(loss) / d(hw)
        grad_loss_predicted_hw = grad_loss_v[:, :, None] * grad_v_predicted_hw
        check_tensor(grad_loss_predicted_hw, (n, m, 2))
        return grad_loss_predicted_hw, None, None, None


def unstable_aspect_ratio_diff(predicted_hw: Tensor, true_hw: Tensor, n: int, m: int) -> Tensor:
    predicted_arctan = _aspect_ratio_arctan(predicted_hw)
    true_arctan = _aspect_ratio_arctan(true_hw)
    arctan_diff = true_arctan - predicted_arctan
    v = arctan_diff.square() * (4 / 3.14159 ** 2)
    check_tensor(predicted_arctan, (n, m))
    check_tensor(true_arctan, (n, m))
    check_tensor(arctan_diff, (n, m))
    check_tensor(v, (n, m))
    return v


def aspect_ratio_diff(predicted_hw: Tensor, true_hw: Tensor, n: int, m: int, stable: bool) -> Tensor:
    if stable:
        return StableAspectRatioDiff.apply(predicted_hw, true_hw, n, m)
    else:
        return unstable_aspect_ratio_diff(predicted_hw, true_hw, n, m)


def _ciou_core(
    iou: Tensor,
    diou: Tensor,
    predicted_hw: Tensor,
    true_hw: Tensor,
    n: int,
    m: int,
    stable: bool
):
    v = aspect_ratio_diff(predicted_hw, true_hw, n, m, stable)
    check_tensor(v, (n, m))

    # alpha: Trade-off parameter
    alpha = v / (1 - iou + v)

    # CIoU: DIoU + alpha * v
    alpha_times_v = alpha * v
    ciou = diou + alpha_times_v
    check_tensor(alpha, (n, m))
    check_tensor(alpha_times_v, (n, m))
    check_tensor(ciou, (n, m))
    return ciou


def _get_top_left_and_bottom_right(yx, hw):
    half_hw = hw * 0.5
    top_left = yx - half_hw
    bottom_right = yx + half_hw
    return top_left, bottom_right


class TestAspectRatioDiffBackprop(unittest.TestCase):
    def test_backprop(self):
        predicted_hw = torch.tensor([
            [
                [10, 10],
                [20, 10],
                [10, 20],
            ],
        ], dtype=torch.float, requires_grad=True)
        check_tensor(predicted_hw, (1, 3, 2))

        true_hw = torch.tensor([
            [
                [20, 20],
                [10, 20],
                [20, 10],
            ],
        ], dtype=torch.float)
        check_tensor(predicted_hw, (1, 3, 2))

        stable_v = aspect_ratio_diff(predicted_hw, true_hw, 1, 3, True)
        unstable_v = aspect_ratio_diff(predicted_hw, true_hw, 1, 3, False)

        # Check v is equal
        self.assertTrue((stable_v == unstable_v).all())

        # Get stable_v gradient (multiply by 2 to check that gradient is also multiplied by 2)
        (stable_v.sum() * 2).backward()
        stable_v_grad = predicted_hw.grad.detach()
        predicted_hw.grad = None
        check_tensor(stable_v_grad, (1, 3, 2))

        # Get stable_v gradient
        (unstable_v.sum() * 2).backward()
        unstable_v_grad = predicted_hw.grad.detach()
        predicted_hw.grad = None
        check_tensor(unstable_v_grad, (1, 3, 2))

        # Check that gradients are "similar" (same direction, but different scale)
        scales = predicted_hw.square().sum(dim=2, keepdim=True)
        unstable_v_grad_scaled = unstable_v_grad * scales
        check_tensor(scales, (1, 3, 1))
        check_tensor(unstable_v_grad_scaled, (1, 3, 2))
        self.assertTrue((stable_v_grad == unstable_v_grad_scaled).all(), f"\n\n{unstable_v_grad_scaled}\n\n{stable_v_grad   }")

