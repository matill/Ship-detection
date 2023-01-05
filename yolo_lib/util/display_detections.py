from __future__ import annotations
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from yolo_lib.data.dataclasses import Detection, Annotation, DetectionBlock, YOLOTile

__all__ = ["display_detections", "display_yolo_tile"]



RBOX = "RBOX"
BOX = "BOX"
WIDE_SQUARE = "WIDE_SQUARE"
VALID_DISPLAY_MODES = [RBOX, BOX, WIDE_SQUARE]


def display_yolo_tile(
    tile: YOLOTile,
    fname: Optional[str]=None,
    detections: DetectionBlock=None,
    grid_spacing: Optional[int]=None,
    display_mode: str=RBOX,
    detection_list: Optional[List[Detection]]=None,
):
    if detection_list is None and detections is not None:
        detection_list = None if detections is None else detections.as_detection_list()

    display_detections(
        tile.image.detach(),
        fname,
        tile.annotations.as_annotation_list(),
        grid_spacing,
        detection_list,
        display_mode,
    )


def display_detections(
    image,
    fname: Optional[str]=None,
    annotations: Optional[List[Annotation]]=None,
    grid_spacing: Optional[int]=None,
    detections: Optional[List[Detection]]=None,
    display_mode: str=RBOX,
):
    # Show image
    fig, ax = plt.subplots()
    img_hwc = image[0].permute(1, 2, 0)
    c = img_hwc.shape[2]
    if c == 1:
        ax.imshow(img_hwc[:, :, 0].cpu(), cmap="gray")
    if c == 2:
        # Repeat one channel twice in red+green (yellow?), the next channel is blue
        ax.imshow(img_hwc[:, :, [0, 0, 1]].cpu())
    if c == 3:
        ax.imshow(img_hwc.cpu())

    # Plot ground-truth annotations
    if annotations is not None:
        for annotation in annotations:
            plot_annotation(ax, annotation, display_mode)

    # Plot detections
    if detections is not None:
        for detection in detections:
            plot_detection(ax, detection, display_mode)

    # Plot grid
    if grid_spacing:
        _b, _c, h, w = image.shape
        assert type(grid_spacing) == int and grid_spacing > 0

        # Draw horizontal lines
        x_values = [0, w-1]
        for i in range(grid_spacing, h, grid_spacing):
            y_values = [i, i]
            ax.plot(x_values, y_values, c="w")

        # Draw vertical lines
        y_values = [0, h-1]
        for i in range(grid_spacing, w, grid_spacing):
            x_values = [i, i]
            ax.plot(x_values, y_values, c="w", linewidth=1)

    # Save or display figure
    if fname is None:
        plt.show()
    else:
        fig.savefig(fname, bbox_inches="tight")
        plt.close("all")


def _get_rbox_corners(center_yx, size_hw, orientation_01):
    # Translate orientation to "forward" and "right" direction
    orientation_radians = orientation_01 * 2 * 3.14159
    sin = np.sin(orientation_radians)
    cos = np.cos(orientation_radians)
    forward_direction = np.array([-cos, sin]) * size_hw[0] / 2
    right_direction = np.array([sin, cos]) * size_hw[1] / 2

    # Find coordinates of top-left, top-right, bottom-left, and bottom-right corners
    top_left     = center_yx + forward_direction - right_direction
    top_right    = center_yx + forward_direction + right_direction
    bottom_left  = center_yx - forward_direction - right_direction
    bottom_right = center_yx - forward_direction + right_direction
    return top_left, top_right, bottom_left, bottom_right


def _plot_line(ax, color, point_1, point_b):
    ax.plot([point_1[1], point_b[1]], [point_1[0], point_b[0]], c=color, linewidth=1)


def _plot_box(ax, color, top_left, top_right, bottom_left, bottom_right):
    _plot_line(ax, color, top_left, bottom_left)     # Left
    _plot_line(ax, color, top_right, bottom_right)   # Right
    _plot_line(ax, color, top_left, top_right)       # Top
    _plot_line(ax, color, bottom_left, bottom_right) # Bottom


def plot_box(ax, center_yx, size_hw, orientation_01, color):
    # Check if the box has orientation. If not, set orientation to be 0
    has_orientation = orientation_01 is not None
    if not has_orientation:
        orientation_01 = 0

    # Get RBox corners
    top_left, top_right, bottom_left, bottom_right = _get_rbox_corners(
        center_yx, size_hw, orientation_01
    )

    # Plot boxes
    _plot_box(ax, color, top_left, top_right, bottom_left, bottom_right)

    # If it has orientation, plot a line from the center to the front of the vessel
    if has_orientation:
        front = (top_left + top_right) * 0.5
        _plot_line(ax, color, center_yx, front)
    # else:
    #     _plot_line(ax, color, top_left, bottom_right)
    #     _plot_line(ax, color, top_right, bottom_left)


def plot_point(ax, center_yx, color):
    y = np.array([center_yx[0]])
    x = np.array([center_yx[1]])
    ax.scatter(x, y, c=color, s=8, marker="x")


def plot_object(
    ax,
    center_yx: np.ndarray,
    size_hw: Optional[np.ndarray],
    rotation: Optional[np.ndarray],
    color: str,
    display_mode: str,
):
    if display_mode == WIDE_SQUARE:
        plot_box(ax, center_yx, np.array([100, 100]), None, color)
    elif display_mode == BOX:
        plot_box(ax, center_yx, size_hw, None, color)
    else:
        # Assuming RBOX
        has_hw = size_hw is not None
        if has_hw:
            plot_box(ax, center_yx, size_hw, rotation, color)
        else:
            plot_point(ax, center_yx, color)

def plot_annotation(ax, annotation: Annotation, display_mode: str):
    plot_object(ax, annotation.center_yx, annotation.size_hw, annotation.rotation, "r", display_mode)

def plot_detection(ax, detection: Detection, display_mode: str):
    plot_object(ax, detection.center_yx, detection.size_hw, detection.rotation, "b", display_mode)


