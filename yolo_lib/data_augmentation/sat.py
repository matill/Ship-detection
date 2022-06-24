import torch
from yolo_lib.cfg import DEVICE
from yolo_lib.data.dataclasses import YOLOTileStack
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector
from .data_augmentation import DataAugmentation


class SAT(DataAugmentation):
    """Self-adversarial training"""

    def __init__(self, static_p: float, epochs_without: int, step_size_min: float, step_size_max: float) -> None:
        self.static_p = static_p
        self.step_size_min = step_size_min
        self.step_size_range = step_size_max - step_size_min
        self.epochs_without = epochs_without

    def get_probability(self, epoch: int) -> float:
        if epoch < self.epochs_without:
            return 0.0
        else:
            return self.static_p

    def apply(self, tiles: YOLOTileStack, model: BaseDetector) -> YOLOTileStack:
        num_images, channels, h, w = tiles.images.shape
        step_sizes = torch.rand((num_images, 1, 1, 1), device=DEVICE) * self.step_size_range + self.step_size_min
        tiles.images.requires_grad = True
        loss, loss_subterms = model.compute_loss(tiles)
        images = tiles.images
        loss.backward(inputs=[images])


        image_grads_sum_per_image = images.grad.abs().sum(dim=(1, 2, 3))
        assert image_grads_sum_per_image.shape == (num_images, )
        num_pixels = channels * h * w
        scales = num_pixels / image_grads_sum_per_image
        grads_scaled = images.grad * scales[:, None, None, None]
        assert grads_scaled.shape == images.shape
        images = images + grads_scaled * step_sizes
        return YOLOTileStack(images.detach(), tiles.img_idxs, tiles.annotations)

