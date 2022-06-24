from torch import Tensor
from yolo_lib.data.annotation import AnnotationBlock


class SpatialPrior:
    def compute(self, img_h: int, img_w: int, annotations: AnnotationBlock, downsample_factor: int) -> Tensor:
        raise NotImplementedError

