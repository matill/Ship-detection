from yolo_lib.data.annotation import AnnotationBlock
from torch import Tensor


class SimilarityMetric:
    def get_matchloss(
        self,
        post_activation_b_posi: Tensor, # post_acivation grid, indexed by prior_mutliplier_b_posi_idx
        prior_mutliplier_b_posi_idx: Tensor, # Indices of grid cells with positive centerness priors
        num_posi_b: int, # Number of grid-cells with positive centerness prior
        annotations_b: AnnotationBlock, # Annotations within the image (not the entire batch)
        downsample_factor: float,
    ) -> Tensor:
        raise NotImplementedError

