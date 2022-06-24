import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
from typing import List
from yolo_lib import (
    YOLOTileStack,
    SyntheticTestDataSet,
    SyntheticTrainDataSet,
    FakeDataCfg,
    EncoderConfig,
    ConfidenceUnawareObjectnessLoss,
    ResNet18Spec,
    BinaryFocalLoss,
    BasePerformanceMetric,
    RBoxPerformanceMetrics,
    DataAugmentation,
)
from yolo_lib.cfg import USE_GPU
from yolo_lib.detectors.cfg_types.detector_cfg import DetectorCfg, YOLOFCfg
from yolo_lib.detectors.cfg_types.head_cfg import LocalMatchingYOLOHeadCfg, OverlappingCellYOLOHeadCfg
from yolo_lib.detectors.yolo_heads.yolo_head import PointPotoMatchlossCfg
from yolo_lib.training_loop import TrainingLoop, TrainCallback
from yolo_lib.model_storage import ModelStorage
from yolo_lib.models.backbones import BackboneCfg


OUTPUT_DIR = "./out/synthtetic_experiments"
MODEL_STORAGE_DIR = "./model_storage"

# Image size
IMG_H = 512
IMG_W = 512


# Epochs
MAX_EPOCHS = 140
EPOCHS_PER_DISPLAY = 20
EPOCHS_PER_CHECKPOINT = 10


# Train set size
IMGS_PER_EPOCH = 1600
BATCH_SIZE = 16


# Test set size
TEST_SET_SIZE = 200
NUM_DISPLAYED_TESTS = 20
NUM_SILENT_TESTS = TEST_SET_SIZE - NUM_DISPLAYED_TESTS

model_checkpoints = ModelStorage(MODEL_STORAGE_DIR)

perfect_ds_cfg = FakeDataCfg(
    IMG_H,
    IMG_W,
    min_fake_vessels=0,
    max_fake_vessels=1,
    fake_vessel_detection_probability=0.0,
    fake_vessel_high_confidence_probability=0.0,
    min_real_vessels=0,
    max_real_vessels=3,
    real_vessel_detection_probability=1.0,
    real_vessel_high_confidence_probability=1.0,
    vessel_length_low=140,
    vessel_length_high=180,
    vessel_width_low=40,
    vessel_width_high=80,
    rear_end_width_multiplier=0.6,
    middle_width_multiplier=1.4,
    rotation_known_probability=1.0,
    hw_known_probability=1.0,
    yx_standard_deviation_px=0.0,
)

# Attributes are rarely known. Low positional standard deviation
low_noise_ds_cfg = FakeDataCfg(
    IMG_H,
    IMG_W,
    min_fake_vessels=0,
    max_fake_vessels=1,
    fake_vessel_detection_probability=0.01,
    fake_vessel_high_confidence_probability=0.0,
    min_real_vessels=0,
    max_real_vessels=3,
    real_vessel_detection_probability=0.99,
    real_vessel_high_confidence_probability=0.99,
    vessel_length_low=140,
    vessel_length_high=180,
    vessel_width_low=40,
    vessel_width_high=80,
    rear_end_width_multiplier=0.6,
    middle_width_multiplier=1.4,
    rotation_known_probability=0.2,
    hw_known_probability=0.2,
    yx_standard_deviation_px=1.0,
)

# Attributes are rarely known. High positional standard deviation
high_noise_ds_cfg = FakeDataCfg(
    IMG_H,
    IMG_W,
    min_fake_vessels=0,
    max_fake_vessels=1,
    fake_vessel_detection_probability=0.01,
    fake_vessel_high_confidence_probability=0.0,
    min_real_vessels=0,
    max_real_vessels=3,
    real_vessel_detection_probability=0.99,
    real_vessel_high_confidence_probability=0.99,
    vessel_length_low=140,
    vessel_length_high=180,
    vessel_width_low=40,
    vessel_width_high=80,
    rear_end_width_multiplier=0.6,
    middle_width_multiplier=1.4,
    rotation_known_probability=0.2,
    hw_known_probability=0.2,
    yx_standard_deviation_px=12.0,
)


# Create data loaders
low_noise_train_dl = DataLoader(
    dataset=SyntheticTrainDataSet(IMGS_PER_EPOCH, low_noise_ds_cfg),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=YOLOTileStack.stack_tiles,
)

high_noise_train_dl = DataLoader(
    dataset=SyntheticTrainDataSet(IMGS_PER_EPOCH, high_noise_ds_cfg),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=YOLOTileStack.stack_tiles,
)

test_dl = DataLoader(
    dataset=SyntheticTestDataSet(NUM_SILENT_TESTS, perfect_ds_cfg),
    batch_size=1,
    shuffle=False,
    collate_fn=YOLOTileStack.stack_tiles,
)

displayed_test_dl = DataLoader(
    dataset=SyntheticTestDataSet(NUM_DISPLAYED_TESTS, perfect_ds_cfg),
    batch_size=1,
    shuffle=False,
    collate_fn=YOLOTileStack.stack_tiles,
)


focal_loss = ConfidenceUnawareObjectnessLoss(
    BinaryFocalLoss(
        gamma=2,
        pos_loss_weight=1.0,
        neg_loss_weight=0.5,
    )
)

def get_anchor_priors_4():
    return torch.nn.Parameter(
        torch.tensor(
            [
                [2.1, 1.9],
                [2.0, 4.1],
                [4.2, 2.2],
                [4.3, 4.4],
            ],
            dtype=torch.float64
        )
    )

local_matching_model_cfg = YOLOFCfg(
    LocalMatchingYOLOHeadCfg(
        512,
        4,
        get_anchor_priors_4(),
        
        matchloss_objectness_weight=0.7,
        matchloss_yx_weight=0.3,
        matchloss_hw_weight=0.0,
        matchloss_sincos_weight=0.0,

        loss_objectness_weight=0.5,
        loss_yx_weight=0.12,
        loss_hw_weight=0.28,
        loss_sincos_weight=0.1,
    ),
    BackboneCfg(1, 18),
    EncoderConfig(512, 128, [2, 4, 6, 8]),
)

global_matching_model_cfg = YOLOFCfg(
    OverlappingCellYOLOHeadCfg(
        512,
        4,
        get_anchor_priors_4(),

        yx_multiplier=2.3,
        yx_match_threshold=2.0,

        matchloss_cfg=PointPotoMatchlossCfg(matchloss_objectness_weight=0.7, matchloss_yx_weight=0.3),

        loss_objectness_weight=0.5,
        loss_yx_weight=0.15,
        loss_hw_weight=0.15,
        loss_sincos_weight=0.2,
    ),
    BackboneCfg(1, 18),
    EncoderConfig(512, 128, [2, 4, 6, 8]),
)


@dataclass
class Exp:
    name: str
    detector_cfg: DetectorCfg
    data_augmentations: List[DataAugmentation] = None
    performance_metrics: BasePerformanceMetric = None
    callbacks: List[TrainCallback] = None
    lr_warmup_epochs: int = 10
    lr_decay_factor: float = 0.95
    base_lr: float = 0.0001

    def default(self, value, default):
        if value is None:
            return default
        else:
            return value

    def get_training_loop(self, max_epochs: int) -> TrainingLoop:
        model = self.detector_cfg.build()
        if USE_GPU:
            model = model.cuda()

        optimizer = Adam(list(model.parameters()), lr=self.base_lr)
        N = self.lr_warmup_epochs
        P = self.lr_decay_factor
        lr_scheduler_lambda = lambda i: ((i+1) / N) if i < N else P ** (i + 1 - N)
        return TrainingLoop(
            self.name,
            model,
            optimizer,
            self.default(self.data_augmentations, []),
            self.default(self.performance_metrics, RBoxPerformanceMetrics(40, 40)),
            self.default(self.callbacks, []),
            lr_scheduler_lambda,
            max_epochs,
        )

experiment_specs = [
    {
        "train_dl": low_noise_train_dl,
        "exp": Exp("LocalMatching+HighNoiseData", local_matching_model_cfg),
    },
    {
        "train_dl": high_noise_train_dl,
        "exp": Exp("LocalMatching+HighNoiseData", local_matching_model_cfg),
    },
    {
        "train_dl": low_noise_train_dl,
        "exp": Exp("GlobalMatching+HighNoiseData", global_matching_model_cfg),
    },
    {
        "train_dl": high_noise_train_dl,
        "exp": Exp("GlobalMatching+HighNoiseData", global_matching_model_cfg),
    },
]

torch.cuda.empty_cache()

for e in experiment_specs:
    train_dl = e["train_dl"]
    exp: Exp = e["exp"]
    train_loop = exp.get_training_loop(MAX_EPOCHS)
    train_loop.run(
        EPOCHS_PER_DISPLAY,
        EPOCHS_PER_CHECKPOINT,
        model_checkpoints,
        train_dl,
        test_dl,
        displayed_test_dl,
        OUTPUT_DIR,
    )
