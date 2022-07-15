from dataclasses import dataclass
from yolo_lib.models.blocks.attention import AttentionCfg, NoAttentionCfg
from yolo_lib.models.backbones import BackboneCfg
from yolo_lib.detectors.cfg_types.dilated_encoder_cfg import EncoderConfig
from yolo_lib.detectors.yolo_heads.yolo_head import YOLOHeadCfg
from yolo_lib.detectors.managed_architectures.attention_yolof import AttentionYOLOF
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector


class DetectorCfg:
    def build(self) -> BaseDetector:
        pass

@dataclass
class AttentionYOLOFCfg(DetectorCfg):
    head_cfg: YOLOHeadCfg
    backbone_cfg: BackboneCfg
    dileated_encoder_cfg: EncoderConfig
    attention_cfg: AttentionCfg

    def build(self) -> AttentionYOLOF:
        return AttentionYOLOF(
            self.backbone_cfg.build(),
            self.dileated_encoder_cfg,
            self.attention_cfg,
            self.head_cfg.build(self.dileated_encoder_cfg.out_channels)
        )
