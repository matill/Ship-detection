from dataclasses import dataclass
from yolo_lib.models.blocks.attention import AttentionCfg, NoAttentionCfg
from yolo_lib.models.backbones import BackboneCfg
from yolo_lib.detectors.cfg_types.dilated_encoder_cfg import EncoderConfig
from yolo_lib.detectors.yolo_heads.yolo_head import YOLOHeadCfg
from yolo_lib.detectors.managed_architectures.attention_yolof import AttentionYOLOF
from yolo_lib.detectors.managed_architectures.auxiliary_head_yolof import AuxiliaryHeadYOLOF
from yolo_lib.detectors.managed_architectures.base_detector import BaseDetector


class DetectorCfg:
    def build(self) -> BaseDetector:
        pass

@dataclass
class YOLOFCfg(DetectorCfg):
    head_cfg: YOLOHeadCfg   
    backbone_cfg: BackboneCfg
    dileated_encoder_cfg: EncoderConfig

    def build(self) -> AttentionYOLOF:
        return AttentionYOLOF(
            self.backbone_cfg.build(),
            self.dileated_encoder_cfg,
            NoAttentionCfg(),
            self.head_cfg.build(self.dileated_encoder_cfg.out_channels)
        )

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

@dataclass
class AuxiliaryHeadYOLOFCfg(DetectorCfg):
    main_head_cfg: YOLOHeadCfg
    auxiliary_head_cfg: YOLOHeadCfg
    auxiliary_loss_weight: float
    backbone_cfg: BackboneCfg
    dileated_encoder_cfg: EncoderConfig
    attention_cfg: AttentionCfg

    def build(self) -> AuxiliaryHeadYOLOF:
        return AuxiliaryHeadYOLOF(
            self.backbone_cfg.build(),
            self.dileated_encoder_cfg,
            self.attention_cfg,
            self.main_head_cfg.build(self.dileated_encoder_cfg.out_channels),
            self.auxiliary_head_cfg.build(self.dileated_encoder_cfg.out_channels),
            self.auxiliary_loss_weight,
        )
