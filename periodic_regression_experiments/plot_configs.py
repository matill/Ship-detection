
from dataclasses import dataclass
from typing import List, Tuple
from models.dv_regression import PDV, NormalizePDV, ProjectedPDV, ProjectedPDV_175
from models.grey_code import GreyCode3PlusOffset, GreyCode5PlusOffset, GreyCode7PlusOffset
from models.logistic_regression import LogisticRegression
from models.circular_smooth_label import CSL_128Bit_01Gaussian, CSL_128Bit_01WindowWidth_PlusOffset, CSL_128Bit_03Gaussian, CSL_128Bit_03WindowWidth_PlusOffset, CSL_256Bit_01Gaussian, CSL_256Bit_03Gaussian, CSL_32Bit_01Gaussian, CSL_32Bit_01WindowWidth_PlusOffset, CSL_32Bit_03Gaussian, CSL_32Bit_03WindowWidth_PlusOffset, CSL_256Bit_01WindowWidth_PlusOffset, CSL_256Bit_03WindowWidth_PlusOffset
from models.base_model import PeriodicRegression


@dataclass
class PlotConfig:
    name: str
    members: List[Tuple[PeriodicRegression, str]]


PLOT_CONFIGS = [
    PlotConfig(
        "rotation_adv_comparison",
        [
            (PDV, "λ₁=1,      λ₂=1"),
            (ProjectedPDV, "λ₁=0.5,   λ₂=1.5"),
            (ProjectedPDV_175, "λ₁=0.25, λ₂=1.75"),
            (NormalizePDV, "Unit normalized"),
        ]
    ),
    PlotConfig(
        "rotation_gcl_comparison",
        [
            (GreyCode3PlusOffset, "GCL (3)"),
            (GreyCode5PlusOffset, "GCL (5)"),
            (GreyCode7PlusOffset, "GCL (7)"),
        ]
    ),
    PlotConfig(
        "rotation_csl_comparison",
        [
            # Traingle window function
            (CSL_32Bit_01WindowWidth_PlusOffset, "Tr. (32 bit, 0.1)"),
            (CSL_32Bit_03WindowWidth_PlusOffset, "Tr. (32 bit, 0.3)"),
            (CSL_128Bit_01WindowWidth_PlusOffset, "Tr. (128 bit, 0.1)"),
            (CSL_128Bit_03WindowWidth_PlusOffset, "Tr. (128 bit, 0.3)"),
            (CSL_256Bit_01WindowWidth_PlusOffset, "Tr. (256 bit, 0.1)"),
            (CSL_256Bit_03WindowWidth_PlusOffset, "Tr. (256 bit, 0.3)"),

            # Gaussian window function
            (CSL_32Bit_01Gaussian, "Ga. (32 bit, 0.1)"),
            (CSL_128Bit_01Gaussian, "Ga. (128 bit, 0.1)"),
            (CSL_256Bit_01Gaussian, "Ga. (256 bit, 0.1)"),
            (CSL_32Bit_03Gaussian, "Ga. (32 bit, 0.3)"),
            (CSL_128Bit_03Gaussian, "Ga. (128 bit, 0.3)"),
            (CSL_256Bit_03Gaussian, "Ga. (256 bit, 0.3)"),
        ]
    ),
    PlotConfig(
        "rotation_csl_gaussian_comparison",
        [
            # Gaussian window function
            (CSL_32Bit_01Gaussian, "Ga. (32 bit, 0.1)"),
            (CSL_128Bit_01Gaussian, "Ga. (128 bit, 0.1)"),
            (CSL_256Bit_01Gaussian, "Ga. (256 bit, 0.1)"),
            (CSL_32Bit_03Gaussian, "Ga. (32 bit, 0.3)"),
            (CSL_128Bit_03Gaussian, "Ga. (128 bit, 0.3)"),
            (CSL_256Bit_03Gaussian, "Ga. (256 bit, 0.3)"),
        ]
    ),
    PlotConfig(
        "rotation_csl_triangle_comparison",
        [
            # Traingle window function
            (CSL_32Bit_01WindowWidth_PlusOffset, "Tr. (32 bit, 0.1)"),
            (CSL_32Bit_03WindowWidth_PlusOffset, "Tr. (32 bit, 0.3)"),
            (CSL_128Bit_01WindowWidth_PlusOffset, "Tr. (128 bit, 0.1)"),
            # (CSL_128Bit_03WindowWidth_PlusOffset, "Tr. (128 bit, 0.3)"),
            (CSL_256Bit_01WindowWidth_PlusOffset, "Tr. (256 bit, 0.1)"),
            # (CSL_256Bit_03WindowWidth_PlusOffset, "Tr. (256 bit, 0.3)"),
        ]
    ),
    PlotConfig(
        "rotation_csl_reduced_comparison",
        [
            # Notes:
            # Ga. (128 bit, 0.3) < Ga. (32 bit, 0.3)
            # Ga. (256 bit, 0.3) < Ga. (32 bit, 0.3)
            # Tr. (256 bit, 0.3) < Tr. (32 bit, 0.3)
            # Tr. (128 bit, 0.3) < Tr. (32 bit, 0.3)

            # Traingle window function
            (CSL_32Bit_01WindowWidth_PlusOffset, "Tr. (32 bit, 0.1)"),
            (CSL_32Bit_03WindowWidth_PlusOffset, "Tr. (32 bit, 0.3)"),
            (CSL_128Bit_01WindowWidth_PlusOffset, "Tr. (128 bit, 0.1)"),
            (CSL_128Bit_03WindowWidth_PlusOffset, "Tr. (128 bit, 0.3)"),
            (CSL_256Bit_01WindowWidth_PlusOffset, "Tr. (256 bit, 0.1)"),
            (CSL_256Bit_03WindowWidth_PlusOffset, "Tr. (256 bit, 0.3)"),

            # Gaussian window function
            (CSL_32Bit_01Gaussian, "Ga. (32 bit, 0.1)"),
            (CSL_128Bit_01Gaussian, "Ga. (128 bit, 0.1)"),
            (CSL_256Bit_01Gaussian, "Ga. (256 bit, 0.1)"),
            (CSL_32Bit_03Gaussian, "Ga. (32 bit, 0.3)"),
            (CSL_128Bit_03Gaussian, "Ga. (128 bit, 0.3)"),
            (CSL_256Bit_03Gaussian, "Ga. (256 bit, 0.3)"),
        ]
    ),
    PlotConfig(
        "rotation_summary",
        [
            (ProjectedPDV, "ADV"),

            (GreyCode3PlusOffset, "GCL (3)"),

            (CSL_32Bit_01WindowWidth_PlusOffset, "NAME"),
            (CSL_32Bit_03WindowWidth_PlusOffset, "NAME"),
            # CSL_128Bit_01WindowWidth_PlusOffset,
            (CSL_128Bit_03WindowWidth_PlusOffset, "NAME"),

            (LogisticRegression, "NAME"),
        ]
    ),
]
