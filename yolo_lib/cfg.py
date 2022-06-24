import torch
import json


# Main configurable constants
SAFE_MODE = True
UNSAFE_MODE = not SAFE_MODE

_FORCE_USE_CPU = False

PRETRAINED_BACKBONE = True

# Computed properties
if _FORCE_USE_CPU:
    USE_GPU = False
elif torch.cuda.is_available():
    USE_GPU = True
else:
    USE_GPU = False

DEVICE = "cuda" if USE_GPU else "cpu"

# Print configs and computed properites
print("Configuration: ",
    json.dumps(
        {
            "SAFE_MODE": SAFE_MODE,
            "UNSAFE_MODE": UNSAFE_MODE,
            "_FORCE_USE_CPU": _FORCE_USE_CPU,
            "USE_GPU": USE_GPU,
            "DEVICE": DEVICE,
            "PRETRAINED_BACKBONE": PRETRAINED_BACKBONE,
        },
        indent=2
    )
)
