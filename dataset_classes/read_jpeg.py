import torch
import matplotlib.pyplot as plt

def read_jpeg(path: str, tile_size: int) -> torch.Tensor:
    image_np = plt.imread(path, format="jpeg")
    image_torch = torch.tensor(image_np)
    image_torch_avg = image_torch.sum(dim=2) / 3
    assert image_torch_avg.shape == (tile_size, tile_size)
    return image_torch_avg

