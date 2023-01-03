from __future__ import annotations
from dataclasses import dataclass
from torch.nn import Module
import os
import torch


@dataclass
class ModelStorage:
    storage_folder: str

    def __post_init__(self):
        os.makedirs(self.storage_folder, exist_ok=True)

    def store_model(self, model: Module, name: str, tag: str = "latest"):
        path = self._get_path(name, tag)
        print(f"Storing {path}")
        torch.save(model.state_dict(), path)

    def load_model(self, model: Module, name: str, tag: str = "latest", not_exists_ok=False):
        try:
            path = self._get_path(name, tag)
            print(f"Loading {path}")
            model.load_state_dict(torch.load(path))
        except FileNotFoundError as e:
            print(f"WARNING: Tried to load {name}:{tag}, but got FileNotFoundError.\n{e}")
            if not not_exists_ok:
                raise e
        except Exception as e:
            print("Error when loading ", name)
            raise e

    def _get_path(self, model_name: str, tag: str) -> str:
        path = os.path.join(self.storage_folder, f"{model_name}:{tag}.pt")
        return path

    def list_models(self):
        full_names = os.listdir(self.storage_folder)
        for n in full_names:
            print(n)

