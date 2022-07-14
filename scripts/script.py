

from dataclasses import dataclass
from email.policy import default
from typing import Callable, Dict, Optional


@dataclass
class Script:
    funcs: Dict[str, Callable[[], None]]

    def run(self, func_name: str):
        assert func_name in self.funcs, f"Invalid func_name: {func_name} not in {list(self.funcs)}"
        self.funcs[func_name]()

