import torch
import torch.nn as nn
import warnings

from benchmark_funs import BenchmarkFunction

class IntegerBenchmarkFunction(BenchmarkFunction):
    def __init__(self):
        super().__init__()
        # 要求必须每个值有取值范围。
        self.lb: torch.Tensor = torch.zeros(self.dimension, dtype=torch.int32)
        self.ub: torch.Tensor = torch.zeros(self.dimension, dtype=torch.int32)

class LargestSum(IntegerBenchmarkFunction):
    """整数规划问题，最小和问题。"""

    def __init__(self, dimension: int = 30):
        super().__init__()
        self.dimension: int = dimension
        self.lb: torch.Tensor = torch.zeros(self.dimension, dtype=torch.int32)
        self.ub: torch.Tensor = torch.arange(self.dimension, dtype=torch.int32)+1
        self.optinum: torch.Tensor = torch.zeros(self.dimension, dtype=torch.int32)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=1)