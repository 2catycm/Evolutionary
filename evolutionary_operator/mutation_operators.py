from typing import Tuple
import numpy as np
from abstract_operators import *
import torch
import torch.nn as nn
import warnings
import benchmark_funs

class NormalMutation(EvolvingOperator):
    """种群基因变异算子。使用正态分布移动种群的基因。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        self.resample_if_bound: bool = bool(
            get_hyper_hyper_param(hh, 'resample_if_bound'))  # 是否在初始化时，如果超出边界，则重新采样。如果不重新采样，则将超出边界的值设为边界值。

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        y = torch.randn(x.shape[0], x.shape[1],
                        device=x.device) * get_hyper_param(h, 'step_size') + x
        if self.resample_if_bound:
            bad_indexes = (y > self.problem.ub) | (
                y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
            while bad_indexes.any():
                y[bad_indexes] = torch.randn(
                    int(bad_indexes.sum()), device=x.device) * get_hyper_param(h, 'step_size') + x
                bad_indexes = (y > self.problem.ub) | (
                    y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb
            bad_indexes = torch.isnan(y) | torch.isinf(y)
            y[bad_indexes] = x[bad_indexes]

        return y, h, 0




strategy_names = ["均匀分布重置", "高斯移动", "柯西移动", "t移动"]