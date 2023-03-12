import math
from typing import Tuple
import numpy as np
from abstract_operators import *
import torch
import torch.nn as nn
import warnings
import benchmark_funs


class UniformReset(EvolvingOperator):
    """种群基因变异算子。使用均匀分布随机重置种群的基因。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        step_size = get_hyper_param(h, 'step_size').reshape(-1, 1)
        self.mut_prob = step_size / (math.sqrt((self.problem.ub-self.problem.lb)**2/12)*x.shape[0])
        r = torch.rand(x.shape[0], x.shape[1], device=x.device)
        replace_indexes = r < self.mut_prob
        x[replace_indexes]=torch.rand(int(replace_indexes.sum()), device=x.device) * \
            (self.problem.ub - self.problem.lb) + self.problem.lb
        return x, h, 0


class NormalMutation(EvolvingOperator):
    """种群基因变异算子。使用正态分布移动种群的基因。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        self.resample_if_bound: bool = bool(
            get_hyper_hyper_param(hh, 'resample_if_bound'))  # 是否在初始化时，如果超出边界，则重新采样。如果不重新采样，则将超出边界的值设为边界值。

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        step_size = get_hyper_param(h, 'step_size').reshape(-1, 1)
        y = torch.randn(x.shape[0], x.shape[1],
                        device=x.device) * step_size + x
        if self.resample_if_bound:
            bad_indexes = (y > self.problem.ub) | (
                y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
            while bad_indexes.any():
                y[bad_indexes] = torch.randn(
                    int(bad_indexes.sum()), device=x.device) * step_size + x
                bad_indexes = (y > self.problem.ub) | (
                    y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb
            bad_indexes = torch.isnan(y) | torch.isinf(y)
            y[bad_indexes] = x[bad_indexes]

        return y, h, 0


class CauthyMutation(EvolvingOperator):
    """种群基因变异算子。使用柯西分布移动种群的基因。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        self.resample_if_bound: bool = bool(
            get_hyper_hyper_param(hh, 'resample_if_bound'))  # 是否在初始化时，如果超出边界，则重新采样。如果不重新采样，则将超出边界的值设为边界值。

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        step_size = get_hyper_param(h, 'step_size').reshape(-1, 1)
        y = torch.tan((torch.rand(x.shape[0], x.shape[1],
                                  device=x.device)-0.5)*torch.pi) * step_size + x
        if self.resample_if_bound:
            bad_indexes = (y > self.problem.ub) | (
                y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
            while bad_indexes.any():
                y[bad_indexes] = torch.tan((torch.rand(int(bad_indexes.sum()),
                                                       device=x.device)-0.5)*torch.pi) * step_size + x
                bad_indexes = (y > self.problem.ub) | (
                    y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb
            bad_indexes = torch.isnan(y) | torch.isinf(y)
            y[bad_indexes] = x[bad_indexes]

        return y, h, 0


class StudentTMutation(EvolvingOperator):
    """种群基因变异算子。使用学生t分布移动种群的基因。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        self.resample_if_bound: bool = bool(
            get_hyper_hyper_param(hh, 'resample_if_bound'))  # 是否在初始化时，如果超出边界，则重新采样。如果不重新采样，则将超出边界的值设为边界值。

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        step_size = get_hyper_param(h, 'step_size').reshape(-1, 1)
        # 使得 dof/dof+2 = step_size**(2)
        # dof = (2/(step_size**(-2) - 1)).to(torch.int32)
        dof = max(int(min(2/(step_size.mean()**(-2) - 1), 20)), 1)
        # dof[dof<1] = 1
        y1 = torch.randn(x.shape[0], self.problem.dimension)
        y2 = torch.sum(torch.randn(dof, x.shape[0], self.problem.dimension), dim=0).reshape(
            x.shape[0], self.problem.dimension)
        y = y1/(y2**0.5)

        y = y * step_size + x

        if self.resample_if_bound:
            bad_indexes = (y > self.problem.ub) | (
                y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
            while bad_indexes.any():
                N = int(bad_indexes.sum())
                y1 = torch.randn(N)
                y2 = torch.sum(torch.randn(dof, N), dim=0).reshape(N)
                y3 = y1/(y2**0.5)

                y[bad_indexes] = y3 * step_size + x
                bad_indexes = (y > self.problem.ub) | (
                    y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb
            bad_indexes = torch.isnan(y) | torch.isinf(y)
            y[bad_indexes] = x[bad_indexes]

        return y, h, 0

strategies = [UniformReset, NormalMutation, CauthyMutation, StudentTMutation]
strategy_names = ["均匀分布重置", "高斯移动", "柯西移动", "t移动"]
