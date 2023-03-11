from typing import Tuple
import numpy as np
from abstract_operators import *
import torch
import torch.nn as nn
import warnings
import benchmark_funs


class UniformInitialization(EvolvingOperator):
    """种群初始化算子。使用均匀分布初始化种群。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction):
        super().__init__(problem)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        y = torch.rand(x.shape[0], self.problem.dimension, device=x.device) * \
            (self.problem.ub - self.problem.lb) + self.problem.lb
        return y, h, 0


class NormalInitialization(EvolvingOperator):
    """种群初始化算子。使用正态分布初始化种群。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        # 根据3σ原则，初始种群的标准差应该是上下界的1/6
        self.init_pop_std_dev = (problem.ub-problem.lb) / 6
        # 根据传入的超超参数调整
        self.init_pop_std_dev *= get_hyper_hyper_param(
            hh, "init_pop_std_dev_ratio")
        self.mean = problem.lb+(problem.ub-problem.lb)/2
        self.resample_if_bound: bool = bool(
            get_hyper_hyper_param(hh, 'resample_if_bound'))  # 是否在初始化时，如果超出边界，则重新采样。如果不重新采样，则将超出边界的值设为边界值。

    def forward(self, x: torch.Tensor, _h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        y = torch.randn(x.shape[0], self.problem.dimension,
                        device=x.device) * self.init_pop_std_dev + self.mean
        if self.resample_if_bound:
            bad_indexes = (y > self.problem.ub) | (y < self.problem.lb)
            while bad_indexes.any():
                y[bad_indexes] = torch.randn(
                    int(bad_indexes.sum()), device=x.device) * self.init_pop_std_dev + self.mean
                bad_indexes = (y > self.problem.ub) | (y < self.problem.lb)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb

        return y, _h, 0


class CauthyInitialization(EvolvingOperator):
    """种群初始化算子。使用柯西分布初始化种群。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        # 最大值一半处的一般宽度的尺度参数 是柯西分布的参数。
        self.init_pop_std_dev = (problem.ub-problem.lb) / 4
        # 根据传入的超超参数调整
        self.init_pop_std_dev *= get_hyper_hyper_param(
            hh, "init_pop_std_dev_ratio")
        self.mean = problem.lb+(problem.ub-problem.lb)/2
        self.resample_if_bound: bool = bool(
            get_hyper_hyper_param(hh, 'resample_if_bound'))  # 是否在初始化时，如果超出边界，则重新采样。如果不重新采样，则将超出边界的值设为边界值。

    def forward(self, x: torch.Tensor, _h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        y = torch.tan((torch.rand(x.shape[0], self.problem.dimension,
                                  device=x.device)-0.5)*torch.pi) * self.init_pop_std_dev + self.mean
        if self.resample_if_bound:
            bad_indexes = (y > self.problem.ub) | (y < self.problem.lb)
            while bad_indexes.any():
                y[bad_indexes] = torch.tan((torch.rand(int(bad_indexes.sum()),
                                                       device=x.device)-0.5)*torch.pi) * self.init_pop_std_dev + self.mean
                bad_indexes = (y > self.problem.ub) | (y < self.problem.lb)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb

        return y, _h, 0


# strategies
# strategy_names
