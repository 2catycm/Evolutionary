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
            bad_indexes = (y > self.problem.ub) | (
                y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
            while bad_indexes.any():
                y[bad_indexes] = torch.randn(
                    int(bad_indexes.sum()), device=x.device) * self.init_pop_std_dev + self.mean
                bad_indexes = (y > self.problem.ub) | (
                    y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb
            y[torch.isnan(y) | torch.isinf(y)] = self.mean

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
            bad_indexes = (y > self.problem.ub) | (
                y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
            while bad_indexes.any():
                y[bad_indexes] = torch.tan((torch.rand(int(bad_indexes.sum()),
                                                       device=x.device)-0.5)*torch.pi) * self.init_pop_std_dev + self.mean
                bad_indexes = (y > self.problem.ub) | (
                    y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb
            y[torch.isnan(y) | torch.isinf(y)] = self.mean

        return y, _h, 0


class StudentTInitialization(EvolvingOperator):
    """种群初始化算子。使用学生t分布初始化种群。"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        # 自由度
        self.dof = max(1, int((problem.ub-problem.lb)*5e-2*get_hyper_hyper_param(
            hh, "init_pop_std_dev_ratio")))
        self.init_pop_std_dev = self.dof / \
            (self.dof+2) * (problem.ub-problem.lb) / 4
        self.mean = problem.lb+(problem.ub-problem.lb)/2
        self.resample_if_bound: bool = bool(
            get_hyper_hyper_param(hh, 'resample_if_bound'))  # 是否在初始化时，如果超出边界，则重新采样。如果不重新采样，则将超出边界的值设为边界值。

    def forward(self, x: torch.Tensor, _h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        y1 = torch.randn(x.shape[0], self.problem.dimension)
        y2 = torch.sum(torch.randn(self.dof, x.shape[0], self.problem.dimension), dim=0).reshape(
            x.shape[0], self.problem.dimension)
        y = y1/(y2**0.5)
        y = y * self.init_pop_std_dev + self.mean
        if self.resample_if_bound:
            bad_indexes = (y > self.problem.ub) | (
                y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
            while bad_indexes.any():
                N = int(bad_indexes.sum())
                y1 = torch.randn(N)
                y2 = torch.sum(torch.randn(self.dof, N), dim=0).reshape(N)
                y3 = y1/(y2**0.5)
                y[bad_indexes] = y3 * self.init_pop_std_dev + self.mean
                bad_indexes = (y > self.problem.ub) | (
                    y < self.problem.lb) | torch.isnan(y) | torch.isinf(y)
        else:
            y[y > self.problem.ub] = self.problem.ub
            y[y < self.problem.lb] = self.problem.lb
            y[torch.isnan(y) | torch.isinf(y)] = self.mean

        return y, _h, 0

# # TODO 整体的一个初始化算子，可以接受h里面的参数，来决定使用哪种初始化算子。
# class Initialization(EvolvingOperator):
#     pass

strategies = [lambda p: UniformInitialization(p),
              lambda p: NormalInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 1)),
              lambda p: NormalInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 0.5)),
              lambda p: NormalInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 0.25)),

              lambda p: CauthyInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 1/4)),
              lambda p: CauthyInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 1/4)),
              lambda p: CauthyInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 1/16)),


              lambda p: StudentTInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 2)),
              lambda p: StudentTInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 1)),
              lambda p: StudentTInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 0.75)),

              lambda p: StudentTInitialization(p, set_hyper_hyper_param(
                  torch.zeros(hyper_hyper_dimension), "init_pop_std_dev_ratio", 0.25)),
              ]
strategy_names = ['均匀分布', '正态分布1置信区间', '正态分布0.5置信区间', '正态分布0.25置信区间', '柯西分布1/4位距',
                          '柯西分布1/16位距', '柯西分布1/64位距',  't分布0.75自由度',  't分布2自由度', 't分布1自由度', 't分布0.25自由度', ]
