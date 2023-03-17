from typing import Tuple, Union
import torch
import torch.nn as nn
import warnings
import benchmark_funs

# 演化算法
class EvolvingAlgorithm(nn.Module):
    """Some Information about EvolvingAlgorithm"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction):
        """生成一个进化算法。
        Args:
            problem (benchmark_funs.BenchmarkFunction): 演化算子需要知道问题的维度和上下界，才能针对性计算。比如上下界越大，显然生成的初始种群的范围要大一些。
        """
        super().__init__()
        self.name: str = self.__class__.__name__
        self.description: str = self.__class__.__doc__ or "No description"
        self._device_test = nn.Parameter(torch.rand(1))  # 用于测试device
        self.problem: benchmark_funs.BenchmarkFunction = problem  # 问题


    def forward(self, hh:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """抽象的进化算法，子类必须实现这个函数。
            从hh中解析出演化算法的超参数，然后重新从problem开始，进行演化。
        Args:
            hh (torch.Tensor): 进化的策略。维度应该为(hyper_hyper_dimension)。
        Returns:
            torch.Tensor: 最优个体。
            torch.Tensor: 最优适应度下降曲线。
        """
        raise Exception("警告：调用了抽象函数。")


    def get_device(self) -> torch.device:
        return self._device_test.device
    
    def __str__(self) -> str:
        return f"{super().__str__()}{self.name}:({self.__dir__()})"