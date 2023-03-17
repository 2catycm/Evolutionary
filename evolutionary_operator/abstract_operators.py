from typing import Tuple, Union
import torch
import torch.nn as nn
import warnings
import benchmark_funs

# 超参数
hyper_param_names = ['population_size', 'step_size', 'torque', 'torque_prime']
hyper_param_is_real = [False, True, True, True]
hyper_param_dict = {k: v for v, k in enumerate(hyper_param_names)}
hyper_dimension = len(hyper_param_names)


def get_hyper_param(h: torch.Tensor, name: str):
    return h[:, hyper_param_dict[name]]
def set_hyper_param(h: torch.Tensor, name: str, value:Union[torch.Tensor, float]):
    h[:, hyper_param_dict[name]] = value
    return h

# 超超参数
hyper_hyper_param_names = ['initialization_strategy', 'mutation_strategy', 'selection_strategy', 'hyper_init_strategy', 'hyper_update_strategy',
                           'resample_if_bound', 'init_pop_std_dev_ratio',  # initialization_strategy 需要的超参数
                           'opponents_ratio', # selection_strategy 需要的超参数
                           'alpha', 'beta' # 
                           ]
hyper_hyper_param_is_real = [False, False, False, False, False,
                             False, True]
hyper_hyper_param_dict = {k: v for v, k in enumerate(hyper_hyper_param_names)}
hyper_hyper_dimension = len(hyper_hyper_param_names)


def get_hyper_hyper_param(hh: torch.Tensor, name: str):
    return hh[hyper_hyper_param_dict[name]]

def set_hyper_hyper_param(hh: torch.Tensor, name: str, value:float):
    hh[hyper_hyper_param_dict[name]] = value
    return hh

def hyper_hyper_param_from_dict(d: dict):
    hh = torch.zeros(hyper_hyper_dimension)
    for k, v in d.items():
        set_hyper_hyper_param(hh, k, v)
    return hh


# 演化算子
class EvolvingOperator(nn.Module):
    """Some Information about EvolvingOperator"""

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        """生成一个进化算子。
        Args:
            problem (benchmark_funs.BenchmarkFunction): 演化算子需要知道问题的维度和上下界，才能针对性计算。比如上下界越大，显然生成的初始种群的范围要大一些。
            hh (torch.Tensor, optional): 超超参数，表示演化算子选择的策略。维度应该为(hyper_hyper_dimension)。 Defaults to torch.zeros(0, hyper_dimension).
        """
        super().__init__()
        self.name: str = self.__class__.__name__
        self.description: str = self.__class__.__doc__ or "No description"
        self._device_test = nn.Parameter(torch.rand(1))  # 用于测试device
        self.problem: benchmark_funs.BenchmarkFunction = problem  # 问题
        self.hh = hh
        self.hyper_dimension: int = hyper_dimension  # 超参数的维度

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """抽象的进化算子，子类必须实现这个函数。
        Args:
            x (torch.Tensor): 输入的种群。维度应该为(population_size, self.dimension)。
            h (torch.Tensor): 输入的种群超参数。维度应该为(population_size, self.hyper_dimension)。
        Returns:
            torch.Tensor: 返回映射之后的新种群, 维度应该为 (population_size', self.dimension)。
            torch.Tensor: 返回映射之后的新超参数, 维度应该为 (population_size', self.hyper_dimension)。
            int: 使用的预算数量。
        """
        warnings.warn("警告：调用了抽象函数。")
        return x, h, 0

    def is_input_valid(self, x: torch.Tensor, h: torch.Tensor) -> bool:
        """判断输入是否合法。 子类无需实现，而且每次forward不应当调用这个函数，节省时间。
        Returns:
            bool: 返回是否合法
        """
        return h.shape[1] == self.hyper_dimension and self.problem.is_input_valid(x)

    def get_device(self) -> torch.device:
        return self._device_test.device
    
    def __str__(self) -> str:
        return f"{super().__str__()}{self.name}:({self.__dir__()})"


class EvolvingAlgorithm(EvolvingOperator):
    """演化算法是特殊的进化算子，直接通过问题构造，然后凭空输出一个最优解的种群。
    """
    pass
        