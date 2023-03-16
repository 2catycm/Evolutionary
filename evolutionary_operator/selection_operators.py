import math
from typing import Tuple
import numpy as np
from abstract_operators import *
import torch
import torch.nn as nn
import warnings
import benchmark_funs

class OffspringOnly(EvolvingOperator):
    """只选择子代作为下一代。
        理论依据：认为父代自然死亡了。
        传参注意：population_size为整数。传入的x后面的是offspring
    """

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        population_size = int(get_hyper_param(h, 'population_size')[0])
        x = x[-population_size:]        
        return x, h, 0

class TournamentSelection(EvolvingOperator):
    