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
    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        self.opponents_ratio = get_hyper_hyper_param(
            hh, "opponents_ratio")

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        population_size = int(get_hyper_param(h, 'population_size')[0])
        overloaded_population_size = x.shape[0]
        opponents_num = math.ceil(
            overloaded_population_size * self.opponents_ratio)
        fitnesses = self.problem(x)
        wins = torch.zeros(overloaded_population_size,
                           device=x.device, dtype=torch.int32)
        for i in range(opponents_num):
            opponents = torch.randint(
                0, overloaded_population_size, (overloaded_population_size,), device=x.device)
            wins += (fitnesses < fitnesses[opponents])
        # winners = torch.argsort(wins, descending=True)[:population_size]
        # 使用topk比argsort快
        winners = torch.topk(wins, population_size,
                             largest=True, sorted=False)[1]
        return x[winners], h, overloaded_population_size


strategies = [
    # OffspringOnly,
    lambda problem: TournamentSelection(
        problem, hyper_hyper_param_from_dict({'opponents_ratio': 1})),
    lambda problem: TournamentSelection(
        problem, hyper_hyper_param_from_dict({'opponents_ratio': 0.75})),
    lambda problem: TournamentSelection(
        problem, hyper_hyper_param_from_dict({'opponents_ratio': 0.5})),
    lambda problem: TournamentSelection(
        problem, hyper_hyper_param_from_dict({'opponents_ratio': 0.25})),
    lambda problem: TournamentSelection(
        problem, hyper_hyper_param_from_dict({'opponents_ratio': 0.1})),
    lambda problem: TournamentSelection(
        problem, hyper_hyper_param_from_dict({'opponents_ratio': 0.05})),
]
strategy_names = ["锦标赛q=1", "锦标赛q=0.75", "锦标赛q=0.5", "锦标赛q=0.25", "锦标赛q=0.1", "锦标赛q=0.05"]

assert len(strategies) == len(strategy_names)
