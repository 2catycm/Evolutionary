import math
from typing import Tuple
import numpy as np
from abstract_operators import *
import torch
import torch.nn as nn
import warnings
import benchmark_funs


class AdaptiveInit(EvolvingOperator):

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        # 从超超参数中获取初始化种群的尺度参数
        self.alpha = get_hyper_hyper_param(hh, 'alpha')
        self.beta = get_hyper_hyper_param(hh, 'beta')
        # 从问题中获取上下界和预算
        self.problem_interval = self.problem.ub - self.problem.lb
        self.problem_budget = self.problem.get_budget()
        self.dimention = self.problem.dimension
        # 计算超参数
        self.population_size = max(1, math.floor(
            self.alpha*self.problem_budget*self.problem_interval*(100/100)
        ))
        # self.step_size = torch.ones((self.population_size, 1),
        #     device=hh.device)*self.beta * self.problem_budget / self.population_size
        self.step_size = torch.ones((self.population_size, 1),
            device=hh.device)*self.beta*self.problem_budget*self.problem_interval*(3/3)

        self.torque = 1 / math.sqrt(2 * math.sqrt(self.dimention));
        self.torque_prime = 1 / math.sqrt(2 * self.dimention);
        
        

    def forward(self, _: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        h = torch.zeros((self.population_size, self.hyper_dimension), 
                       device=h.device)
        set_hyper_param(h, 'step_size', self.step_size.flatten())
        set_hyper_param(h, 'population_size', self.population_size)
        set_hyper_param(h, 'torque', self.torque)
        set_hyper_param(h, 'torque_prime', self.torque_prime)
        return _, h, 0
    

class BsfSuggestion(EvolvingOperator):

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)
        # 从超超参数中获取初始化种群的尺度参数
        self.beta = get_hyper_hyper_param(hh, 'beta')
        # 从问题中获取上下界和预算
        self.problem_interval = self.problem.ub - self.problem.lb
        self.problem_budget = self.problem.get_budget()
        self.dimention = self.problem.dimension
        # 计算超参数
        self.population_size = 100
        self.step_size = torch.ones((self.population_size, 1),
            device=hh.device)*3*self.beta
        self.torque = 1 / math.sqrt(2 * math.sqrt(self.dimention));
        self.torque_prime = 1 / math.sqrt(2 * self.dimention);
        
        

    def forward(self, _: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        h = torch.zeros((self.population_size, self.hyper_dimension), 
                       device=h.device)
        set_hyper_param(h, 'step_size', self.step_size.flatten())
        set_hyper_param(h, 'population_size', self.population_size)
        set_hyper_param(h, 'torque', self.torque)
        set_hyper_param(h, 'torque_prime', self.torque_prime)
        return _, h, 0

strategies = [
    # lambda problem: AdaptiveInit(problem, hyper_hyper_param_from_dict({'alpha': 0.5, 'beta': 0.5})),
              lambda problem: BsfSuggestion(problem, hyper_hyper_param_from_dict({'beta': 0.0625})),
              lambda problem: BsfSuggestion(problem, hyper_hyper_param_from_dict({'beta': 0.125})),
              lambda problem: BsfSuggestion(problem, hyper_hyper_param_from_dict({'beta': 0.25})),
              lambda problem: BsfSuggestion(problem, hyper_hyper_param_from_dict({'beta': 0.5})),
              lambda problem: BsfSuggestion(problem, hyper_hyper_param_from_dict({'beta': 0.75})),
              lambda problem: BsfSuggestion(problem, hyper_hyper_param_from_dict({'beta': 1})),
              ]
strategy_names = [
    # "自适应公式", 
                  "BSF建议0.0625", "BSF建议0.125", "BSF建议0.25", "BSF建议0.5", "BSF建议0.75", "BSF建议1"]

assert len(strategies) == len(strategy_names)
