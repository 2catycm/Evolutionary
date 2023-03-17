import math
from typing import Tuple
import numpy as np
from abstract_operators import *
import torch
import torch.nn as nn
import warnings
import benchmark_funs

class Identity(EvolvingOperator):

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)        

    def forward(self, _: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return _, h, 0

class SelfAdaptive(EvolvingOperator):

    def __init__(self, problem: benchmark_funs.BenchmarkFunction, hh: torch.Tensor = torch.zeros(hyper_hyper_dimension)):
        super().__init__(problem, hh)        

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        shared_mutation = torch.randn(1)
        anew_mutation = torch.randn(1, x.shape[0])
        step_size = get_hyper_param(h, 'step_size')
        torque_prime = get_hyper_param(h, 'torque_prime')
        torque = get_hyper_param(h, 'torque')
        
        step_size = step_size * torch.exp(torque_prime * shared_mutation + torque * anew_mutation)
        
        set_hyper_param(h, 'step_size', step_size.flatten())
        
        return x, h, 0  
    
strategies = [Identity, SelfAdaptive]
strategy_names = ["不发生变化", "经典自适应"]

assert len(strategies) == len(strategy_names)