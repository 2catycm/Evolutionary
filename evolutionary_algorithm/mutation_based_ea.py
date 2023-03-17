#%%
import math
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import warnings
import benchmark_funs
import abstract_algorithms
import evolutionary_operator as eo
from evolutionary_operator import abstract_operators as ao
from integer_funs import IntegerBenchmarkFunction
from tqdm import tqdm, trange

#%%

class MutationBasedEA(abstract_algorithms.EvolvingAlgorithm):
    def __init__(self, problem: benchmark_funs.BenchmarkFunction):
        super().__init__(problem)
        self.sss = [eo.initialization_operators.strategies,
         eo.mutation_operators.strategies,
        eo.selection_operators.strategies,
         eo.hyper_init_operators.strategies,
        eo.hyper_update_operators.strategies]
        
    def forward(self, hh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ss = [self.sss[i][hh[i]](self.problem) for i in range(len(self.sss))]
        h = torch.zeros(1, device=hh.device)
        # 执行 超参数初始化 策略
        _,h,_ = ss[3](None, h) 
        population_size = int(ao.get_hyper_param(h, 'population_size').mean())
        x = torch.zeros((population_size, self.problem.dimension), device=hh.device)
        # 执行 搜索变量初始化 策略
        x,h,_ = ss[0](x, h)
        
        budget = self.problem.get_budget()
        # print(f"算法{self}准备完成，开始执行。")
        rounds = budget//(population_size*self.problem.dimension)
        
        best_x, best_fitnesses = None, [-torch.inf]
        # while budget>0:
        for _ in trange(rounds, desc=f"演化算法优化中"):
            # 执行超参数变异策略
            x, h, _ = ss[4](x, h)
            offspring,h,_ = ss[1](x,h)
            x,h,e = ss[2](torch.vstack([x, offspring]), h)
            # budget-=e
            fitness = self.problem(x)
            
            best_fitnesses.append(fitness.max().item())
            if best_fitnesses[-1]>best_fitnesses[-2]:
                best_x = x[fitness.argmax()]
        return best_x, best_fitnesses
        


#%%
import benchmark_funs as bf
class BestMutationBasedEA(IntegerBenchmarkFunction):
    """在十个数值优化问题上，基于变异的进化算法的最优结果
    """
    def __init__(self, dataset:List[bf.BenchmarkFunction], inner_experiment_times=3):
        super().__init__()
        self.inner_experiment_times = inner_experiment_times
        self.xnames = [eo.initialization_operators.strategy_names,
         eo.mutation_operators.strategy_names,
        eo.selection_operators.strategy_names,
         eo.hyper_init_operators.strategy_names,
        eo.hyper_update_operators.strategy_names
        ]
        
        self.dataset:List[bf.BenchmarkFunction] = dataset
        self.dimension: int = len(self.xnames)
        self.lb: torch.Tensor = torch.zeros(self.dimension, dtype=torch.int32)
        self.ub: torch.Tensor = torch.Tensor([len(s) for s in self.xnames]).to(torch.int32)-1
        self.optinum: torch.Tensor = None
        self.optival: float = 0 # 与最优值的差值最好是0
        self.larger_better = True
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): 很多个策略组合。

        Returns:
            torch.Tensor: 这些策略组合优化出来的值。
        """
        result = torch.zeros(x.shape[0], device=x.device)
        for b_fun in tqdm(self.dataset, '使用数据集评估一个演化算法中'):
            mutation_ea = MutationBasedEA(b_fun) 
            for i in range(x.shape[0]):
                best_fitnesses = sum([mutation_ea(x[i])[1][-1] for _ in range(self.inner_experiment_times)])/self.inner_experiment_times
                result[i]+=(best_fitnesses-b_fun.optival)
                if (best_fitnesses-b_fun.optival)>0:
                    warnings.warn(f"演化算法{mutation_ea}在{b_fun}上的最优值{best_fitnesses}大于最优值{b_fun.optival}。")
        return result
        