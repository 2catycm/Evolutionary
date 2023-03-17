import math
from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import warnings
import benchmark_funs
import abstract_algorithms

class MutationBasedEA(abstract_algorithms.EvolvingAlgorithm):
    def __init__(self, problem: benchmark_funs.BenchmarkFunction):
        super().__init__(problem)
    