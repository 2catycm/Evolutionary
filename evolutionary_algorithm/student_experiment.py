from datetime import datetime
from pathlib import Path
from matplotlib.font_manager import FontProperties
import torch.nn as nn
import torch
import integer_funs
import abstract_algorithms
from scipy.stats import mannwhitneyu
from matplotlib import pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.rcParams['axes.facecolor']='black'
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
font = FontProperties(fname=project_directory/"fonts/SimHei.ttf")


_current_time = lambda : f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

class StudentExperiment(abstract_algorithms.EvolvingAlgorithm):
    """学生实验算法，用于控制变量做实验。"""
    def __init__(self, problem:integer_funs.IntegerBenchmarkFunction, k_round:int=5, experiment_times:int=5,
                 draw_prob:float=1,  draw_path:str="runs/", 
                 xnames=None, varname=None, fitness_name:str="fitness"):
        super().__init__(problem)
        self.problem:integer_funs.IntegerBenchmarkFunction = problem
        self.k_round:int = k_round
        self.experiment_times:int = experiment_times
        self.draw_path:str = draw_path
        self.draw_prob:float = draw_prob
        self.xnames = xnames
        self.varname = varname or {i:f"var{i}" for i in range(problem.dimension)}
        self.fitness_name = fitness_name

    def forward(self, hh = None):
        """开始控制变量做实验。

        Args:
            hh (torch.Tensor): 从什么变量开始做实验。应当是维度为(problem.dimension)的向量。

        Returns:
            _type_: _description_
        """
        if hh==None:
            hh = (self.problem.lb+(self.problem.ub-self.problem.lb)//2)
            
        hh = hh.repeat(self.experiment_times, 1)
        
        # best_hh, best_fitness = hh, torch.Tensor([self.problem(hh) for _ in range(self.experiment_times)])
        best_hh, best_fitness = hh, -torch.inf*torch.ones(self.experiment_times) # 不做评估，节省时间。
        
        best_fitness_avgs = torch.zeros(self.k_round*self.problem.dimension)
        
        for round in range(self.k_round):
            for i in range(self.problem.dimension):
                hh = best_hh # 下一轮从上一轮最优秀的开始。
                
                # 控制i为自变量，其他变量为无关变量，做实验
                length:int= int(self.problem.ub[i]-self.problem.lb[i]+1)
                fitness = torch.zeros(length, self.experiment_times)
                for j in range(length):
                    hh[:, i] = self.problem.lb[i]+j
                    fitness[j, :] = self.problem(hh)
                    if self.experiment_times >=3:
                        # 记录最优值与最优解
                        _, p = mannwhitneyu(fitness[j, :], best_fitness, alternative='greater') #H0: best_fitness >= fitness
                        if p<0.05:
                            best_fitness = fitness[j]
                            best_hh = hh
                    else:
                        if torch.mean(fitness[j, :])>torch.mean(best_fitness):
                            best_fitness = fitness[j]
                            best_hh = hh.clone()
                    
                # print(f"\t控制变量为{[i.item() for i in hh[0, :]]}时，探究变量{i}对因变量的影响。")
                # print(f"\t实验结论：变量{i}最好取{best_hh[0, i].item()}， 此时fitness平均为{torch.mean(fitness[best_hh[0, i], :]).item()}")
                
                if (torch.rand(1)<self.draw_prob):
                    # 画图展示自变量i对因变量的影响
                    x = torch.arange(int(self.problem.lb[i]), int(self.problem.ub[i]+1))
                    fitness = torch.mean(fitness, dim=1)
                    plt.figure()
                    plt.scatter(x, fitness, c='lawngreen', marker='x')
                    plt.plot(x, fitness, c='salmon')
                    # 避免图画的不好，以后可以重新画
                    torch.save((x, fitness), Path(self.draw_path)/f"{self.problem.name}_round{round}_{self.varname[i]}{_current_time()}.pt")
                    plt.xlabel(f"{self.varname[i]}", fontproperties=font)
                    plt.ylabel(f"{self.fitness_name}", fontproperties=font)
                    if self.xnames is not None:
                        plt.xticks(x, self.xnames[i], fontproperties=font) # 中文显示
                    # plt.title(f"When vars={[i.item() for i in hh[0, :]]}\nExplores the influence of {self.varname[i]} on {self.fitness_name}")
                    plt.title(f"控制变量为{[i.item() for i in hh[0, :]]}时，\n探究自变量“{self.varname[i]}”对因变量“{self.fitness_name}”的影响", fontproperties=font)
                    plt.savefig(Path(self.draw_path)/f"{self.problem.name}_round{round}_{self.varname[i]}{_current_time()}.png")     
                    plt.close()
                best_fitness_avgs[round*self.problem.dimension+i] = torch.mean(best_fitness)
                               
            print(f"第{round}轮实验, fitness平均为{torch.mean(best_fitness).item()}")

                
        return best_hh[0], best_fitness_avgs
                    
                    