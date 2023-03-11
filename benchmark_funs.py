import torch
import torch.nn as nn
import warnings


class BenchmarkFunction(nn.Module):
    """Some Information about BenchmarkFunction"""

    def __init__(self):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.description: str = self.__class__.__doc__ or "No description"
        # 子类只需要写下面的部分
        self.dimension: int = 0
        self.lb: float = 0  # 简化为一个数
        self.ub: float = 0
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0
        self._device_test = nn.Parameter(torch.rand(1))  # 用于测试device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """抽象的适应度函数，子类必须实现这个函数。
        Args:
            x (torch.Tensor): 维度应该为(population_size, self.dimension)
        Returns:
            torch.Tensor: 返回适应度数值, 维度应该为 (population_size)
        """
        warnings.warn("警告：调用了抽象函数。")
        return torch.zeros(x.shape[0])

    def is_input_valid(self, x: torch.Tensor) -> bool:
        """判断输入是否合法。 子类无需实现，而且每次forward不应当调用这个函数，节省时间。
        Args:
            x (torch.Tensor): 维度应该为(population_size, self.dimension)
        Returns:
            bool: 返回是否合法
        """
        return x.shape[1] == self.dimension and bool(torch.all(x >= self.lb)) and bool(torch.all(x <= self.ub))

    def get_device(self) -> torch.device:
        """获取device。 子类无需实现。
            https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
            forward中生成的局部变量，需要指定device。因此需要这样一个测试函数。
        Returns:
            torch.device: 返回device
        """
        return self._device_test.device

    def get_budget(self, alpha: int = 10000) -> int:
        """获取预算。 子类无需实现。
        """
        return self.dimension*alpha


class Sphere(BenchmarkFunction):
    """Sphere Model"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 30
        self.lb: float = -100  # 简化为一个数
        self.ub: float = 100
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x**2, dim=1)


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock's Function"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 30
        self.lb: float = -30  # 简化为一个数
        self.ub: float = 30
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(100*(x[:, 1:]-x[:, :-1]**2)**2+(x[:, :-1]-1)**2, dim=1)


class Step(BenchmarkFunction):
    """Step Function"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 30
        self.lb: float = -100  # 简化为一个数
        self.ub: float = 100
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.floor(x+0.5)**2, dim=1)


class NoisyQuartic(BenchmarkFunction):
    """Noisy Quartic"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 30
        self.lb: float = -1.28  # 简化为一个数
        self.ub: float = 1.28
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.sum(torch.arange(1, x.shape[1]+1).to(self.get_device())*x**4+torch.rand(1).to(self.get_device()), dim=1)


class Schwefel226(BenchmarkFunction):
    """Schwefel's Problem 2.26"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 30
        self.lb: float = -500  # 简化为一个数
        self.ub: float = 500
        self.optinum: torch.Tensor = 420.9687*torch.ones(self.dimension)
        self.optival: float = -12569.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        return -torch.sum(x*torch.sin(torch.sqrt(abs_x)), dim=1)


class Ackley(BenchmarkFunction):
    """Ackley"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 30
        self.lb: float = -32  # 简化为一个数
        self.ub: float = 32
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -20 * torch.exp(
            -0.2 * torch.sqrt(torch.sum(x**2, dim=1) / x.shape[1])
        ) - torch.exp(torch.sum(torch.cos(2 * torch.pi * x), dim=1) / x.shape[1]) + 20 + torch.e


class Schwefel222(BenchmarkFunction):
    """Schwefel's Problem 2.22"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 30
        self.lb: float = -10  # 简化为一个数
        self.ub: float = 10
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        return torch.sum(abs_x**2, dim=1)+torch.prod(abs_x, dim=1)


class Holder(BenchmarkFunction):
    """Holder Table"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 2
        self.lb: float = -10  # 简化为一个数
        self.ub: float = 10
        self.optinum: torch.Tensor = torch.Tensor([8.05502, 9.66459])
        # 还可以有其他解，就是这个解的正负任取。
        self.optival: float = -19.2085

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        return -torch.abs(torch.sin(x1)*torch.cos(x2)*torch.exp(torch.abs(1-torch.sqrt(x1**2+x2**2)/torch.pi)))


class Camel3(BenchmarkFunction):
    """Three Hump Camel"""

    def __init__(self):
        super().__init__()
        self.dimension: int = 2
        self.lb: float = -5  # 简化为一个数
        self.ub: float = 5
        self.optinum: torch.Tensor = torch.zeros(self.dimension)
        self.optival: float = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2


class Michal(BenchmarkFunction):
    """Michalewicz"""

    def __init__(self, m=10):
        super().__init__()
        self.dimension: int = 2
        self.lb: float = 0  # 简化为一个数
        self.ub: float = torch.pi
        self.optinum: torch.Tensor = torch.Tensor([2.20, 1.57])
        self.optival: float = -1.8013
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sin(x) * torch.sin(torch.arange(1,
                                                  x.shape[1]+1).to(self.get_device())*x**2 / torch.pi)**(2 * self.m)
        return -torch.sum(y, dim=1)


benchmark_functions = [Sphere(), Rosenbrock(), Step(), NoisyQuartic(
), Schwefel226(), Ackley(), Schwefel222(), Holder(), Camel3(), Michal()]
