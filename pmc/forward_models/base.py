from abc import ABC, abstractmethod
from pmc.utils.add_awgn import addawgn

class BaseForwardModel(ABC):
    def __init__(self, input_snr, var) -> None:
        self.input_snr = input_snr
        self.var = var

    def __call__(self, x):
        '''
        generate the noisy measurements y
        Args:
            x: input tensor with shape [B, C, H, W]
        '''
        y, _, noise_level = addawgn(self.forward(x), self.input_snr)
        print(f'Current noise level (beta=sqrt(variance)) is {noise_level}')
        return y

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def adjoint(self, y):
        pass

    def grad(self, x, y):
        return self.adjoint(self.forward(x) - y).real / self.var
    
    def eval(self, x, y):
        return (self.forward(x) - y).norm() ** 2 / (2 * self.var)