from abc import ABC, abstractmethod

class BaseDiffusionCoeff(ABC):
    
    def __init__(self, initial_value) -> None:
        self.initial_value = initial_value

    @abstractmethod
    def brownian_coeff(self, pmc, x, t):
        pass

    @abstractmethod
    def score_coeff(self, pmc, x, t):
        pass