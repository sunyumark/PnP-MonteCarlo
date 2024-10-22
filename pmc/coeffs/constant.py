from .base import BaseDiffusionCoeff

class ConstantCoeffModule(BaseDiffusionCoeff):
    
    def __init__(self) -> None:
        super().__init__(0)

    def score_coeff(self, pmc, x, t):
        return pmc.sigma

    def brownian_coeff(self, pmc, x, t):
        return (2 * pmc.gamma) ** 0.5