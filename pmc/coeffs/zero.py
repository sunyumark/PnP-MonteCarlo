from .base import BaseDiffusionCoeff

class ZeroCoeffModule(BaseDiffusionCoeff):
    
    def __init__(self) -> None:
        super().__init__(0)
        
    def brownian_coeff(self, autodm, x, t): 
        return 0

    def score_coeff(self, autodm, x, t): 
        return autodm.sigma
