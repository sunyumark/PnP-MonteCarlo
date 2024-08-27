from .constant import ConstantCoeffModule

class GeometricAnnealingCoeffModule(ConstantCoeffModule):
    
    def __init__(
        self,
        decay_rate, # coefficient for controlling the strength of score eg. 0.999
        min_sigma, # minimum value of tau
    ) -> None:
        super().__init__()
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma

    def score_coeff(self, pmc, x, t):
        return max(pmc.sigma * self.decay_rate ** t, self.min_sigma)

        
