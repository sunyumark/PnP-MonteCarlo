import torch
import numpy as np
from tqdm import tqdm

class AnnealedLangevin:
    
    """
    This class is an re-implementation of Yang Song's annealed langevin algorithm 
    for unconditional image generation. When using an identity forward model (A=I),
    the PMC algorithms are reduced to the annealed Langevin algorithm.
    """
    
    def __init__(
        self, 
        forward_model: object,
        score_fn: torch.nn.Module, 
        coeff: object,
        transform: object,
        gamma: float,
        T: int
    ):
        self.forward_model = forward_model
        self.score_fn = score_fn
        self.score_fn.eval()
        self.coeff = coeff
        self.transform = transform
        self.gamma = gamma
        self.T = T
        self.sigmas = coeff.get_sigmas()
        
    def __call__(self, x, y, t, tmax):
        alpha = self.gamma * (self.sigmas[t] / self.sigmas[-1]) ** 2


        if self.score_fn.__class__.__name__ in ['UNetModel']:
            label = torch.tensor([self.sigmas[t]]).float()

        for _ in range(self.T):
            with torch.no_grad():
                score = self.score_fn(x, label)

            # get the variance from the output if learned variance
            if score.shape[1] == 2*x.shape[1]:
                score, _ = torch.split(score, x.shape[1], dim=1)

            x = x + alpha * score + np.sqrt(2*alpha) * torch.randn_like(x)

        return x, x, x, x, x, x, x