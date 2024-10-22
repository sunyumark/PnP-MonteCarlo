import torch
from .base import BaseAutoDM
from typing import Optional

class PMCRED(BaseAutoDM):

    def __init__(
        self, 
        forward_model: object,
        score_fn: torch.nn.Module, 
        coeff: object,
        gamma: float, 
        alpha: float,
        sigma: float,
        transform: Optional[torch.nn.Module]= None,
    ) -> None:
        super().__init__(forward_model, score_fn, coeff, transform, gamma)
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, x, y, t, *arg):
        drift, df_grad, score = self.drift(x, y, t)
        xnextdrift = x + drift
        diffusion = self.diffusion(x, t)
        xnext = xnextdrift + diffusion
        return xnext, xnextdrift, x, drift, score, diffusion, df_grad

    def drift(self, x, y, t):
        '''
        The iterate x has the following size 
        [B, C, H, W]
        '''
        # get gradient of the forward model
        df_grad = self.forward_model.grad(x, y)
        # transform
        if self.transform is not None:
            x = self.transform(x)
        # compute the score
        sigma = self.coeff.score_coeff(self, x, t)
        # switch to evaluation mode
        if self.alpha == 0:
            score = torch.zeros_like(x)
        else:
            self.score_fn.eval()
            with torch.no_grad():
                alpha = max(self.alpha * sigma ** 2, 1)
                score = alpha * self.score_fn(
                                x,
                                sigma * torch.ones(x.shape[0])
                            )            
        # combine to get the drift (Note the output of the score_fn is negative score)
        drift = self.gamma*(-df_grad + score)
        return drift, df_grad, score

    def diffusion(self, x, t):
        return self.coeff.brownian_coeff(self, x, t) * torch.randn_like(x)