import torch

class Normalize(torch.nn.Module):

    def __init__(self, min, max) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        N, C, H, W = x.shape
        # normalize to [0,1]
        x_reshaped = x.reshape([N, -1])
        xmax = x_reshaped.max(dim=-1)[:, None, None, None]
        xmin = x_reshaped.min(dim=-1)[:, None, None, None]
        x = (x - xmin) / (xmax - xmin)
        return (self.max-self.min) * x + self.min