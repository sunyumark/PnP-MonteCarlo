import torch

class Clip(torch.nn.Module):

    def __init__(self, min, max) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clip(x, self.min, self.max)
        