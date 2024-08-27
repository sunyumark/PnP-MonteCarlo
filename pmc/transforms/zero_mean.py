import torch

class ZeroMean(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        # shift
        x = x - x.min()
        # zero mean
        x = 2*x - x.max()
        return x