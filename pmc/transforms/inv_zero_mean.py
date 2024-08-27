import torch

class InverseZeroMean(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return (x+1)/2