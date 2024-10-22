import torch
from pmc.forward_models.base import BaseForwardModel

class CompressedSensing(BaseForwardModel):

    def __init__(self, input_snr, var, shape, compression_ratio, channel=3) -> None:
        super().__init__(input_snr, var)
        assert len(shape) == 2, 'The shape must be 2D!'
        n = shape[0] * shape[1]
        m = int(n * compression_ratio)
        self.A = torch.randn(1, channel, m, n) / m ** 0.5  # you may add a random seed to fix the matrix
        self.AT = self.A.transpose(-2, -1)
        self.C = channel
        self.H = shape[0]
        self.W = shape[1]
        
    def forward(self, x):
        N, C, H, W = x.shape
        return torch.matmul(self.A, x.reshape(N, C, H*W, 1))

    def adjoint(self, y):
        return torch.matmul(self.AT, y).reshape([-1, self.C, self.H, self.W])

    def grad(self, x, y):
        return super().grad(x, y).reshape(x.shape)

if __name__ == '__main__':
    pass