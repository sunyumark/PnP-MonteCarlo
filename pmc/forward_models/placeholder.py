import numpy as np
from .base import BaseForwardModel

class Placeholder(BaseForwardModel):

    'Placeholder forward model for unconditional image generation'

    def __init__(self) -> None:
        super().__init__(np.inf, 1)

    def forward(self, x):
        return x

    def adjoint(self, y):
        return y