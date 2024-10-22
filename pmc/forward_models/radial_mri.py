import numpy as np
import math, decimal, torch
from pmc.utils.add_awgn import addawgn
from pmc.forward_models.base import BaseForwardModel
from torch.fft import fftshift, ifftshift, fftn, ifftn

class MRIRadial(BaseForwardModel):
    
    def __init__(self, input_snr, var, shape, num_lines) -> None:
        super().__init__(input_snr, var)
        self.mask = self.generate_mask(shape, num_lines)
        print(f'lines: {num_lines}, acceleration: {1/self.mask.type(torch.float32).mean()}x')

    def __call__(self, x):
        y, _, noise_level = addawgn(self.forward(x), self.input_snr, mask=self.mask)
        print(f'Current noise level (beta=sqrt(variance)) is {noise_level}')
        return self.mask*y

    def forward(self, x):
        return self.mask*self.fftn_(x)

    def adjoint(self, y):
        return self.ifftn_(self.mask*y)

    def grad(self, x, y):
        return super().grad(x, y).real
    
    @staticmethod
    def fftn_(image, dim=None):
        if dim is None:
            dim = tuple(range(1, len(image.shape)))
        return fftshift(fftn(ifftshift(image, dim=dim), dim=dim, norm='ortho'), dim=dim)
    
    @staticmethod
    def ifftn_(kspace, dim=None):
        if dim is None:
            dim = tuple(range(1, len(kspace.shape)))
        return fftshift(ifftn(ifftshift(kspace, dim=dim), dim=dim, norm='ortho'), dim=dim)
    
    @staticmethod
    def generate_mask(shape, num_lines):
        if shape[0] % 2 != 0 or shape[1] % 2 != 0:
            raise RuntimeError('image must be even sized! ')
        # conver to torch
        shape = np.array(shape)
        center = np.array(shape / 2) + 1
        freqMax = np.ceil(
                    np.sum(
                        (shape/2) ** 2
                    ) ** 0.5
                ).astype(int)

        ang = np.linspace(0, math.pi, num=num_lines+1)
        mask = np.zeros(shape, dtype=bool)
        
        for indLine in range(0,num_lines):
            ix = np.zeros(2*freqMax + 1)
            iy = np.zeros(2*freqMax + 1)
            ind = np.zeros(2*freqMax + 1, dtype=bool)
            for i in range(2*freqMax + 1):
                ix[i] = decimal.Decimal(center[1] + (-freqMax+i)*math.cos(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
            for i in range(2*freqMax + 1):
                iy[i] = decimal.Decimal(center[0] + (-freqMax+i)*math.sin(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
                 
            for k in range(2*freqMax + 1):
                if (ix[k] >= 1) & (ix[k] <= shape[1]) & (iy[k] >= 1) & (iy[k] <= shape[0]):
                    ind[k] = True
                else:
                    ind[k] = False
                
            ix = ix[ind]
            iy = iy[ind]
            ix = ix.astype(np.int64)
            iy = iy.astype(np.int64)
            
            for i in range(len(ix)):
                mask[iy[i]-1][ix[i]-1] = True
        
        return torch.tensor(mask).unsqueeze(0).unsqueeze(0)

if __name__ == '__main__':
    pass
    