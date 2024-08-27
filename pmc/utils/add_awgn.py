import torch
import numpy as np

snr = lambda xtrue, x: 20*torch.log10(xtrue.norm()/(xtrue-x).norm())

def addawgn(xs, input_snr, mask=None):
    noises = torch.zeros_like(xs)
    if mask is not None:
        mask = torch.tile(mask, [xs.shape[0],1,1,1])
        xs = xs * mask
        noises = noises * mask
    noise_levels = []
    if input_snr is None:
        return xs, noises, torch.zeros(len(xs))
    else:
        for i in range(len(xs)):
            x = xs[i]
            noise_norm = torch.norm(x) * 10 ** (-input_snr / 20)
            noise = torch.randn_like(x)
            noises[i] = noise / torch.norm(noise) * noise_norm
            # noises[i] = noise * 0.05
            xs[i] += noises[i]
            noise_levels.append(noise_norm/torch.norm(noise))
            # noise_levels.append(0.05)
        return xs, noises, torch.tensor(noise_levels)