import torch, pathlib, os, warnings
from typing import Optional, Iterable
from .base import BaseCallbackModule
from pmc.utils.normalize_image import normalize_image
from pmc.utils.save_image import save_image

class LocalGrayImageCallbackModule(BaseCallbackModule):
    
    def __init__(self) -> None:
        super().__init__()

    def on_batch_end(self, module, samples, means, stds, batch, batch_idx):
        # x ~ [N, C, H, W]
        # y ~ [N, C, h, w]
        # recons ~ [N, C, H, W]
        # stds ~ [N, C, H, W]
        x, y = batch
        xrecon_mean, xdrift_recon_mean = means
        xrecon_std, xdrift_recon_std = stds
        diff = xrecon_mean[0] - x[0]
        
        # check directory
        save_dir = os.path.join(module.cfg.exp_dir, f'batch{batch_idx}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        # assert xrecon_mean.shape == xrecon_std.shape == x.shape
        
        save_image(
            os.path.join(save_dir, f'batch{batch_idx}_mean.png'),
            xrecon_mean[0].permute(1,2,0),
            colormap='gray',
        )
        save_image(
            os.path.join(save_dir, f'batch{batch_idx}_std.png'),
            xrecon_std[0].permute(1,2,0),
            colormap='gray'
        )
        save_image(
            os.path.join(save_dir, f'batch{batch_idx}_xdrift_mean.png'),
            xdrift_recon_mean[0].permute(1,2,0),
            colormap='gray',
        )
        save_image(
            os.path.join(save_dir, f'batch{batch_idx}_xdrift_std.png'),
            xdrift_recon_std[0].permute(1,2,0),
            colormap='gray'
        )
        save_image(
            os.path.join(save_dir, f'batch{batch_idx}_diff.png'),
            diff.abs().permute(1,2,0),
            colormap='gray'
        )

    def on_sample_start(self, module, batch, batch_idx):
        # check directory
        save_dir = os.path.join(module.cfg.exp_dir, f'batch{batch_idx}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        x, y = batch

        save_image(
            os.path.join(save_dir, f'batch{batch_idx}_gt.png'), 
            x[0].permute(1,2,0), 
            colormap='gray'
        )
        save_image(
            os.path.join(save_dir, f'batch{batch_idx}_adjoint_recon.png'), 
            module.pmc.forward_model.adjoint(y)[0].real.permute(1,2,0),
            colormap='gray'
        )
        try:
            save_image(
                os.path.join(save_dir,f'batch{batch_idx}_meas.png'), 
                y[0].permute(1,2,0), 
                colormap='gray'
            )
        except RuntimeError:
            warnings.warn('measurements is not a 2D matrix and thus cannot be saved')