import torch, pathlib, os, warnings
import ehtplot.color
from .base import BaseCallbackModule
from pmc.utils.normalize_image import normalize_image
from pmc.utils.save_image import save_image

class LocalGrayImageMidsaveCallbackModule(BaseCallbackModule):
    
    def __init__(
        self,
        vis_freq=1,
        save_dir='deleteme'
    ) -> None:
        super().__init__()
        self.vis_freq = vis_freq
        self.save_dir = save_dir

    def on_iteration_end(self, module, iteration_outputs, batch, batch_idx, t):
        x, y = batch
        _, x_t_drift, _, _, score, diffusion, df_grad = iteration_outputs
        
        if (t+1) % self.vis_freq == 0:
            # save root
            root_dir = os.path.join(f"{self.save_dir}/{module.cfg.exp_name}", f'batch{batch_idx}')
            pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True)
            # save
            save_image(f"{root_dir}/mean_iter{t}.png", x_t_drift.mean(dim=0).permute(1,2,0))
            save_image(f"{root_dir}/std_iter{t}.png", x_t_drift.std(dim=0).permute(1,2,0))
            save_image(f"{root_dir}/score_iter{t}.png", score.mean(dim=0).permute(1,2,0))
            save_image(f"{root_dir}/dfgrad_iter{t}.png", df_grad.mean(dim=0).permute(1,2,0))
            save_image(f"{root_dir}/diffusion_iter{t}.png", diffusion[0].permute(1,2,0))
            save_image(f"{root_dir}/diff_iter{t}.png", (x_t_drift.mean(dim=0)-x[0]).abs().permute(1,2,0))