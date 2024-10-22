import torch, pathlib, os, warnings, ehtplot.color, time
from typing import Optional, Iterable
from .base import BaseCallbackModule
from pmc.utils.normalize_image import normalize_image
from pmc.utils.save_image import save_image


class LocalTimerCallbackModule(BaseCallbackModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.mean = 0
        self.start = 0
        self.end = 0

    def on_iteration_start(self, module, batch, batch_idx, t):
        self.start = time.time()       
        
    def on_iteration_end(self, module, iteration_outputs, batch, batch_idx, t):
        self.end = time.time()
        self.duration = self.end - self.start
        self.mean = (self.mean * t + self.duration)/(t+1)
        print(self.duration)
        module.logger.log_val({
            f'batch{batch_idx}_mean_duration': self.mean,
        })