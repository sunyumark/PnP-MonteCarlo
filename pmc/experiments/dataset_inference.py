import torch
from pmc.inference import PMCInference 

class DatasetInference:
    
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, pmc, dataloader, callbacks):

        # auto DM model
        inference = PMCInference(
            cfg = self.cfg,
            pmc = pmc,
            callbacks = callbacks
        )

        # perform inference
        inference(dataloader, **dict(self.cfg.inference.inference_args))