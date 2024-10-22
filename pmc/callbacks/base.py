from abc import ABC, abstractmethod

class BaseCallbackModule(ABC):
    
    def __init__(self) -> None:
        super().__init__()

    def on_inference_start(self, module):
        pass
    
    def on_inference_end(self, module):
        pass
    
    def on_batch_start(self, module, batch, batch_idx):
        pass
    
    def on_batch_end(self, module, samples, means, stds, batch, batch_idx):
        pass

    def on_sample_start(self, module, batch, batch_idx):
        pass
    
    def on_sample_end(self, module, sample_outputs, batch, batch_idx):
        pass

    def on_iteration_start(self, module, batch, batch_idx, t):
        pass
    
    def on_iteration_end(self, module, iteration_outputs, batch, batch_idx, t):
        pass