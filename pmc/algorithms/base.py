from abc import ABC, abstractmethod

class BaseAutoDM(ABC):
    
    def __init__(
        self, 
        forward_model, 
        score_fn, 
        coeff,
        transform,
        gamma,
        **kwargs
    ) -> None:
        # shared attributes of all DMs
        self.forward_model = forward_model
        self.score_fn = score_fn
        self.coeff = coeff
        self.transform = transform
        self.gamma = gamma
        
        # specific attributes of each DM
        self.__dict__.update(kwargs)

    @abstractmethod
    def drift(self, x, y):
        pass

    @abstractmethod
    def diffusion(self, x, t):
        pass

    @abstractmethod
    def __call__(self):
        pass