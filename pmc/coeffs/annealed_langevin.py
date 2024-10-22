import numpy as np

class AnnealedLangevinCoeffModule:
    def __init__(self, sigma_0, sigma_L, schedule_name, steps) -> None:
        self.sigma_0 = sigma_0
        self.sigma_L = sigma_L
        self.schedule_name = schedule_name 
        self.steps = steps

    def get_sigmas(self):
        if self.schedule_name == "geometric_progression":
            return np.geomspace(self.sigma_0, self.sigma_L, self.steps, dtype=np.float64)
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.schedule_name}")