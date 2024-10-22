import os, pathlib
import numpy as np
import scipy.io as scio
from typing import Optional, Iterable
from pmc.utils.compute_snr import compute_snr

class LocalLogger:

    def __init__(
        self,
        cfg,
    ) -> None:
        self.cfg = cfg
        self.tmax = cfg.inference.sample_args.tmax

    def init_dict(self):
        self.log_dict = {}
    
    def log_iter_tensor(self, dict, t):
        # load tmax from module
        tmax = self.cfg.inference.sample_args.tmax
        for key, val in dict.items():
            if key not in self.log_dict.keys():
                self.log_dict[key] = val.item()
            else:
                # logger will automatically generate an array for saving scalars 
                # if the same key is called more than once
                if type(self.log_dict[key]) is not np.ndarray:
                    temp = self.log_dict[key]
                    self.log_dict[key] = np.zeros([tmax,1])
                    self.log_dict[key][0] = temp
                self.log_dict[key][t] = val.detach().cpu().numpy()        
    
    def log_iter_persample_tensor(self, dict, sample_idx, t):
        # load tmax from module
        tmax = self.cfg.inference.sample_args.tmax
        n_samples = self.cfg.inference.inference_args.n_samples
        for key, val in dict.items():
            if key not in self.log_dict.keys():
                self.log_dict[key] = val.item()
            else:
                # logger will automatically generate an array for saving scalars 
                # if the same key is called more than once
                if type(self.log_dict[key]) is not np.ndarray:
                    temp = self.log_dict[key]
                    self.log_dict[key] = np.zeros([n_samples,tmax])
                    self.log_dict[key][sample_idx][0] = temp
                self.log_dict[key][sample_idx][t] = val.detach().cpu().numpy()

    def log_batch_tensor(self, dict, img_idx):
        # load tmax from module
        num_imgs = self.cfg.dataloader.batch_size
        for key, val in dict.items():
            if key not in self.log_dict.keys():
                self.log_dict[key] = val.item()
            else:
                # logger will automatically generate an array for saving scalars 
                # if the same key is called more than once
                if type(self.log_dict[key]) is not np.ndarray:
                    temp = self.log_dict[key]
                    self.log_dict[key] = np.zeros([num_imgs,])
                    self.log_dict[key][0] = temp
                self.log_dict[key][img_idx] = val.detach().cpu().numpy()

    def log_tensor(self, dict):
        for key, val in dict.items():
            self.log_dict[key] = val.detach().cpu().numpy()

    def log_val(self, dict):
        for key, val in dict.items():
            self.log_dict[key] = val

    def to_npz(self, file_name):
        np.savez(file_name, self.log_dict)

    def to_mat(self, file_name):
        scio.savemat(file_name, self.log_dict)