import torch, pathlib
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from dataclasses import dataclass
from random import Random


class FastMRIBrainData(Dataset):

    def __init__(
        self,
        first_num_slices_only: int = 12,
        zero_mean: bool = True,
        denoise: bool = False,
        split = [90,5,5],
        split_seed: float = 1234,
        finetune: bool = False,
    ):
        super().__init__()
        self.zero_mean = zero_mean
        self.transform = T.Resize((256,256))
        self.slices = None
        self.split = split
        self.split_seed = split_seed
        self.denoise = denoise
        
        # extract test data
        data_list = np.load(pathlib.Path(__file__).parent.resolve() / 'fastmri_brain_info.npy', allow_pickle=True).item()['info']

        Random(self.split_seed).shuffle(data_list)
        num_train = round(len(data_list) * self.split[0] / np.sum(self.split))
        num_val = round(len(data_list) * self.split[1] // np.sum(self.split))  
        
        # test volumes
        test_volumes = data_list[(num_train+num_val):]
        
        self.slices = []
        for volume_info in test_volumes:
            volume_path, num_slices = volume_info
            self.slices += [
                (volume_path, i) 
                for i in range(min(num_slices, first_num_slices_only))
            ]
        # shuffle the slices
        Random(self.split_seed).shuffle(self.slices)
        if finetune:
            # use last ten slices for finetuning
            self.slices = self.slices[-10:]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i: int):
        # load image
        volume_path, slice_idx = self.slices[i]
        image = np.load(volume_path / f'slice_{slice_idx}.npy', allow_pickle=True)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # resize to [256,256]
        image = self.transform(image)
        # remove the background noise
        if self.denoise:
            imax = image.max()
            image[image<0.07*imax] = 0
        # normalize to [0,1]
        image /= image.max()
        # zero mean
        image = 2*image - 1 if self.zero_mean else image
        return image