import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

INDICES=[29876]

class CelebA256TestData(Dataset):

    def __init__(
        self,
        grayscale: bool = True,
        zero_mean: bool = True,
    ):
        super().__init__()
        self.grayscale = grayscale
        self.zero_mean = zero_mean
        self.indices = INDICES

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        image_np = plt.imread(f'./test_images/{self.indices[i]:05}.jpg')
        image = torch.tensor(image_np).permute(2,0,1) / 255
        if self.grayscale:
            image = F.rgb_to_grayscale(image)
        # zero mean
        image = 2*image - 1 if self.zero_mean else image
        return image
    