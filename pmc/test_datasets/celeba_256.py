import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

INDICES = [
    29876, 27336, 28341, 27881, 29458, 28860, 29029, 29260, 27604,
    27103, 29267, 29719, 29021, 29924, 27930, 29804, 28768, 28496,
    29810, 27131, 29654, 28274, 28491, 28352, 27249, 29698, 28911,
    27285, 29430, 29074, 29243, 29146, 27125, 28855, 27609, 27046,
    27389, 29092, 29182, 29373, 27541, 29552, 29356, 27703, 29847,
    29573, 28952, 29615, 28868, 29737, 28100, 29461, 28592, 27291,
    27975, 27934, 28456, 28201, 29655, 29871, 28297, 28089, 28913,
    29045, 27712, 27148, 29591, 29997, 27095, 28275, 28747, 27204,
    27307, 29125, 28032, 27424, 29966, 29247, 27072, 28781, 29645,
    27309, 27799, 28419, 28138, 27625, 27637, 29249, 28184, 27271,
    29197, 28716, 29718, 27076, 28146, 29284, 28552, 27365, 29456,
    29826
]

# INDICES = [
#     28860, 29029, 29260, 29267, 29804, 28274, 27249, 29430
# ]

Finetune_INDICES = [29197, 28716, 29718, 27076, 28146, 29284, 28552, 27365, 29456, 29826]

class CelebA256TestData(Dataset):

    def __init__(
        self,
        grayscale: bool = True,
        zero_mean: bool = True,
        finetune: bool = False,
    ):
        super().__init__()
        self.indices = INDICES if not finetune else Finetune_INDICES
        self.grayscale = grayscale
        self.zero_mean = zero_mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        image_np = plt.imread(f'/scratch/imaging/projects/zwu2/imaging-03/zwu2/datasets/celeb_a/celeba_hq_256/{self.indices[i]:05}.jpg')
        image = torch.tensor(image_np).permute(2,0,1) / 255
        if self.grayscale:
            image = F.rgb_to_grayscale(image)
        # zero mean
        image = 2*image - 1 if self.zero_mean else image
        return image
    