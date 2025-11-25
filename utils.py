import random
import numpy as np
import torch

from torchvision import transforms

def set_seet(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ToTensor3D:
    def __call__(self, sample):
        img, label = sample
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        img = transforms.Normalize(mean=[0.5], std=[0.5])
        label = torch.tensor(label).long()
        return img, label