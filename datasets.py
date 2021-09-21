import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DataSet1(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        # self.fnames = sorted(glob.glob(root + '/**/*'))
        print()

    def __getitem__(self, index):
        f = self.fnames[index]
        name = os.path.basename(os.path.dirname(f))
        label = int(colors_dict[name])-1
        img = self.transform(Image.open(f))
        return img, label

    def __len__(self):
        return len(self.fnames)