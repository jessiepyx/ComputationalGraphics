from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from PIL import Image
import numpy as np
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import os


def has_file_allowed_extension(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def get_img_paths(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            if has_file_allowed_extension(d):
                images.append(d)
        else:
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname):
                        path = os.path.join(root, fname)
                        images.append(path)

    return images


class GrayImageDataset(Dataset):
    def __init__(self, directory, train=True):
        self.img_paths = get_img_paths(directory)

        if train:
            self.transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        img = np.asarray(img)
        img_lab = rgb2lab(img)
        img_lab = (img_lab + 128) / 255
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
        img_l = rgb2gray(img)
        img_l = torch.from_numpy(img_l).unsqueeze(0).float()

        return img_l, img_ab

    def __len__(self):
        return len(self.img_paths)
