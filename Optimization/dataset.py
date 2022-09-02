import random

import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import mytransforms
from utils.utils import paths_images_in_folder


class CycleGANDataset(Dataset):
    """Standard CycleGAN dataset."""

    def __init__(self, opt):
        """
        Args:
            A_folder (string): Path to images from domain A
            B_folder (string): Path to images from domain B
            transforms (callable, optional): Optional transforms to be applied on sample.
        """
        self.opt = opt
        self.img_dim = opt["img_dim"]
        self.folder = opt["dataset_folder"]
        self.train = self.opt["mode"] == "train"
        self.direction = self.opt["direction"] if self.train else self.opt["direction_inference"]
        self.a2b = self.direction == 'AtoB'
        self.b2a = self.direction == 'BtoA'
        
        # In Inference mode only one direction is needed
        if self.train or self.a2b:
            self.A_paths = paths_images_in_folder(self.folder, self.opt["mode"], "A")
            self.A_size = len(self.A_paths)
            self.A_transforms = get_transforms(self.opt)
        if self.train or self.b2a:
            self.B_paths = paths_images_in_folder(self.folder, self.opt["mode"], "B")
            self.B_size = len(self.B_paths)
            self.B_transforms = get_transforms(self.opt)


    def __len__(self):
        """Returns the number of images in domain A
        """
        if self.a2b:
            size = self.A_size
        else:
            size = self.B_size

        return size

    def __getitem__(self, index):
        """ Returns an image of each domain A and B.
        For training, every image in domain A is taken, while B is randomly shuffled.
        """
        # In Inference mode only one direction is needed
        if self.a2b:
            A_path = self.A_paths[index % self.A_size]  # assure the index is in range of A
            if self.train:
                B_path = self.B_paths[random.randint(0, self.B_size-1)]  # get random index for B to avoid fixed pairs
        else:
            B_path = self.B_paths[index % self.B_size]  # assure the index is in range of B
            if self.train:
                A_path = self.A_paths[random.randint(0, self.A_size-1)]  # get random index for A to avoid fixed pairs

        if self.train or self.a2b:
            A_img = tiff.imread(A_path)
            samples_A = self.A_transforms({'image': A_img})
        if self.train or self.b2a:
            B_img = tiff.imread(B_path)
            samples_B = self.B_transforms({'image': B_img})

        if self.train:
            return {'A': samples_A['image'], 'B': samples_B['image'], 'A_path': A_path, 'B_path': B_path}
        elif self.a2b:
            return {'A': samples_A['image'], 'A_path': A_path}
        elif self.b2a:
            return {'B': samples_B['image'], 'B_path': B_path}

def get_transforms(opt):
    transforms_list = []
    img_dim = opt["img_dim"]

    if opt["mode"] == "train":
        if opt["preprocess"] == "crop":
            transforms_list.append(mytransforms.RandomCrop(opt["crop_size"], img_dim)) # Random crop
        transforms_list.append(mytransforms.RandomFlip())  # Random horizontal flip
    # transforms_list.append(transforms.Lambda(lambda img: __changetonumpy(img)))
    transforms_list.append(mytransforms.Normalize())
    transforms_list.append(mytransforms.ToTensor())
    return transforms.Compose(transforms_list)


def __changetonumpy(img):
    img = np.asarray(img)
    img = img.astype('float32')
    return img


def __normalize(img):
    """ Normalize uint8"""
    img = 2*img/np.iinfo("uint8").max-1
    return img
