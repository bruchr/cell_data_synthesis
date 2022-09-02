import random

import numpy as np
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image[None,...])}



class Normalize(object):
    """ Normalize uint8"""
    def __call__(self, sample):
        image = sample['image']

        image = image.astype(np.float32)
        image = 2*image/np.iinfo(np.uint8).max - 1
    
        return {'image': image}



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, img_dim):
        self.img_dim = img_dim
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size) if img_dim==2 else (output_size, output_size, output_size)
        else:
            if len(output_size) == 3 and img_dim == 2:
                # 3D image will be cropped to 2D
                assert any(el == 1 for el in output_size), 'If ndim is 2 and length of output size is 3D, one element needs to be one!'
            else:
                assert len(output_size) == img_dim
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        if self.img_dim ==2 and len(self.output_size) == 2:
            h, w = image.shape[-2:]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[...,
                          top: top + new_h,
                          left: left + new_w]
        
        else:
            d, h, w = image.shape[-3:]
            new_d, new_h, new_w = self.output_size

            upper = np.random.randint(0, d - new_d)
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[upper: upper + new_d,
                          top: top + new_h,
                          left: left + new_w]
            image = np.squeeze(image)

        return {'image': image}


class RandomFlip(object):
    """Flip or rotate (90°) image and label image (label-changing transformation)."""

    def __init__(self, p=0.5):
        """
        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """
         :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """
        img = sample['image']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        if random.random() < self.p:

            # img.shape: (Height, Width)
            h = random.randint(0, 2)
            if h == 0:
                # Flip left-right
                img = np.flip(img, axis=1) if img.ndim ==2 else np.flip(img, axis=2)
            elif h == 1:
                # Flip up-down
                img = np.flip(img, axis=0) if img.ndim ==2 else np.flip(img, axis=1)
            elif h == 2:
                # Rotate 90°
                img = np.rot90(img, axes=(0, 1)) if img.ndim ==2 else np.rot90(img, axes=(1, 2))

        return {'image': img}