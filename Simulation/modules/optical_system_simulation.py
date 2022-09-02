import os.path
import time
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
import scipy.ndimage
import tifffile as tiff
from skimage.transform import rescale
from torch.nn.functional import conv3d


class OpticalSystemSimulation:
    path_output: str
    paths_psf: List[str]
    path_spheroid_mask: str
    z_append: int
    brightness_red_fuction: str
    brightness_red_factor: int
    gpu_conv: int


    def __init__(self) -> None:
        pass

    def import_settings(self, params: Dict[str, Any]) -> None:
        """
        Sets the given parameters.

        Args:
            params: A dict containing all parameters for the simulation process

        Returns:
            None
        """
        self.path_output = params["path_output"]
        self.paths_psf = params["paths_psf"]
        self.path_spheroid_mask = params["path_spheroid_mask"]
        self.z_append = params["z_append"]
        self.brightness_red_fuction = params["brightness_red_fuction"]
        self.brightness_red_factor = params["brightness_red_factor"]
        self.gpu_conv = params["gpu_conv"]

    @staticmethod
    def load_images(image_files: List[str]) -> List[np.ndarray]:
        images = list()
        for image_file in image_files:
            images.append(tiff.imread(image_file))

        return images

    def brightness_reduction(self, image_sim: np.ndarray, function: str = "f3", factor: float = 200,
                             save_mask_to: Optional[str] = None) -> np.ndarray:
        """
        Reduces the brightness of pixels inside the cell culture.

        Args:
            image_sim: phantom image
            minimum: the minimum percentage of intensity that no pixel should go below
            save_mask_to: optional path, to save the mask to

        Returns:
            The image with reduced intensity inside the cell culture
        """
        image_sim = image_sim.copy()
        z, y, x = image_sim.shape
        reduction_mask = np.ones(image_sim.shape, dtype=np.float)
        img_max = image_sim.max()

        true_shape = np.copy(image_sim.shape); true_shape[0] -= self.z_append # True shape without appended slices
        spheroid_mask = tiff.imread(self.path_spheroid_mask)
        scale = np.asarray(true_shape) / np.asarray(spheroid_mask.shape)
        spheroid_mask = rescale(spheroid_mask, scale, order=0,
                                channel_axis=None, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        spheroid_mask = np.pad(spheroid_mask, ((0,self.z_append), (0,0), (0,0)), mode='edge')

        if function=="fOld":
            b_funct = lambda i: max(min(((i / z + 1) ** (-6) + 0.2), 1), 0) # fOld
        elif function=="f1":
            b_funct = lambda i: max(min(((i + 5) / z + 1) ** (-6) , 1), 0) # f1
        elif function=="f2":
            b_funct = lambda i: max(min((i / 150 + 1) ** (-6) , 1), 0) # f2
        elif function=="f2p":
            b_funct = lambda i: max(min((i / factor + 1) ** (-6) , 1), 0) # f2
        elif function=="f3":
            b_funct = lambda i: max(min(((i-10) / 200 + 1) ** (-6) , 1), 0) # f3
        elif function=="f3p":
            b_funct = lambda i: max(min(((i-10) / factor + 1) ** (-6) , 1), 0) # f3 with parameter
        elif function=="None":
            b_funct = lambda i: 1 # No Reduction
        
        for y_i in range(0, y):
            for x_i in range(0, x):
                if np.count_nonzero(spheroid_mask[:, y_i, x_i]) <= 1:
                    continue
                z_set = np.nonzero(spheroid_mask[:, y_i, x_i])[0]
                for i in range(0, len(z_set)):
                    if image_sim[z_set[i], y_i, x_i] == 0:
                        continue
                    i_ = i/scale[0]
                    reduction_mask[z_set[i], y_i, x_i] = b_funct(i_)

        if save_mask_to is not None:
            tiff.imwrite(save_mask_to, (reduction_mask * 255).astype(np.uint8))

        image_sim = image_sim.astype(np.float)
        image_sim *= reduction_mask
        img_max_new = image_sim.max()
        image_sim *= img_max / img_max_new
        image_sim = image_sim.astype(np.uint8)
        return image_sim

    @staticmethod
    def min_max_normalization(img: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
        """
        Min-max-normalizes the given image between new_min and new_max.

        Args:
            img: image to be normalized
            new_min: desired minimum
            new_max: desired maximum

        Returns:
            The normalized image
        """
        img_min = img.min()
        img_max = img.max()

        img_norm = ((img - img_min) / (img_max - img_min)) * (new_max - new_min) + new_min

        return img_norm.astype(np.uint8)

    @staticmethod
    def convolve_image_multiple_psf(image: np.ndarray, psfs: List[np.ndarray], gpu_conv: bool = True, verbose: bool = False) -> np.ndarray:
        num_of_psf = len(psfs)

        if num_of_psf == 1:
            return OpticalSystemSimulation.convolve_image(image, psfs[0], gpu_conv=gpu_conv, verbose=verbose)

        images = list()
        z, y, x = image.shape
        num_of_psf = len(psfs)
        num_of_slices = (num_of_psf * 2 - 1)
        len_of_slices = int(z / num_of_slices)

        for n, psf in enumerate(psfs):
            if n == 0:
                offset = 0
                width = 2 * len_of_slices
            elif n == num_of_psf - 1:
                offset = (n * 2 - 1) * len_of_slices
                width = 2 * len_of_slices + (z - num_of_slices * len_of_slices)
            else:
                offset = (n * 2 - 1) * len_of_slices
                width = 3 * len_of_slices

            curr_image = np.zeros_like(image)
            curr_image[offset:offset + width] = OpticalSystemSimulation.convolve_image(image[offset: offset + width],
                                                                                       psf, gpu_conv=gpu_conv,
                                                                                       verbose=False)
            images.append(curr_image)

        image_final = images[-1]

        for n in range(num_of_psf * 2 - 1):
            if n % 2 == 0:
                image_final[n * len_of_slices: n * len_of_slices + len_of_slices] = \
                    images[int(n / 2)][n * len_of_slices: n * len_of_slices + len_of_slices]
            else:
                # transition between two point spread functions
                offset = n * len_of_slices
                for i in range(len_of_slices):
                    image_final[offset + i] = (1 - i / len_of_slices) * images[int(n / 2)][offset + i] \
                                              + (i / len_of_slices) * images[int(n / 2) + 1][offset + i]

        return image_final

    @staticmethod
    def convolve_image(img: np.ndarray, psf: np.ndarray,  gpu_conv: bool = True, verbose: bool = False) -> np.ndarray:
        """
        Convolves the given image with the given point spread function.

        Args:
            img: image to be convolved with the given psf
            psf: point spread function to be used for the convolution
            verbose: if set, there is command line output about the progress

        Returns:
            The convolved image
        """
        if verbose:
            print("Conv: Input: Min/Max value in image: {} / {}. dtype: {}".format(np.min(img), np.max(img), img.dtype))

        # Check if cuda is available to run convolution on gpu
        if gpu_conv and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True

            pad = tuple(np.round((np.asarray(psf.shape) - 1)/2).astype(int))

            # Prepare range and axis for torch
            img = img.astype(np.float32) / 255
            img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
            psf = np.expand_dims(np.expand_dims(psf, axis=0), axis=0)

            img_c = torch.from_numpy(img).to(device)
            psf_c = torch.from_numpy(psf).to(device)
            img = conv3d(img_c, psf_c, padding=pad).cpu().numpy() # padding_mode=zeros

            img = img[0, 0, :, :, :]

        else:
            if verbose:
                print("Conv: CPU is used. This may take a while.")

            img = scipy.ndimage.convolve(img.astype(np.float32), np.flip(psf), mode="constant", cval=0)

        img = OpticalSystemSimulation.min_max_normalization(img, 0, 255)

        if verbose:
            print("Conv: Output: Min/Max value in image: {} / {}. dtype: {}"
                  .format(np.min(img), np.max(img), img.dtype))

        return img.astype(np.uint8)

    def run(self, image: np.ndarray, label: np.ndarray, mask_num: int, img_num: int, verbose: bool = False,
            save_interim_images: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the optical system for the given image

        Args:
            image: image to process
            label: label to the given image
            mask_num: number of the current mask used for placement of nuclei (is used for saving the images when save_interim_images is set)
            img_num: number of the current image (is used for saving the images when save_interim_images is set)
            verbose: if set, the command line output is more verbose
            save_interim_images: if set, all interim results are saved

        Returns:
            The processed image and label
        """

        reduction_mask_out = None
        if save_interim_images:
            reduction_mask_out = os.path.join(self.path_output, "{:03d}_{:03d}_brightness_red_mask.tif".format(mask_num, img_num))

        image = self.brightness_reduction(image, save_mask_to=reduction_mask_out, function=self.brightness_red_fuction, factor=self.brightness_red_factor)

        if save_interim_images:
            tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_brightness_red_z.tif".format(mask_num, img_num)), image)
        if verbose:
            print("Saved brightness reduction along zaxis")

        start = time.time()
        image = self.convolve_image_multiple_psf(image, self.load_images(self.paths_psf), gpu_conv=self.gpu_conv, verbose=verbose)
        duration = time.time() - start
        if verbose:
            print("{:.3f}s ({:.3f}min) needed for convolution of 1 image".format(duration, duration / 60))
        if save_interim_images:
            tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_convolved.tif".format(mask_num, img_num)),
                        image)
        if verbose:
            print("Saved convolved image")

        return image, label
