import os.path
from random import randint, random
from typing import List, Tuple, Dict, Any

import numpy as np
import skimage.measure
import tifffile as tiff
from scipy.ndimage import rotate
from skimage.transform import rescale
from scipy.ndimage import binary_dilation, generate_binary_structure

from utils import load_data, get_sphere


class PhantomImageGenerator:
    path_phantom_folder: str
    path_spheroid_mask: str
    shape: Tuple[int, int, int]
    z_append: int
    px_size_phantoms: Tuple[float, float, float]
    px_size_sim_img: Tuple[float, float, float]
    max_overlap: float
    min_dist: int
    break_criterion: int

    num_of_subcultures: int
    subculture_radius_xy: int
    subculture_radius_z: int
    input_subculture_folder: str
    px_size_subculture_phantoms: Tuple[float, float, float]
    subculture_min_dist: int

    images_folder: str
    labels_folder: str

    phantom_num: int

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
        self.path_phantom_folder = params["path_phantom_folder"]
        self.path_spheroid_mask = params["path_spheroid_mask"]
        self.shape = params["shape"]
        self.z_append = params["z_append"]
        self.px_size_phantoms = params["px_size_phantoms"]
        self.px_size_sim_img = params["px_size_sim_img"]
        self.max_overlap = params["max_overlap"]
        self.min_dist = params["min_dist"]
        self.break_criterion = params["break_criterion"]

        self.num_of_subcultures = params["num_of_subcultures"]
        self.subculture_radius_xy = params["subculture_radius_xy"]
        self.subculture_radius_z = params["subculture_radius_z"]
        self.input_subculture_folder = params["input_subculture_folder"]
        self.px_size_subculture_phantoms = params["px_size_subculture_phantoms"]
        self.subculture_min_dist = params["subculture_min_dist"]

        self.images_folder = params["images_folder"]
        self.labels_folder = params["labels_folder"]

    @staticmethod
    def get_phantoms_with_labels(folder: str, px_size_phantoms: Tuple[float, float, float],
                                 px_size_sim_img: Tuple[float, float, float]) -> Tuple[list, list]:
        """
        Opens all phantoms in the given folder and resizes them to the given pixel size of the simulated image.

        Args:
            folder: Folder, that contains all phantoms and their labels
            px_size_phantoms: pixel/voxel size of the phantoms
            px_size_sim_img: pixel/voxel size of the simulated image

        Returns:
            Tuple of lists with the first list being the phantoms and the second list being the labels
        """
        images = list()
        labels = list()

        scale = np.asarray(px_size_phantoms) / np.asarray(px_size_sim_img)

        image_files = [os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.isfile(os.path.join(folder, f))
                       and f.split(".")[-1] == "tif"
                       and f.count("_label") == 0]

        for image_file in image_files:
            img = load_data(image_file)
            label = load_data(image_file[0:-4] + "_label.tif")

            img = rescale(img, (scale[0], scale[1], scale[2]), order=1, channel_axis=None, preserve_range=True,
                          anti_aliasing=False)
            label = rescale(label, (scale[0], scale[1], scale[2]), order=1, channel_axis=None, preserve_range=True,
                            anti_aliasing=False)
            label[label > 0] = 1
            
            images.append(img)
            labels.append(label)

        return images, labels

    @staticmethod
    def rotate_phantom(img: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotates an image (phantom) and the corresponding label by a random angle between -45 and 45 degree.

        Args:
            img: image of a phantom to be rotated
            label: corresponding label

        Returns:
            The rotated image and label
        """
        angle = randint(-45, 45)

        img = rotate(img, angle=angle, axes=(1, 2), reshape=True).astype(np.uint8)

        label[label != 0] = 255
        label = rotate(label, angle=angle, axes=(1, 2), reshape=True, prefilter=False).astype(np.uint8)
        label[label != 0] = 1

        return img, label

    @staticmethod
    def get_dilated_phantom(phantom_label: np.ndarray, structure: np.ndarray, min_dist: int) -> np.ndarray:
        """
        Dilates a phantom label by the given structures min_dist times

        Args:
            phantom_label: phantom label to be dilated
            structure: structure to be used for the dilation
            min_dist: number of iterations

        Returns:
            The dilated label
        """
        label = np.pad(phantom_label, pad_width=min_dist + 1)
        label = binary_dilation(label, structure=structure, iterations=min_dist)
        return label

    @staticmethod
    def get_region_props(label: np.ndarray) -> dict:
        """
        Returns all necessary region properties of a label necessary for placing a phantom in the phantom image

        Args:
            label: label to get the properties for

        Returns:
            Dictionary of the necessary region properties
        """
        props = dict()

        label = skimage.measure.label(label)
        # 0 because there should be just one cell on each image
        sk_props = skimage.measure.regionprops(label)[0]

        props["area"] = sk_props.area
        props["label_coords"] = sk_props.coords
        label_centroid = np.round(np.asarray(sk_props.centroid)).astype(np.uint16)
        props["coords_centered"] = props["label_coords"] - label_centroid

        label_dim_max_z = max(np.max(props["coords_centered"][:, 0]), -np.min(props["coords_centered"][:, 0]))
        label_dim_max_y = max(np.max(props["coords_centered"][:, 1]), -np.min(props["coords_centered"][:, 1]))
        label_dim_max_x = max(np.max(props["coords_centered"][:, 2]), -np.min(props["coords_centered"][:, 2]))

        props["label_dim_max"] = np.asarray([label_dim_max_z, label_dim_max_y, label_dim_max_x])

        return props

    def place_subculture_phantoms_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray) -> Tuple[np.ndarray,
                                                                                                        np.ndarray]:
        """
        Places a subculture in the given image.

        Args:
            image_sim: phantom image to place the subculture in
            label_sim: label to the phantom image

        Returns:
            phantom image and it's label with the added subculture
        """
        phantom_images, phantom_labels = self.get_phantoms_with_labels(self.input_subculture_folder,
                                                                       self.px_size_subculture_phantoms,
                                                                       self.px_size_sim_img)
        z, y, x = self.shape
        mask = np.zeros_like(label_sim)

        true_shape = np.copy(self.shape); true_shape[0] -= self.z_append # True shape without appended slices
        spheroid_mask = load_data(self.path_spheroid_mask)
        if not np.array_equal(np.asarray(true_shape), np.asarray(spheroid_mask.shape)):
            spheroid_mask = rescale(spheroid_mask, (np.asarray(true_shape) / np.asarray(spheroid_mask.shape)), order=0,
                                    channel_axis=None, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        spheroid_mask = np.pad(spheroid_mask, ((0,self.z_append), (0,0), (0,0)), mode='edge')

        r_xy = self.subculture_radius_xy
        r_z = self.subculture_radius_z
        subculture_mask = get_sphere(r_z, r_xy, r_xy, z_step=int(r_xy / r_z))
        z_mask, y_mask, x_mask = subculture_mask.shape

        count_fail = 0

        while True:
            z_corner = randint(0, z - z_mask - 10)  # Don't cut the subculture
            y_corner = randint(0, y - y_mask)
            x_corner = randint(0, x - x_mask)

            overlap_mask = np.count_nonzero(np.bitwise_and(spheroid_mask[z_corner:z_corner + z_mask,
                                                           y_corner:y_corner + y_mask,
                                                           x_corner:x_corner + x_mask],
                                                           subculture_mask)) / np.count_nonzero(subculture_mask)
            overlap_image = np.count_nonzero(np.bitwise_and(label_sim[z_corner:z_corner + z_mask,
                                                            y_corner:y_corner + y_mask,
                                                            x_corner:x_corner + x_mask],
                                                            subculture_mask)) / np.count_nonzero(subculture_mask)

            if overlap_mask > 0.9 and overlap_image <= self.max_overlap:
                mask[z_corner:z_corner + z_mask,
                     y_corner:y_corner + y_mask,
                     x_corner:x_corner + x_mask] = subculture_mask
                return self.place_objects_in_image(image_sim, label_sim, phantom_images, phantom_labels, mask,
                                                   self.max_overlap, self.subculture_min_dist, self.break_criterion)

            count_fail += 1
            if count_fail > 100:
                print("Warning: could not find a place for the subculture")
                break

        return image_sim, label_sim

    def place_phantoms_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Places phantoms in the given phantom image

        Args:
            image_sim: phantom image to place the phantoms in
            label_sim: label to the phantom image

        Returns:
            phantom image and it's label with the added phantoms
        """
        phantom_images, phantom_labels = self.get_phantoms_with_labels(self.path_phantom_folder, self.px_size_phantoms,
                                                                       self.px_size_sim_img)

        true_shape = np.copy(self.shape); true_shape[0] -= self.z_append # True shape without appended slices
        spheroid_mask = load_data(self.path_spheroid_mask)
        if not np.array_equal(np.asarray(true_shape), np.asarray(spheroid_mask.shape)):
            spheroid_mask = rescale(spheroid_mask, (np.asarray(true_shape) / np.asarray(spheroid_mask.shape)), order=0,
                                    channel_axis=None, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        spheroid_mask = np.pad(spheroid_mask, ((0,self.z_append), (0,0), (0,0)), mode='edge')

        return self.place_objects_in_image(image_sim, label_sim, phantom_images, phantom_labels, spheroid_mask,
                                           self.max_overlap, self.min_dist, self.break_criterion)

    def place_objects_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray, phantom_images: List[np.ndarray],
                               phantom_labels: List[np.ndarray], mask: np.ndarray, max_overlap: float, min_dist: int,
                               break_criterion: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Places objects/phantoms in an image inside a given mask.
        The mask can be a subculture mask or the mask of the whole cell culture.

        Args:
            image_sim: phantom image to place the phantoms in
            label_sim: label to the phantom image
            phantom_images: list of phantoms
            phantom_labels: list of the phantom labels
            mask: mask to place the phantoms in
            max_overlap: maximum overlap between phantoms
            min_dist: minimum overlap between phantoms
            break_criterion: number of insertion fails after that the generation is canceled

        Returns:
            phantom image and it's label with the added phantoms
        """
        count_fail = 0

        dilation_structure = generate_binary_structure(3, 1)

        args = np.argwhere(mask)
        min_z = args[:, 0].min()
        max_z = args[:, 0].max()
        min_y = args[:, 1].min()
        max_y = args[:, 1].max()
        min_x = args[:, 2].min()
        max_x = args[:, 2].max()

        while True:
            # random selection of the phantom to be placed in the image
            nuclei_nr = randint(0, len(phantom_images) - 1)
            phantom_image = phantom_images[nuclei_nr]
            phantom_label = phantom_labels[nuclei_nr]

            # augmentation (rotation)
            if random() > 0.5:
                phantom_image, phantom_label = PhantomImageGenerator.rotate_phantom(phantom_image, phantom_label)

            props = PhantomImageGenerator.get_region_props(phantom_label)
            coords_centered = props["coords_centered"]

            if min_dist > 0:
                # if a minimum distance is given a dilated label is used, to check overlap with other phantoms
                label_dilated = PhantomImageGenerator.get_dilated_phantom(phantom_label, dilation_structure, min_dist)
                props_dilated = PhantomImageGenerator.get_region_props(label_dilated)
                coords_centered_dilated = props_dilated["coords_centered"]
                label_dim_max_dilated = props_dilated["label_dim_max"]
            else:
                coords_centered_dilated = props["coords_centered"]
                label_dim_max_dilated = props["label_dim_max"]

            # augmentation (mirroring)
            if random() > 0.5:
                coords_centered = coords_centered[:, [0, 2, 1]]
                # label_dim_max = label_dim_max[[0, 2, 1]]
                coords_centered_dilated = coords_centered_dilated[:, [0, 2, 1]]
                label_dim_max_dilated = label_dim_max_dilated[[0, 2, 1]]
            if random() > 0.5:
                coords_centered[:, 1] = -coords_centered[:, 1]
                coords_centered_dilated[:, 1] = -coords_centered_dilated[:, 1]
            if random() > 0.5:
                coords_centered[:, 2] = -coords_centered[:, 2]
                coords_centered_dilated[:, 2] = -coords_centered_dilated[:, 2]

            # random position to place the object
            value = [randint(max(label_dim_max_dilated[0], min_z),
                             min(image_sim.shape[0] - label_dim_max_dilated[0] - 1, max_z)),
                     randint(max(label_dim_max_dilated[1], min_y),
                             min(image_sim.shape[1] - label_dim_max_dilated[1] - 1, max_y)),
                     randint(max(label_dim_max_dilated[2], min_x),
                             min(image_sim.shape[2] - label_dim_max_dilated[2] - 1, max_x))]
            coords = np.add(coords_centered, value)
            coords_dilated = np.add(coords_centered_dilated, value)

            # calculate overlap to existing objects
            overlap = np.count_nonzero(label_sim[coords_dilated[:, 0], coords_dilated[:, 1], coords_dilated[:, 2]])

            # calculate overlap with the given mask (100% of phantom should be inside for later brightness reduction)
            overlap_with_mask = np.count_nonzero(mask[coords[:, 0], coords[:, 1], coords[:, 2]])

            if overlap / props["area"] <= max_overlap and overlap_with_mask == props["area"]:
                label_sim[coords[:, 0], coords[:, 1], coords[:, 2]] = self.phantom_num
                image_sim[coords[:, 0],
                          coords[:, 1],
                          coords[:, 2]] = phantom_image[props["label_coords"][:, 0],
                                                        props["label_coords"][:, 1],
                                                        props["label_coords"][:, 2]]

                self.phantom_num += 1
                count_fail = 0
            else:
                count_fail += 1

            if count_fail >= break_criterion:
                break

        return image_sim, label_sim

    def run(self, mask_num: int, img_num: int, verbose: bool = False, save_interim_images: bool = False) -> Tuple[np.ndarray,
                                                                                                   np.ndarray]:
        """
        Generates a phantom image using the given parameters

        Args:
            mask_num: number of the current mask used for placement of nuclei (is used for saving the images when save_interim_images is set)
            img_num: number of the current image (is used for saving the images when save_interim_images is set)
            verbose: if set, there is command line output about the process
            save_interim_images: if set, all interim results are saved

        Returns:
            The generated image and it's label
        """
        self.phantom_num = 1

        # create the new empty images
        image_sim = np.zeros(self.shape, dtype=np.uint16)
        label_sim = np.zeros(self.shape, dtype=np.uint16)

        image_sim = np.clip(image_sim, 0, 255).astype(np.uint8)

        # place subcultures in the simulated image
        for n in range(0, self.num_of_subcultures):
            image_sim, label_sim = self.place_subculture_phantoms_in_image(image_sim, label_sim)

        if self.num_of_subcultures > 0:
            if verbose:
                print("Added all subcultures")
            if save_interim_images:
                tiff.imwrite(os.path.join(self.images_folder, "{:03d}_{:03d}_subcultures.tif".format(mask_num, img_num)), image_sim)
                tiff.imwrite(os.path.join(self.labels_folder, "{:03d}_{:03d}_subcultures_label.tif".format(mask_num, img_num)), label_sim)

        # place the extracted cells in the simulated image
        image_sim, label_sim = self.place_phantoms_in_image(image_sim, label_sim)

        return image_sim, label_sim
