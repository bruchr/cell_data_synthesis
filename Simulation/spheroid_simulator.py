import json
import os.path
from typing import List, Optional

import numpy as np
import tifffile as tiff

from modules.camera_acquisition_simulation import CameraAcquisitionSimulation
from modules.optical_system_simulation import OpticalSystemSimulation
from modules.phantom_image_generator import PhantomImageGenerator


class SpheroidSimulator:
    phantom_image_generator: PhantomImageGenerator
    optical_system_simulation: OpticalSystemSimulation
    camera_acquisition_simulation: CameraAcquisitionSimulation

    path_spheroid_mask_list: List[str]
    num_of_images: int
    path_output: str
    path_out_folders: List[str]
    generation_params: dict
    optical_system_params: dict
    camera_acquisition_params: dict

    params_set: bool

    def __init__(self) -> None:
        self.phantom_image_generator = PhantomImageGenerator()
        self.optical_system_simulation = OpticalSystemSimulation()
        self.camera_acquisition_simulation = CameraAcquisitionSimulation()

        self.path_out_folders = ["images", "labels"]
        self.params_set = False

    @staticmethod
    def missing_param(params: dict) -> Optional[str]:
        """
        Checks if all necessary params are given and returns None.
        If params are missing, the key/name of the first one is returned.

        Args:
            params: A dict containing all parameters for the simulation process

        Returns:
            None if no key is missing, otherwise the key/name (str) of the first missing parameter
        """
        necessary_keys = [
            "output_path", "input_phantom_folder", "input_spheroid_mask", "paths_psf", "use_gpu_conv", "generated_image_shape", 
            "num_of_images", "max_overlap", "min_dist", "break_criterion", "brightness_red_fuction", "brightness_red_factor_b", 
            "dc_baseline", "noise_gauss", "noise_gauss_absolute", "noise_poisson_factor", "num_of_acquisition_iterations", 
            "px_size_phantom_img", "px_size_sim_img", "px_size_desired", "input_subculture_folder", "num_of_subcultures", 
            "subculture_radius_xy", "subculture_radius_z", "subculture_min_dist", "px_size_subculture_phantoms", 
            ]

        for key in necessary_keys:
            if key not in params:
                return key

        return None

    def import_settings(self, params: dict) -> None:
        """
        Imports the given settings for the simulation process.

        Args:
            params: A dict containing all parameters for the simulation process

        Returns:
            None
        """
        missing = self.missing_param(params)
        if missing is not None:
            raise KeyError("Key '" + missing + "' is missing in the given parameters")

        
        z_append = round(30 / (params["px_size_sim_img"][0]/params["px_size_phantom_img"][0]))
        if len(params["generated_image_shape"]) != 0:
            z, y, x = params["generated_image_shape"]
            shape = (z + z_append, y, x)
        else:
            shape = None


        self.path_spheroid_mask_list = params["input_spheroid_mask"]
        self.num_of_images = params["num_of_images"]
        self.path_output = params["output_path"]

        self.generation_params = {
            "path_phantom_folder": params["input_phantom_folder"],
            "path_spheroid_mask": None,
            "shape": shape,
            "z_append": z_append,
            "max_overlap": params["max_overlap"],
            "min_dist": params["min_dist"],
            "break_criterion": params["break_criterion"],
            "px_size_phantoms": params["px_size_phantom_img"],
            "px_size_sim_img": params["px_size_sim_img"],
            "input_subculture_folder": params["input_subculture_folder"],
            "num_of_subcultures": params["num_of_subcultures"],
            "subculture_radius_xy": params["subculture_radius_xy"],
            "subculture_radius_z": params["subculture_radius_z"],
            "subculture_min_dist": params["subculture_min_dist"],
            "px_size_subculture_phantoms": params["px_size_subculture_phantoms"],
            "images_folder": os.path.join(self.path_output, self.path_out_folders[0]),
            "labels_folder": os.path.join(self.path_output, self.path_out_folders[1])
        }

        self.optical_system_params = {
            "paths_psf": params["paths_psf"],
            "path_spheroid_mask": None,
            "gpu_conv": params["use_gpu_conv"],
            "z_append": z_append,
            "brightness_red_fuction": params["brightness_red_fuction"],
            "brightness_red_factor": params["brightness_red_factor_b"],
            "path_output": os.path.join(self.path_output, self.path_out_folders[0])
        }

        self.camera_acquisition_params = {
            "baseline": params["dc_baseline"],
            "noise_gauss": params["noise_gauss"],
            "noise_gauss_absolute": params["noise_gauss_absolute"],
            "noise_poisson_factor": params["noise_poisson_factor"],
            "num_of_acquisition_iterations": params["num_of_acquisition_iterations"],
            "px_size_img": params["px_size_sim_img"],
            "px_size_desired": params["px_size_desired"],
            "path_output": os.path.join(self.path_output, self.path_out_folders[0])
        }

        self.params_set = True

    def import_settings_json(self, file_path: str) -> None:
        """
        Tries to open and import parameters from a given text file in java object notation

        Args:
            file_path: path to the text file

        Returns:
            None
        """
        f = open(file_path, "r")
        params = json.loads(f.read())
        f.close()

        self.import_settings(params)

    def check_n_make_out_folders(self) -> None:
        """
        Checks if all necessary output folders are available. If not, they are created.

        Returns:

        """
        for folder in self.path_out_folders:
            path_out = os.path.join(self.path_output, folder)
            if not os.path.isdir(path_out):
                os.makedirs(path_out)

    def run(self, verbose: bool = False, save_interim_images: bool = False) -> None:
        """
        Runs the simulation process with the given parameters.

        Args:
            verbose: if verbose is set, the command line output is more verbose
            save_interim_images: if set, all interim results (e.g. the phantom image) are saved

        Returns:
            None
        """
        if not self.params_set:
            raise AttributeError("No parameters set. Call import_settings or import_settings_json first!")

        self.check_n_make_out_folders()

        shape_from_mask = True if self.generation_params['shape'] is None else False

        for n_mask, path_spher_mask in enumerate(self.path_spheroid_mask_list):
            self.generation_params['path_spheroid_mask'] = path_spher_mask
            self.optical_system_params['path_spheroid_mask'] = path_spher_mask

            if shape_from_mask:
                shape = np.asarray(tiff.imread(path_spher_mask).shape)
                if not np.array_equal(self.camera_acquisition_params['px_size_img'], self.camera_acquisition_params['px_size_desired']):
                    shape = np.divide(shape, np.divide(self.camera_acquisition_params['px_size_img'], self.camera_acquisition_params['px_size_desired']))
                shape[0] += self.generation_params['z_append']
                self.generation_params['shape'] = np.round(shape).astype(int)

            self.phantom_image_generator.import_settings(self.generation_params)
            self.optical_system_simulation.import_settings(self.optical_system_params)
            self.camera_acquisition_simulation.import_settings(self.camera_acquisition_params)

            for n in range(1, self.num_of_images + 1):
                print("Image {}_{}".format(n_mask, n))
                if verbose:
                    print("Generating prototype ...")
                image, label = self.phantom_image_generator.run(mask_num=n_mask, img_num=n, verbose=verbose, save_interim_images=save_interim_images)

                if save_interim_images:
                    tiff.imwrite(os.path.join(self.path_output, self.path_out_folders[0],
                                            "{:03d}_{:03d}_prototype.tif".format(n_mask, n)), image)
                    if verbose:
                        print("Saved prototype")

                if verbose:
                    print("Simulating optical system ...")
                image, label = self.optical_system_simulation.run(image, label, mask_num=n_mask, img_num=n, verbose=verbose,
                                                                save_interim_images=save_interim_images)

                if verbose:
                    print("Simulating optical system done")

                image = image[:-self.generation_params['z_append'], :, :]
                label = label[:-self.generation_params['z_append'], :, :]

                if verbose:
                    print("Simulating camera ...")
                image, label = self.camera_acquisition_simulation.run(image, label, mask_num=n_mask, img_num=n, verbose=verbose,
                                                                    save_interim_images=save_interim_images)

                tiff.imwrite(os.path.join(self.path_output, self.path_out_folders[0], "{:03d}_{:03d}_final.tif".format(n_mask, n)), image)
                tiff.imwrite(os.path.join(self.path_output, self.path_out_folders[1], "{:03d}_{:03d}_final_label.tif".format(n_mask, n)),
                            label)

                if verbose:
                    print("Saved final image")

                print("--------------------------------------------------------------")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate synthetic microscopy images of cell spheroids")
    parser.add_argument("-json", type=str, default="./params.json",
                        help="The json-file, that contains all relevant parameters")
    parser.add_argument("--verbose", dest='verbose', action='store_true',
                        help="If verbose is set, the command line output is more detailed")
    parser.add_argument("--save_interim", dest='save_interim', action='store_true',
                        help="If save_interim is set, interim results are saved (e.g. phantom image)")
    parser.set_defaults(verbose=False)
    parser.set_defaults(save_interim=False)
    args = parser.parse_args()

    verbose_arg = vars(args)["verbose"]
    save_interim_arg = vars(args)["save_interim"]
    json_file = vars(args)["json"]
    if not os.path.isfile(json_file):
        if json_file != parser.get_default("json"):
            error_message = "The given json '" + json_file + "' is no valid file"
        else:
            error_message = "Please specify an existing file for the parameters using the command line argument '-json'"
        raise AttributeError(error_message)

    simulator = SpheroidSimulator()
    simulator.import_settings_json(json_file)
    simulator.run(verbose=verbose_arg, save_interim_images=save_interim_arg)
