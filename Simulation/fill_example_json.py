import json


if __name__ == "__main__":
    params = dict()

    # path that the resulting images are saved to
    params["output_path"] = "./Data/Simulated_Out/"
    
    # files to extract the cells/prototypes from
    params["input_phantom_folder"] = "./Data/Simulation_Proc_Files/Prototypes/HighRes_AugmentedV2"
    # path to binary images, which define the possible positions for nuclei placement
    params["input_spheroid_mask"] = ["./Data/Simulation_Proc_Files/Masks/spheroid_mask_0.tif"]
    # paths to the point spread functions (in order of usage from front to back)
    params["paths_psf"] = [
        "./Data/Simulation_Proc_Files/PSF/extraxted_psf_9d3zoom_32b_minCrop__scaled-0d5-0d5-0d5.tif"
    ]
    
    # use gpu for convolution
    params["use_gpu_conv"] = 0
    # shape of the image that's generated (before synthesizing the downsampling of the camera!)
    # can be left empty () to use the mask shapes
    params["generated_image_shape"] = [] # (50, 1024, 1024)
    # how many images are to be generated
    params["num_of_images"] = 1
    
    params["max_overlap"] = 0 # max overlap of the cells in the phantom image
    params["min_dist"] = 0 # minimal distance between two nuclei
    params["break_criterion"] = 1000 # max no. of consecutive failed placement attempts
    
    params["brightness_red_fuction"] = "f2p"
    params["brightness_red_factor_b"] = 150

    params["dc_baseline"] = 15  # baseline as a simplified dark current (in range [0, 255])
    params["noise_gauss"] = 0  # sigma for the gauss noise (in range [0, 255])
    params["noise_gauss_absolute"] = 0  # sigma for the absolute gauss noise (in range [0, 255])
    params["noise_poisson_factor"] = 1  # scale of the poisson noise
    params["num_of_acquisition_iterations"] = 1 # no. of images used for averaging

    # pixel size of the original and therefore the generated image (before the camera effects) and desired pixel size
    params["px_size_phantom_img"] = [0.4501347, 0.0610951, 0.0610951] # px size of the nuclei prototypes
    params["px_size_sim_img"] = [0.9002694, 0.1221902, 0.1221902] # px size of the generated image before downsampling
    params["px_size_desired"] = [1.5001311, 0.488759, 0.488759] # desired px size of output image

    # subcultures
    params["input_subculture_folder"] = ""
    params["num_of_subcultures"] = 0
    params["subculture_radius_xy"] = 175
    params["subculture_radius_z"] = 10
    params["subculture_min_dist"] = 0
    params["px_size_subculture_phantoms"] = [0.3250898, 0.04253006666, 0.04253006666]

    f = open("./Simulation/params.json", "w")
    f.write(json.dumps(params, indent=4))
    f.close()
