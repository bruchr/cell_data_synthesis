# Synthesis of large scale 3D microscopic images of 3D cell cultures for training and benchmarking
**Roman Bruch, Florian Keller, Moritz Böhland, Mario Vitacolonna, Lukas Klinger, Rüdiger Rudolf and Markus Reischl**

## Installation

Clone this repository with [`git`](https://git-scm.com/downloads) or download it as .zip end extract it.

Install a conda distribution like [Anaconda](https://www.anaconda.com/products/individual).

Create the environment with conda:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate cell_synthesis
```
Once the environment is activated, the submodules can be run as described below.


## Data

The example data to run this code can be found at Zenodo:
<https://doi.org/10.5281/zenodo.7040492>

Extract the folder and move it in the same directory as this repository. The folderstructure should look like this:
```
├── cell_data_synthesis
│   ├── Data
│   │   ...
│   ├── Evaluation
│   │   ...
│   ├── Optimization
│   │   ...
│   ├── Simulation
│   │   ...
│   ...
...
```


## Simulation

A .json file is required for parameterization of the pipeline. One can either copy the file *./Simulation/params_template.json* and dapt it, or generate a new one with ```python ./Simulation/fill_example_json.py```. The settings file used in the paper can be found in *./Data/Simulated/params.json*.

Start the Simulation with:
```
python ./Simulation/spheroid_simulator.py -json ./Simulation/params.json
```

Optionally the parameters `--verbose` and `--save_interim` can be used. If *save_interim* is given, intermediate image results are saved in the specified output directory.


## Optimization

Once the simulation is completed, the generated images can be optimized. For this, a dataset with real and simulated images needs to be created. Create a folder structure as follows:
```
├── name_of_dataset
│   ├── trainA
│   │   ...
│   ├── trainB
│   │   ...
│   ├── inferenceA
│   │   ...
│   ├── inferenceB
│   │   ...
...
```
The structure and naming is required for the files to be detected.


### Training

Insert the real and the simulated images in the *trainA* and *trainB* folders respectively.

Copy the template settings file *./Optimization/settings_template.hjson*, rename it to *settings.hjson* and adjust the parameters. Set an experiment name and insert the path to the dataset at *dataset_folder*. The *mode* should be 'train' and the *direction* and *direction_inference* should be 'BtoA' if the real images are located in *trainA*.

**Note**: Some parameters specified in the settings file can be overwritten with command line options. See 
```python ./Optimization/start.py --help``` for more details.

Start the training of the optimization with:
```
python ./Optimization/start.py --settings ./Optimization/settings.hjson
```

**Note**: the gpu memory requirements for training the 3D Cycle-GAN are quite high. If you run into memory issues, consider reducing the batch size or the image/crop size to (32, 128, 128). If reducing the crop/image size, the inf_patch_size should be reduced accordingly for optimal results.


### Inference

Once the training is completed, the network can be used to optimize the simulated data. Insert simulated images in the folder *inferenceB*.

The trained model and corresponding settings used in the paper are located at *./Data/Optimized/sim_optimization*. If you want to use this model move/copy this folder into *./Optimization/checkpoints/* and adjust the *dataset_folder* in the *settings.hjson*.

Check the directory *./Optimization/checkpoints/* for the desired experiment folder. Note: the experiment name is appended by a timestamp and the epoch after which the modes were saved. In the following statement, replace the FOLDER_NAME with the name of the desired folder. The inference of the model can the be started with:
```
python ./Optimization/start.py --settings ./Optimization/checkpoints/FOLDER_NAME/settings.hjson --mode inference
```

The results will be placed in *./Optimization/checkpoints/FOLDER_NAME/*.


## Evaluation

Two evaluation metrics (edge_degredation.py and wasserstein_dist.py) are available.
The evaluation is performed by setting the parameters in the respective python file. The files can then be run with:
```
python ./Evaluation/edge_degredation.py
python ./Evaluation/wasserstein_dist.py
```

The segementation quality can be evaluated with the detection metric of the cell tracking challenge. For further information see:
<http://celltrackingchallenge.net/evaluation-methodology/>

The file *./Evaluation/covert_to_ctc.py* allows the generation of the required folder structure for the ctc metric. The metric results can be visualized with:
```
python ./Evaluation/segmentation_eval.py
```