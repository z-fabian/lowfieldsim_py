## Python Code for Low SNR Simulation at Low Field Strengths
This is a direct Python implementation of the [Low Field Simulator](https://github.com/usc-mrel/lowfieldsim) from the [Magnetic Resonance Engineering Laboratory](https://mrel.usc.edu/) at USC.

For example usage take a look at [fastMRI_to_lowfield.py](fastMRI_to_lowfield.py) that converts the fastMRI knee dataset (1.5T/3.0T) to simulated low field data.

The fastMRI dataset can be downloaded [here](https://fastmri.med.nyu.edu/).

# Requirements
Low Field Sim Py:
- python >= 3.7.6
- numpy >= 1.19.2

fastMRI to low field:
- h5py >= 2.8.0
- xmltodict > 0.12.0

# Running the script
To simulate low field noise on fastMRI data, set INPUT_PATH to the folder containing `.h5` files of the dataset, OUTPUT_PATH to an empty folder where the output will be saved and  B_LOW to the desired low field strength. Then run 

`python3 fastMRI_to_lowfield.py --input-path INPUT_PATH
                                --output-path OUTPUT_PATH
                                --B-low B_LOW`
