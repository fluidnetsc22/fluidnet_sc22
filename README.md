# Accelerating Fluid Simulation with Convolutional Neural Networks.
This repository is based on the paper, [Accelerating Eulerian Fluid Simulation With Convolutional Networks](http://cims.nyu.edu/~schlacht/CNNFluids.htm) by Jonathan Tompson, Kristofer Schlachter, Pablo Sprechmann, Ken Perlin on the accelation of fluid simulations by embedding a neural network in an existing solver for pressure prediction, replacing an expensive pressure projection linked to a Poisson equation on the pressure, which is usually solved with iterative methods (CG or Jacobi methods). We implemented our code with PyTorch, effectively replacing all the original Torch/Lua and C++/CUDA implementation of the inviscid, incompressible fluid solver (based on the open-source fluid simulator [Mantaflow](http://mantaflow.com/), aimed at the Computer Graphics community).
Find the original FluidNet repository [here](https://github.com/google/FluidNet).

## Requirements
* Python 3.8
* GCC 7.5.0 (Also tested on 7.3.0 and 8.2.0)
* (Optional) Paraview


## Installation
To install this repo:

1. Clone this repo:
```
git clone https://github.com/fluidnetsc22/fluidnet_sc22.git
```

2. Create and activate a conda environement
```
conda create --name py38 python=3.8
conda activate py38
```


3. Install Pytorch 1.7.1 compiled with cuda 11, and the rest of necessary packages from the requirements file. To install the rest of the packages just go to the main fluidnet path and perform:

```
cd /path/to/git/repo/fluidnetsc22
pip install -r requirements.txt
```

Then, pytorch is manually installed to ensure the correct cuda compilation.

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```


4. Install cpp extensions for fluid solver:
C++ scripts have been written using PyTorch's backend C++ library ATen.
These scripts are used for the advectionn part and the non CNN pressure solvers.
Note that the code uses helper functions from cuda samples, which are not always easily found when compiling,
thus they are manually especified!(change to the correct path located in path of the cuda sources)
From the main directory follow:

```
cd pytorch/lib/fluid/cpp
CPATH=/usr/local/cuda-11.0/include:/usr/local/cuda-11.0/samples/common/inc python3 setup.py install
```

Do not forget to load the correct gcc compiler and to connect to the gpus to find the cuda sources. Also note that the samples need to be from cuda 11 (instead it will fail to compile). Also note that in some cases the cuda installation might not include the samples folder. To avoid compiling issues, the cuda samples can be downloaded from [this repository](https://github.com/NVIDIA/cuda-samples) (git clone the repoitory), and change ```/usr/local/cuda-11.0/samples/common/inc```by ```/path/to/downloaded/samples/cuda-samples/Common```

5. As an optional feature, ```s5cmd```should be installed if the dataset needs to be loaded from a remote storing system, such as ```S3 buckets```. This is achived by downloading the binary files from the [source.](https://github.com/peak/s5cmd/releases), which should be a file similar to ```s5cmd_1.4.0_Linux-64bit.tar.gz```. This file should be copied at the bin directory located in the home path. This tar file should be decompressed, and then a symbolic link should be created:

```
ln -s ~/bin/s5cmd s5cmd
```

With all this in mind, the cpp extension should be correctly installed. Note that even if some warning pop up this does not block the code compilation (these warning will be removed in posterior versions of the code). Thus, your virtual environment should be ready to run the training and inference. cases!




## Training

**Dataset**
We use the same **3D dataset** as the original FluidNet [Section 1: Generating the data - Generating training data](https://github.com/google/FluidNet#1-generating-the-data) (generated with MantaFlow) for training our CNN.

The entire dataset weighs arounf 1.5 Tb after preprocessing, so depending on the storage capacities, 3 options are included:

1. If enough memory is available in your local file system, the dataset can be directly loaded from the local file system, which enables to obtain the best performances during training. This option is by default set to false in the training config yaml files,
```remote: False```

2. Even if enough storing space is temporally available, in some cases this storing capacity is only momentaneously available, e.g. in on-demand cloud instances. For such cases, every time the machines are stopped, the data stored in the local NVME filesystems is lost. For such configurations, instead of generating the dataset from scratch, the dataset can be stored in a remote memory storage, or ``S3 bucket``. Thus, this code enables to read the dataset from a remote storing object, and it copies the entire dataset locally in the first epoch using [s5cmd](https://github.com/peak/s5cmd), which yields copying velocities of around ~2 Gb/s. This operation despite being long, only needs to be done once for everytime the remote machine is restarted. For optimum download velocity the dataset was compressed to around 1 GB tar.gz files. This option is activated in the config file by specifying: ```remote: True``` and ```local_remote: True```.

3. In cases where the NVME is not large enough, the dataset can be directly read from the remote ``S3 bucket``, by setting the option ```remote: True``` and ```local_remote: False```. Similarly to the previous case, for optimum performance the dataset should be stored in around 1 GB compressed files. For these cases, considerably increasing the number of workers has a negative impact on the training performance, as the prefetching step gets considerably longer, leading to slower trainings as well as possible RAM issues.


The paths and files for the remote reading are still hardcoded in the ```fluid_net_train.py``` file (lines 119-150), further automatization should be developed in future versions of the code.


### **Running the training**
To train the model, go to pytorch folder:

```
cd pytorch
```
The dataset file structure should be located in ```<dataDir>``` folder with the following structure:
```
.
└── dataDir
    └── dataset
        ├── te
        └── tr

```
Precise the location of the dataset in ```pytorch/config.yaml``` writing the folder location at ```dataDir``` (__use absolute paths__, line 14).
Precise also ```dataset``` (name of the dataset, line 16), and output folder ```modelDir``` (line 34) where the trained model and loss logs will be stored and the model name ```modelFilename```.

Run the training :
```
python3 fluid_net_train.py --trainingConfig config_files/training/DESIRED_CONFIG_FILE.yml
```
For a given dataset, a **pre-processing** operation must be performed to save it as PyTorch objects, easily loaded when training. This is done automatically if no preprocessing log is detected.
This process can take some time but it is necessary only once per dataset.

Training can be stopped using Ctrl+C and then resumed by running:
```
python3 fluid_net_train.py --resume
```

You can also monitor the loss during training by running in ```/postproc```

```
python3 plot_loss.py <modelDir> #For total training and validation losses
```

The training configuration models are found in the ```pytorch/config_files/training``` folder. Particularly, the Unet networks is defined with the following parameters (lines 66-78):

```
model_name: 'FlexiUNet' # Choose between FlexiNet, FlexiUnet and FourierNet
scales:
    depth_0: [[2, 26, 28, 26, 24], [48, 26, 28, 26, 1]]
    depth_1: [[24, 28, 28, 24], [48, 28, 28, 24]]
    depth_2: [24, 51, 51, 24]
kernel_sizes: 3
input_res: [64,64,64]
pad_method: 'replicate'              # String for padding method
```

The scales dictionary contains the number of branches in the network, and controls the network height and width as well. The different config files show examples of the network difinitions, so do not hesitate to create your own Unet configuration. All the kernels will have the same size (here set to 3) and the ```ìnput_res``` matches the input resolution for the training dataset (in this case 64 x 64 x 64). Note that when lauching the training a summary of the network parameters is shown:

```
------------------- Model ----------------------------
Input res  [64, 64, 64]
Height of the network: 2
Global properties of each scale:
           |    RF    |  depth   | nparams  |  k-size  |
Branch 0   |        16|         8|    131484|         3|
Branch 1   |        24|         6|    133240|         3|
Branch 2   |        24|         3|    136449|         3|
Total      |        65|        17|    401173|
```
This table shows the number of parameters, depth and receptive field per branch and in total.


### Training options
You can set the following options for training from the terminal command line:
* ```-h``` : displays help message
* ```--trainingConf``` : YAML config file for training. Default = config.yaml.
* ```--modelDir``` : Output folder location for trained model. When resuming, reads from this location.
* ```--modelFilename``` : Model name.
* ```--dataDir``` : Dataset location.
* ```--resume``` : Resumes training from checkpoint in ```modelDir```
* ```--bsz``` : Batch size for training.
* ```--maxEpochs``` : Maximum number training epochs.
* ```--noShuffle``` : Remove dataset shuffle when training.
* ```--lr``` : Learning rate.
* ```--numWorkers``` : Number of parallel workers for dataset loading.
* ```--outMode``` : Training debug options. Prints or shows validation dataset.
        ```save```  = saves plots to disk
        ```show```  = shows plots in window during training
        ```none```  = do nothing

The rest of the training parameters are set in the config files ```pytorch/config_files/training/*.yaml```.

Parameters in the YAML config file are copied into a python dictionary and saved as two separated dictionaries in ```modelDir```, one conf dictionary for parameters related to training (batch size, maximum number of epochs) and one mconf dictionary for parameters related to the model (inputs, losses, scaling options etc)



### Parallelization

The code enables to parallelize the training process to reduce the training time. This is achieved by using two built-in classes implementing data parallelism proposed by the \textit{Pytorch} framework: (i) DataParallel [(DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html), and (ii) DistributedDataParallel [(DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (though PyTorch plans to deprecate DP in favor of DDP, as shown in \url{https://github.com/pytorch/pytorch/issues/65936}). To use more than one GPU during training the ```paralel```option in the yaml file (line 60) needs to be set to true. Due to the simpler implementation as the trainings have been performed in a single node, by default the developed code uses DP. However, DDP has also been tried, and the difference between both trainings are only around 5\% in training time. To launch a DDP training launch the ```fluid_net_train_ddp.py``` file instead of the classical ```fluid_net_train.py``:

```
python3 fluid_net_train_ddp.py --trainingConfig config_files/training/DESIRED_CONFIG_FILE.yml
```

It should be highlighted that the bsz should be consequently modified in the config files, as while for DP the batch size is divided into smaller mibatches, DDP spawns separate processes that read the data with the batch size especified in the config file (line 54).


## Testing

Different architectuires can be tested with the available trained model (or by training a new model):

* Von Karman Vortex Street
* Plume

These test can be run indepently without the need of redoing new trainings, as the trained models used for this study are included in the ```/path/to/fluidnet/trained_models/RF/``` path. To run each case, folow the following commands:

```
cd pytorch
python3 WANTED_SCRIPT.py --simConf config_files/inference/CASE_3D/Config_FILE.yaml
```

Note that the saved configurations correspond to the tested VK configurations with the 256x384x128 domain sizes and Reynolds 300, 1000 and 3000, while the plumes correspond to 128x128x128 domains and Richardsons 0.1, 1 and 10. These config files can be modified to launch further Plume and VK configurations. Note that while the VK test case can be launched with the CG solver, this is not yet possible for the plume test case.


### Test options
* ```-h``` : displays help message
* ```--simConf``` : YAML config file for simulation. Default = plumeConfig.yaml.
* ```--modelDir``` : Trained model location.
* ```--modelFilename``` : Model name.
* ```--outputFolder``` : Location of output results.
* ```--restartSim``` : Restart simulation from checkpoint in ```<outputFolder>```.


The main parameters that the user should modify for the inference test cases are:

- ```outputFolder```, folder in which the output will be saved (line 22)
- ```simMethod```, method to solve the pressure projection step (line 24). Choose between 'convnet' for the CNN, 'jacobi' or 'CG'.
- ```modelDir```, folder in which the trained model is saved (line 26). Note that the networks used in this paper are found in ```/path/to/fluidnet/trained_models/RF/```

The Richardson and Reynolds numbers are respectively changed by modifying the ```buoyancyScale``` parameter (line 51)  and the ```viscosity``` parameter (line 60). The plume test case enables to perform inferences of multiple networks sequentially. To do so, the networks to be studied need to be located in the path indicated in ```modelDir``` (line 28), and included in a list (line 28):
```study_nets: [UNet_4_RF_128_p_100, UNet_5_RF_128_p_100, Jacobi]```
