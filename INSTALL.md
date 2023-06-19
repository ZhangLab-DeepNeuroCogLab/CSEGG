# Installation Guide

## Pre-requisite

The code has been successfully tested in Ubuntu 20.04 with four GPUs (NVIDIA RTX A5000). It requires the following:
- PyTorch = 1.10.1 
- python = 3.8.13
- CUDA = 11.3
- torchvision = 0.11.2
- torchaudio = 0.10.1
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV

To begin the installation, follow the following instructions.

Download our repository:

```bash
git clone https://github.com/ZhangLab-DeepNeuroCogLab/CSEGG.git
```

## Conda Environments

Refer to [link](https://docs.anaconda.com/free/anaconda/install/index.html) for Anaconda installation. Alternatively, execute the following command:

```bash 
#setting up conda
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
```
After Anaconda installation, create a new conda environment using the following command:
```bash 
conda create --name sggc --file spec_file.txt
```
Activate the conda environment:
```bash
conda activate sggc
```
## MISC packages
Install the miscellaneous (MISC) dependencies using the following commands:
```bash
pip install colorama setuptools==59.5.0 ninja yacs cython matplotlib tqdm opencv-python overrides
conda install ipython
conda install scipy
conda install h5py
```

## Build Project
Build the project on your local system using the following command: 
```bash
python setup.py build develop
```


## Installing Pycoco and Apex dependencies 

Execute the following command to set the installation directory:

```bash
#This command sets the "INSTALL_DIR" environment variable to the current working directory, 
#simplifying the referencing of the installation directory in subsequent steps.
export INSTALL_DIR=$PWD
```

Install pycocotools (python version of cocoapi) using the following set of commands:
```bash
cd $INSTALL_DIR

#downloading cocoapi package
git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI

#install pycocotools
python setup.py build_ext install
```

Install apex using the following set of commands:
```bash
cd $INSTALL_DIR

#downloading apex package
git clone https://github.com/NVIDIA/apex.git

cd apex

#install apex
python setup.py install --cuda_ext --cpp_ext
cd $INSTALL_DIR
```

Execute the following command to unset the installation directory:
```bash
unset INSTALL_DIR
```



