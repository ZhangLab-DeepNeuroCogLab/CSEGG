# Install Packages

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

## Conda Environments

Refer to [link](https://docs.anaconda.com/free/anaconda/install/index.html) for Anaconda installation. Alternatively, execute the following command:

```bash 
#setting up conda
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
```
After Anaconda installation, create a new conda environment using the following command:
``` 
conda create --name sggc --file spec_file.txt
```
Activate the conda environment:
```
conda activate sggc
```
## MISC packages
```
pip install colorama setuptools==59.5.0 ninja yacs cython matplotlib tqdm opencv-python overrides
conda install ipython
conda install scipy
conda install h5py
```

## Build Project
```
python setup.py build develop
```


## Installing Pycoco and Apex dependencies 
```
export INSTALL_DIR=$PWD
```

### Install pycocotools
```
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

### Install apex
```
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd $INSTALL_DIR
```

```
unset INSTALL_DIR
```



