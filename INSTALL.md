# Install Packages

## Conda Environments

``` bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name sggc --file spec_file.txt
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



