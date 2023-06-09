#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by BaseDetection group. All Rights Reserved

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "cvpods", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("D2_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [line for line in init_py if not line.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "cvpods", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (
        torch.cuda.is_available() and CUDA_HOME is not None and os.path.isdir(CUDA_HOME)
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "cvpods._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    cvpods/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "cvpods", "model_zoo", "configs"
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True)
    return config_paths


def build_cvpods_script():
    cur_dir = os.getcwd()
    head = "#!/bin/bash\n\nexport OMP_NUM_THREADS=1\n\n"
    with open("tools/pods_train_S1", "w") as pods_train:
        pods_train.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'train_net_s1.py')} $@")

    with open("tools/pods_train_S2", "w") as pods_train:
        pods_train.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'train_net_s2.py')} $@")
    
    with open("tools/pods_train_S3", "w") as pods_train:
        pods_train.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'train_net_s3.py')} $@")

    with open("tools/pods_test_S1", "w") as pods_test:
        pods_test.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'test_net_s1.py')} $@")
    
    with open("tools/pods_test_S2", "w") as pods_test:
        pods_test.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'test_net_s2.py')} $@")

    with open("tools/pods_test_S3", "w") as pods_test:
        pods_test.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'test_net_s3.py')} $@")

    with open("tools/pods_debug", "w") as pods_debug:
        pods_debug.write(head + f"python3 {os.path.join(cur_dir, 'tools', 'debug_net.py')} $@")


if __name__ == "__main__":
    build_cvpods_script()
    setup(
        name="cvpods",
        version=get_version(),
        author="BaseDetection",
        description="cvpods is BaseDetection's research "
        "platform for object detection and segmentation based on cvpods.",
        packages=find_packages(exclude=("configs", "tests")),
        python_requires=">=3.6",
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
        scripts=[
            "tools/pods_train_S1",
            "tools/pods_train_S2",
            "tools/pods_train_S3",
            "tools/pods_test_S1",
            "tools/pods_test_S2",
            "tools/pods_test_S3",
            "tools/pods_debug",
        ],
    )
