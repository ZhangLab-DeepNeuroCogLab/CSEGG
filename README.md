# Adaptive Visual Scene Understanding: Incremental Learning in Scene Graph Generation

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10.0-%237732a8)](https://pytorch.org/get-started/previous-versions/)

Link to our [paper](https://arxiv.org/pdf/2310.01636) 

Authors: Naitik Khandelwal, Xiao Liu, Mengmi Zhang

This repository houses our CSEGG benchmark implementation, encompassing source code for experimenting with Transformer-based SGG methods across various continual learning algorithms in all proposed learning scenarios outlined in our paper. Additionally, it includes the code for data generation in all the scenarios presented in the paper.

## Project Description 

Scene graph generation (SGG) analyzes images to extract meaningful information about objects and their relationships. In the dynamic visual world, it is crucial for AI systems to continuously detect new objects and establish their relationships with existing ones. Recently, numerous studies have focused on continual learning within the domains of object detection and image recognition. However, a limited amount of research focuses on a more challenging continual learning problem in SGG. This increased difficulty arises from the intricate interactions and dynamic relationships among objects, and their associated contexts. Thus, in continual learning, SGG models are often required to expand, modify, retain, and reason scene graphs within the process of adaptive visual scene understanding. To systematically explore Continual Scene Graph Generation (CSEGG), we present a comprehensive benchmark comprising three learning regimes: relationship incremental, scene incremental, and relationship generalization. Moreover, we introduce a Replays via Analysis by Synthesis method named RAS. This approach leverages the scene graphs, decomposes and re-composes them to represent different scenes, and replays the synthesized scenes based on these compositional scene graphs. The replayed synthesized scenes act as a means to practice and refine proficiency in SGG in known and unknown environments. Our experimental results not only highlight the challenges of directly combining existing continual learning methods with SGG backbones but also demonstrate the effectiveness of our proposed approach, enhancing CSEGG efficiency while simultaneously preserving privacy and memory usage.

Below is an illustration of all the learning scenarios in CSEGG:

| [![CSEGG learning scenarios](samples/illustration_learning_scenarios.png)](samples/illustration_learning_scenarios.png) | 
|:---:|
| CSEGG Learning Scenarios. |

From left to right, they are S1. relationship (Rel.) incremental learning (Incre.); S2. relationship and object (Rel. + Obj.) Incre.; and S3. relationship generalization (Rel. Gen.) in Object Incre.. In S1 and S2, example triplets in the training (solid line) and test sets (dash line) from each task are presented. The training and test sets from the same task are color-coded. The new objects or relationships in each task are bold and underlined. In S3, one single test set (dashed gray box) is used for benchmarking the relationship generalization ability of object incre. learning models across all the tasks.

<!--
Some visualization examples from all the scenarios are shown below.

| [![Visualization examples for Learning Scenario 1](samples/viz_S1.png)](samples/viz_S1.png) | 
|:---:|
| Visualization examples for Learning Scenario 1. |

| [![Visualization examples for Learning Scenario 2](samples/viz_S2.png)](samples/viz_S2.png) | 
|:---:|
| Visualization examples for Learning Scenario 2. |

| [![Visualization examples for Learning Scenario 3](samples/viz_S3.png)](samples/viz_S3.png) | 
|:---:|
| Visualization examples for Learning Scenario 3. | -->

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Now look at HOW_T0_USE.md for knowing various commands to run the train and eval scripts (especially if you are using multiple gpus)

Check [HOW_T0_USE.md](HOW_TO_USE.md) for the instructions

After running the evaluation code, you can load the "Final_Results.csv" in the jupyter notebook "results_figure.ipynb"

## Training and Evaluation

### Understanding Args 

Training:
- --num-gpus : Number of GPUs used for training. 
- --start_task : To resume the training from certain task.
- --sgg : To activate Stage 2 for Learning Scenario S2, S3. (This argument is not present in Learning Scenario S1).
- --continual : To choose which CSEGG model to train.
  - Learning Scenario S1 :- "replay_10", "ewc", "replay_100", "packnet", "ras" (Our new model. See RAS.md for details and ras folder for the code.
  .) To train "naive", exclude this argument from training command.
  - Learning Scenario S2 :- "replay_10", "ewc", "replay_20", "packnet", "ras". To train "naive", exclude this argument from training command.
  - Learning Scenario S3 :- "replay_10", "ras". To train "naive", exclude this argument from training command.

Evaluation:
- --num-gpus : Number of GPUs used for testing.

### Learning Scenario S1 

There is only Stage 2 training for Learning Scenario S1. To train the model, run the following in the command window:

```bash
cd ~/CSEGG/playground/sgg/detr.res101.c5.one_stage_rel_tfmer
pods_train_S1 --num-gpus 4 --continual "replay_10"

```
To evaluate,

```bash
cd ~/CSEGG/playground/sgg/detr.res101.c5.one_stage_rel_tfmer
pods_test_S1 --num-gpus 1 

```
### Learning Scenario S2 

To train the model, run the following in the command window:

```bash
#Stage 1
cd ~/CSEGG/playground/sgg/detr.res101.c5.multiscale.150e.bs16
pods_train_S2 --num-gpus 4 --continual "replay_10"
```

```bash
#Stage 2
cd ~/CSEGG/playground/sgg/detr.res101.c5.one_stage_rel_tfmer
pods_train_S2 --num-gpus 4 --continual "replay_10" --sgg "sgg"
```

To evaluate,

```bash
cd ~/CSEGG/playground/sgg/detr.res101.c5.one_stage_rel_tfmer
#Evaluation of Object Detection (Stage 1) and SGG (Stage 2) is combined
pods_test_S2 --num-gpus 1 

```

### Learning Scenario S3 

To train the model, run the following in the command window:

```bash
#Stage 1
cd ~/CSEGG/playground/sgg/detr.res101.c5.multiscale.150e.bs16
pods_train_S3 --num-gpus 4 --continual "replay_10"
```

```bash
#Stage 2
cd ~/CSEGG/playground/sgg/detr.res101.c5.one_stage_rel_tfmer
pods_train_S3 --num-gpus 4 --continual "replay_10" --sgg "sgg"
```

To evaluate,

```bash
#evaluation of R_bbox and R@k_relation_gen
cd ~/CSEGG/playground/sgg/detr.res101.c5.one_stage_rel_tfmer
pods_test_S3 --num-gpus 1 

```

## Acknowledgment
This repository borrows code from scene graph benchmarking frameworks: [Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) developed by KaihuaTang, [PySGG](https://github.com/SHTUPLUS/PySGG) and [SGTR](https://github.com/Scarecrow0/SGTR/tree/main) developed by Rongjie Li.

# Q&A

- Import ipdb in anywhere in your code will cause the multi-process initialization error, try pdb when you debug in multi-process mode.




