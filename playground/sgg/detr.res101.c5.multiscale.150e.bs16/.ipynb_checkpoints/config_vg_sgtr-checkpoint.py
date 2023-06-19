import os
import os.path as osp

import cvpods
from cvpods.configs.base_detection_config import BaseDetectionConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = osp.realpath(__file__)[:-9]

# use epoch rather than iteration in this model

_config_dict = dict(
    DEBUG=False,
    EXPERIMENT_NAME="detr_vg_pre_train_imagenet_vg_short_factor_10_bbox_alpha_0.3_kd_loss_only",
    MODEL=dict(
        AS_PRETRAIN = True,
        WEIGHTS="/home/naitik/projects/SGG_Continual/models/experiments/SGTR_short/playground/sgg/detr.res101.c5.multiscale.150e.bs16/log/2022-11-27_17-36-detr_vg_pre_train_imagenet_vg_short_TASK_1/model_final.pth",
        # PIXEL_STD=[57.375, 57.120, 58.395],
        PIXEL_STD=[1.0,1.0,1.0],
        PIXEL_MEAN=[103.530, 116.280, 123.675], # detectron2 pixel normalization config
        WEIGHTS_Task_1="",
        WEIGHTS_Task_2="",
        MASK_ON=False,
        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res5"],
        ),
        DETR=dict(
            TRANSFORMER=dict(
                D_MODEL=256,
                N_HEAD=8,
                NUM_ENC_LAYERS=6,
                NUM_DEC_LAYERS=6,
                DIM_FFN=2048,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=False,
                RETURN_INTERMEDIATE_DEC=True,
            ),
            IN_FEATURES="res5",
            TEMPERATURE=10000,
            POSITION_EMBEDDING="sine",  # choice: [sine, learned]
            NUM_QUERIES=100,
            NO_AUX_LOSS=False,
            COST_CLASS=1.0,
            COST_BBOX=5.0,
            COST_GIOU=2.0,
            CLASS_LOSS_COEFF=1.0,
            BBOX_LOSS_COEFF=5.0,
            GIOU_LOSS_COEFF=2.0,
            EOS_COEFF=0.1,  # Relative classification weight of the no-object class
            NUM_CLASSES=150,  # for VG
        ),
        ROI_RELATION_HEAD=dict(
            ENABLED=False,
            USE_GT_BOX=False,
            USE_GT_OBJECT_LABEL=False,
            DATA_RESAMPLING=dict(
                ENABLED=False,

                METHOD="bilvl",
                REPEAT_FACTOR=0.2,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,

                ENTITY={
                    "ENABLED": False,
                    "REPEAT_FACTOR": 0.2,
                    "INSTANCE_DROP_RATE": 1.5,
                    "REPEAT_DICT_DIR": None,
                },

            ),
        ),
    ),
    DATASETS=dict(
        TRAIN=("vgs_train",),
        TEST=("vgs_val",),
        FILTER_EMPTY_ANNOTATIONS=True,
        FILTER_NON_OVERLAP=False,
        FILTER_DUPLICATE_RELS=True
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_EPOCH=100,
            #MAX_ITER=15000,
            # MAX_ITER=1000,
            WARMUP_ITERS=0,
            STEPS=(100,),
        ),
        OPTIMIZER=dict(
            NAME="DETRAdamWBuilder",
            BASE_LR=1e-4,
            BASE_LR_RATIO_BACKBONE=0.1,
            WEIGHT_DECAY=1e-4,
            BETAS=(0.9, 0.999),
            EPS=1e-08,
            AMSGRAD=False,
        ),
        CLIP_GRADIENTS=dict(
            ENABLED=True,
            CLIP_VALUE=0.1,
            CLIP_TYPE="norm",
            NORM_TYPE=2.0,
        ),
        IMS_PER_BATCH=18,
        IMS_PER_DEVICE=6,
        CHECKPOINT_PERIOD=5,
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(480, 496, 512, 536, 552, 576, 600,),
                    max_size=1000, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=600, max_size=1000, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        EVAL_PERIOD=6,
    ),
    GLOBAL=dict(
        DUMP_TEST=False,
        LOG_INTERVAL=300
    ),
    OUTPUT_DIR=curr_folder.replace(
        cvpods_home, os.getenv("CVPODS_OUTPUT")
    ),
)


class DETRConfig(BaseDetectionConfig):
    def __init__(self):
        super(DETRConfig, self).__init__()
        self._register_configuration(_config_dict)


config = DETRConfig()
