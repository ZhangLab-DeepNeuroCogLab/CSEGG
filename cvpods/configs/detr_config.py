import os
import os.path as osp

import cvpods
from cvpods.configs.base_detection_config import BaseDetectionConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = osp.realpath(__file__)[:-9]

# use epoch rather than iteration in this model

_config_dict = dict(
    DEBUG=False,
    EXPERIMENT_NAME="detr",
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        # PIXEL_STD=[57.375, 57.120, 58.395],
        PIXEL_STD=[1.0,1.0,1.0],
        PIXEL_MEAN=[103.530, 116.280, 123.675], # detectron2 pixel normalization config
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
        ),
    ),

    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_EPOCH=150,
            MAX_ITER=None,
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
        EVAL_PERIOD=1,
    ),
    GLOBAL=dict(
        DUMP_TEST=False,
        LOG_INTERVAL=300
    ),
    OUTPUT_DIR="/home/naitik2/projects/SGG_Continual/models/experiments/c2/playground/sgg/detr.res101.c5.one_stage_rel_tfmer",
)


class DETRConfig(BaseDetectionConfig):
    def __init__(self):
        super(DETRConfig, self).__init__()
        self._register_configuration(_config_dict)


config = DETRConfig()
