import logging

from cvpods.modeling.backbone import Backbone, build_resnet_backbone
from cvpods.modeling.meta_arch.detr import DETR
from cvpods.layers import ShapeSpec

import sys
sys.path.append("..")
# must import you new add models, for calling the init function
from optimizer import OPTIMIZER_BUILDER  # noqa



def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_model(cfg,task_number):
    cfg.build_backbone = build_backbone
    model = DETR(cfg,task_number)

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
