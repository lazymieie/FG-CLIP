from typing import Union

import torch

from .fgclip2 import FG_CLIP2_Model
from .stage1_model import FG_CLIP2_Stage1_Model


class FG_CLIP2_Stage2_Model(FG_CLIP2_Model):
    """Stage2 model with box text head and dense ROI feature head."""

    pass


def init_stage2_heads_from_backbone(model: FG_CLIP2_Stage2_Model):
    model.copy_weight()
    model.copy_dense_feature_head()
    return model


def load_stage1_into_stage2(
    stage2_model: FG_CLIP2_Stage2_Model,
    stage1_model_or_path: Union[str, FG_CLIP2_Stage1_Model],
):
    if isinstance(stage1_model_or_path, str):
        stage1_state_dict = FG_CLIP2_Stage1_Model.from_pretrained(stage1_model_or_path).state_dict()
    else:
        stage1_state_dict = stage1_model_or_path.state_dict()

    missing_keys, unexpected_keys = stage2_model.load_state_dict(stage1_state_dict, strict=False)
    with torch.no_grad():
        stage2_model.boxtext_head.weight.data.copy_(stage2_model.text_model.head.weight.data)
        stage2_model.boxtext_head.bias.data.copy_(stage2_model.text_model.head.bias.data)
    stage2_model.copy_dense_feature_head()
    stage2_model.config.training_stage = 2
    return missing_keys, unexpected_keys
