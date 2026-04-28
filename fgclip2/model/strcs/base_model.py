from typing import Optional

import torch
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
from torch import nn

from .configuration_fgclip2 import Fgclip2Config
from .modeling_fgclip2 import Fgclip2Model, Fgclip2TextModel, Fgclip2VisionModel


class FG_CLIP2_Base_Model(Fgclip2Model):
    config_class = Fgclip2Config
    main_input_name = "text_long"

    def __init__(self, config: Fgclip2Config, post_init: bool = True):
        super().__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        r'''
        # First, initialize the text and vision models with proper attention implementation
        # defalut is sdpa
        # NOTE If you need to train FG_CLIP2_Model and your device supports flash_attn, you can open them!
        text_config._attn_implementation = "flash_attention_2"
        vision_config._attn_implementation = "flash_attention_2"

        '''

        text_model = Fgclip2TextModel._from_config(text_config)
        vision_model = Fgclip2VisionModel._from_config(vision_config)

        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model
        self.vision_config = vision_config

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        self.embed_dim = text_config.hidden_size
        self.longtext_head = nn.Linear(self.embed_dim, self.embed_dim)

        self.thresholds = 0.0
        self.pad_token_id = 0
        self.world_size = 0
        self.loss_type = None

        if post_init:
            self.post_init()

    def copy_weight(self):
        with torch.no_grad():
            self.longtext_head.weight.data.copy_(self.text_model.head.weight.data)
            self.longtext_head.bias.data.copy_(self.text_model.head.bias.data)

    def copy_dense_feature_head(self):
        raise RuntimeError("Base model does not contain dense_feature_head. Use FG_CLIP2_Stage2_Model.")

    def resize_postion_embeding(self, newsize=196):
        old_position_embedding_res = self.text_model.embeddings.position_embedding_res
        old_position_embedding_ori = self.text_model.embeddings.position_embedding_ori

        positional_embedding_pre = self.text_model.embeddings.position_embedding.weight.data

        length, dim = positional_embedding_pre.shape
        keep_len = 20
        posisitonal_embedding_new = torch.zeros([4 * length - 3 * keep_len, dim], dtype=positional_embedding_pre.dtype)
        for i in range(keep_len):
            posisitonal_embedding_new[i] = positional_embedding_pre[i]
        for i in range(length - 1 - keep_len):
            posisitonal_embedding_new[4 * i + keep_len] = positional_embedding_pre[i + keep_len]
            posisitonal_embedding_new[4 * i + 1 + keep_len] = (
                3 * positional_embedding_pre[i + keep_len] / 4
                + 1 * positional_embedding_pre[i + 1 + keep_len] / 4
            )
            posisitonal_embedding_new[4 * i + 2 + keep_len] = (
                2 * positional_embedding_pre[i + keep_len] / 4
                + 2 * positional_embedding_pre[i + 1 + keep_len] / 4
            )
            posisitonal_embedding_new[4 * i + 3 + keep_len] = (
                1 * positional_embedding_pre[i + keep_len] / 4
                + 3 * positional_embedding_pre[i + 1 + keep_len] / 4
            )

        posisitonal_embedding_new[4 * length - 3 * keep_len - 4] = positional_embedding_pre[length - 1] + 0 * (
            positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]
        ) / 4
        posisitonal_embedding_new[4 * length - 3 * keep_len - 3] = positional_embedding_pre[length - 1] + 1 * (
            positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]
        ) / 4
        posisitonal_embedding_new[4 * length - 3 * keep_len - 2] = positional_embedding_pre[length - 1] + 2 * (
            positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]
        ) / 4
        posisitonal_embedding_new[4 * length - 3 * keep_len - 1] = positional_embedding_pre[length - 1] + 3 * (
            positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]
        ) / 4

        positional_embedding_res = posisitonal_embedding_new.clone()

        self.text_model.embeddings.position_embedding_ori.weight.data = posisitonal_embedding_new
        self.text_model.embeddings.position_embedding_ori.num_embeddings = posisitonal_embedding_new.shape[0]

        self.text_model.embeddings.position_embedding_res.weight.data = positional_embedding_res
        self.text_model.embeddings.position_embedding_res.num_embeddings = positional_embedding_res.shape[0]

        old_position_embedding_ori_requires_grad = old_position_embedding_ori.weight.requires_grad
        self.text_model.embeddings.position_embedding_ori.requires_grad_(old_position_embedding_ori_requires_grad)

        old_position_embedding_res_requires_grad = old_position_embedding_res.weight.requires_grad
        self.text_model.embeddings.position_embedding_res.requires_grad_(old_position_embedding_res_requires_grad)

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        need_densefeature: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]
        if need_densefeature:
            return pooled_output, self.get_dense_feature(vision_outputs.last_hidden_state, pixel_attention_mask)
        return pooled_output

    def get_dense_feature(self, *args, **kwargs):
        raise RuntimeError("Base model does not support dense ROI features. Use FG_CLIP2_Stage2_Model.")

    def get_image_box_roi_features(self, *args, **kwargs):
        raise RuntimeError("Base model does not support box ROI features. Use FG_CLIP2_Stage2_Model.")

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]

        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes.type(torch.float32))
        return denormed_boxes

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            eyevalue = torch.eye(num_logits, device=device, dtype=torch.float)
            labels = 2 * eyevalue.bfloat16() + labels

        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid((labels * logits).float()).sum() / image_features.shape[0]
        return loss

    def all_gather_siglip_loss_(self, image_features, text_features, logit_scale, logit_bias, cur_rank, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        text_features_all = torch.stack(nn_dist.all_gather(text_features), dim=0)

        for i in range(self.world_size):
            loss += float(i != cur_rank) * self._loss(
                image_features,
                text_features_all[i],
                logit_scale,
                logit_bias,
                negative_only=True,
            )

        return loss

    def all_reduce_siglip_loss(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias,
        cur_rank,
        no_longtext_indices=None,
        output_dict=False,
    ):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        for i in range(self.world_size):
            text_from_other = torch.distributed.nn.all_reduce(
                text_features * (cur_rank == i),
                torch.distributed.ReduceOp.SUM,
            )

            loss += float(i != cur_rank) * self._loss(
                image_features,
                text_from_other,
                logit_scale,
                logit_bias,
                negative_only=True,
            )

        return loss
