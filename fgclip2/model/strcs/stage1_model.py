from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

from .fgclip2 import FG_CLIP2_Model
from .modeling_fgclip2 import Fgclip2Output


class FG_CLIP2_Stage1_Model(FG_CLIP2_Model):
    """Stage1 model with only global short-text and long-text heads."""

    main_input_name = "text_long"

    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, "boxtext_head"):
            del self.boxtext_head
        if hasattr(self, "dense_feature_head"):
            del self.dense_feature_head
        self.config.training_stage = 1

    def copy_weight(self):
        with torch.no_grad():
            self.longtext_head.weight.data.copy_(self.text_model.head.weight.data)
            self.longtext_head.bias.data.copy_(self.text_model.head.bias.data)

    def copy_dense_feature_head(self):
        raise RuntimeError("Stage1 model does not contain dense_feature_head. Use FG_CLIP2_Stage2_Model.")

    def get_dense_feature(self, *args, **kwargs):
        raise RuntimeError("Stage1 model does not support dense ROI features. Use FG_CLIP2_Stage2_Model.")

    def get_image_box_roi_features(self, *args, **kwargs):
        raise RuntimeError("Stage1 model does not support box ROI features. Use FG_CLIP2_Stage2_Model.")

    def forward(
        self,
        text_short: Optional[torch.LongTensor] = None,
        text_long: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_box_loss: bool = False,
        use_hard_neg: bool = False,
        **kwargs,
    ) -> Union[Tuple, Fgclip2Output]:
        if add_box_loss or use_hard_neg:
            raise RuntimeError("Stage1 forward does not accept box or hard-negative losses.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rank = dist.get_rank()

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        if text_short is None and text_long is None:
            raise ValueError("At least one of `text_short` or `text_long` must be provided.")

        image_embeds = vision_outputs[1]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        short_text_embeds = None
        if text_short is not None:
            short_text_outputs = self.text_model(
                input_ids=text_short,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            short_text_embeds = short_text_outputs[1]
            short_text_embeds = short_text_embeds / short_text_embeds.norm(p=2, dim=-1, keepdim=True)

        long_text_embeds = None
        if text_long is not None:
            long_text_outputs = self.text_model(
                input_ids=text_long,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                walk_type="long",
            )
            long_text_embeds = long_text_outputs[1]
            long_text_embeds = self.longtext_head(long_text_embeds)
            long_text_embeds = long_text_embeds / long_text_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logit_bias = self.logit_bias
        loss = None
        if self.loss_type == "gather":
            if text_long is not None:
                loss = self.all_gather_siglip_loss_(image_embeds, long_text_embeds, logit_scale, logit_bias, rank)
            if text_short is not None:
                loss_short = self.all_gather_siglip_loss_(image_embeds, short_text_embeds, logit_scale, logit_bias, rank)
                loss = loss_short if loss is None else loss + loss_short
        elif self.loss_type == "reduce":
            if text_long is not None:
                loss = self.all_reduce_siglip_loss(image_embeds, long_text_embeds, logit_scale, logit_bias, rank)
            if text_short is not None:
                loss_short = self.all_reduce_siglip_loss(image_embeds, short_text_embeds, logit_scale, logit_bias, rank)
                loss = loss_short if loss is None else loss + loss_short
        else:
            assert self.loss_type is not None

        return Fgclip2Output(loss=loss)
