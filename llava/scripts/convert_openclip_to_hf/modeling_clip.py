from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, VisionTextDualEncoderConfig, VisionTextDualEncoderModel
from transformers.models.vision_text_dual_encoder.modeling_vision_text_dual_encoder import clip_loss, CLIPOutput


class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x, attention_mask):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


class OpenCLIPVisionTextDualEncoderModel(VisionTextDualEncoderModel):
    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
        add_text_model_pooling_layer: bool = False,
    ):
        super().__init__(config, vision_model, text_model)

        # Remove text pooling layer
        if not add_text_model_pooling_layer:
            self.text_model.pooler = None

        # Add mean pooling
        self.pooler = MeanPooler()
        # Overwrite text projection
        hidden_size = (self.text_embed_dim + self.projection_dim) // 2
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, self.projection_dim, bias=False),
        )

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.pooler(text_outputs, attention_mask)
        text_features = self.text_projection(pooled_output)

        return text_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]  # pooler_output
        image_embeds = self.visual_projection(image_embeds)

        pooled_output = self.pooler(text_outputs, attention_mask)
        text_embeds = self.text_projection(pooled_output)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )