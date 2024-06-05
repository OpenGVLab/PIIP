import torch
import torch.nn as nn
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import (
    TRANSFORMER_LAYER_SEQUENCE, FEEDFORWARD_NETWORK, DROPOUT_LAYERS)
from mmdet.models.utils.transformer import inverse_sigmoid
from .transformer import DinoTransformerDecoder, DinoTransformer, build_MLP


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class GroupDinoTransformerDecoder(DinoTransformerDecoder):

    def __init__(self, *args, with_rp_noise=False, **kwargs):
        super(GroupDinoTransformerDecoder, self).__init__(*args, **kwargs)
        self.with_rp_noise = with_rp_noise
        self._init_layers()

    def _init_layers(self):
        self.ref_point_head = build_MLP(
            self.embed_dims * 2,
            self.embed_dims,
            self.embed_dims,
            2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]
        
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            
            if self.with_rp_noise and self.training:
                device = reference_points.device
                b, n, d = reference_points.size()
                noise = torch.rand(b, n, d).to(device) * 0.02 - 0.01
                reference_points = (reference_points + noise).clamp(0, 1)

            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            query_pos = query_pos.permute(1, 0, 2)
            
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                # TODO: should do earlier
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class GroupDinoTransformer(DinoTransformer):

    def __init__(self, group_nums, *args, **kwargs):
        self.group_nums = group_nums
        super(DinoTransformer, self).__init__(*args, **kwargs)

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals * self.group_nums,
                                        self.embed_dims)
        self.num_queries = self.two_stage_num_proposals

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        cls_out_features = cls_branches[self.decoder.num_layers].out_features
        topk = self.two_stage_num_proposals
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        # topk_proposal = torch.gather(
        #     output_proposals, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()
        # topk_memory = torch.gather(
        #     output_memory, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, self.embed_dims))
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_anchor = topk_coords_unact.sigmoid()
        # NOTE In the original DeformDETR, init_reference_out is obtained
        # from detached topk_coords_unact, which is different with DINO.  TODO
        topk_coords_unact = topk_coords_unact.detach()  # (bs, num_query, 4)
        
        bs, N = topk_coords_unact.shape[:2]
        if self.training:
            query = self.query_embed.weight[:, None, :].repeat(1, bs,
                                                           1).transpose(0, 1)
            topk_coords_unact = topk_coords_unact.unsqueeze(1).repeat(
                            1, self.group_nums, 1, 1).reshape(bs, N * self.group_nums, 4)
        else:
            query = self.query_embed.weight[:self.num_queries, None, :].repeat(1, bs,
                                                           1).transpose(0, 1) 

        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
            dn_num_queries = dn_bbox_query.shape[1]
        else:
            reference_points = topk_coords_unact
            dn_num_queries = 0
        reference_points = reference_points.sigmoid()
       
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            dn_num_queries=torch.tensor([dn_num_queries]).to(level_start_index),
            **kwargs)

        inter_references_out = inter_references
        return inter_states, inter_references_out, topk_score, topk_anchor