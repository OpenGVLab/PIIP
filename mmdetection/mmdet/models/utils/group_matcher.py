# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# HungarianMatcher
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/Atten4Vis/ConditionalDETR/blob/GroupDETR/models/matcher.py
# ------------------------------------------------------------------------------------------------
import torch
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import BaseAssigner, AssignResult
from mmdet.core.bbox.match_costs import build_match_cost


@BBOX_ASSIGNERS.register_module()
class GroupHungarianMatcher(BaseAssigner):
    """HugarianMatcher supports Group-DETR

    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
    """

    def __init__(self, 
                 num_query,
                 group_nums,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)):
        self.num_query = num_query
        self.group_nums = group_nums 
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        if num_bboxes > self.num_query:
            g_num_bbox = num_bboxes // self.group_nums
            cost_list = cost.split(g_num_bbox, dim=0)

            row_inds_list, col_inds_list = [], []
            for g_i in range(self.group_nums):
                cost_g = cost_list[g_i]
                cost_g = cost_g.detach().cpu()
                matched_row_inds, matched_col_inds = linear_sum_assignment(cost_g)
                matched_row_inds = torch.from_numpy(matched_row_inds).to(
                    bbox_pred.device)
                matched_col_inds = torch.from_numpy(matched_col_inds).to(
                    bbox_pred.device)
                if g_i == 0:
                    row_inds_list.append(matched_row_inds)
                else:
                    row_inds_list.append(matched_row_inds + g_i * g_num_bbox)

                col_inds_list.append(matched_col_inds)
            
            matched_row_inds = torch.cat(row_inds_list, dim=0)
            matched_col_inds = torch.cat(col_inds_list, dim=0)

        else:
            # encoder queries
            cost = cost.detach().cpu()
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                bbox_pred.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)