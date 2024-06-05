# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox2result, bbox_cxcywh_to_xyxy, bbox_flip
from .detr import DETR

import torch
import warnings 
warnings.filterwarnings("ignore", category=Warning)


@DETECTORS.register_module()
class DINO(DETR):

    def __init__(self, rule=None, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
        for k, v in self.named_parameters():
            if rule == "freeze_backbone":
                if "backbone" in k:
                    v.requires_grad = False
            if rule == "freeze_backbone_encoder":
                if "backbone" in k or "encoder" in k:
                    v.requires_grad = False
            if rule == "freeze_stage_1_2":
                if "patch_embed" in k:
                    v.requires_grad = False
                if "levels.0." in k or "levels.1." in k:
                    v.requires_grad = False
            if rule == "freeze_stage_1_2_3":
                if "patch_embed" in k:
                    v.requires_grad = False
                if "levels.0." in k or "levels.1." in k or "levels.2." in k:
                    v.requires_grad = False
        
    def aug_test(self, imgs, img_metas, rescale=False):
        return [self.aug_test_flip(imgs, img_metas, rescale)]
    
    def rescale_boxes(self, det_bboxes, det_scores, img_meta):
        det_scores = det_scores.sigmoid()  # [900, 80]
        scores, indexes = det_scores.view(-1).topk(self.test_cfg.max_per_img)
        bbox_index = indexes // self.bbox_head.num_classes
        det_labels = indexes % self.bbox_head.num_classes
        det_bboxes = det_bboxes[bbox_index]
        det_scores = det_scores[bbox_index]
    
        det_bboxes = bbox_cxcywh_to_xyxy(det_bboxes)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
    
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1] # to image-scale
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        det_bboxes = bbox_flip(det_bboxes, img_shape,
                               flip_direction) if flip else det_bboxes
        det_bboxes = det_bboxes.view(-1, 4) / det_bboxes.new_tensor(scale_factor) # to object-scale
        return det_bboxes, scores, det_labels
    
    def aug_test_flip(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.
        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        
        feats = self.extract_feats(imgs)

        aug_bboxes, aug_scores, aug_labels = [], [], []

        for i, (feat, img_meta) in enumerate(zip(feats, img_metas)):
            self.bbox_head = self.bbox_head.to(torch.float32)
            det_bboxes, det_logits = self.bbox_head.tta_test_bboxes(
                feat, img_meta, rescale=False) # [1, 300, 4] & [1, 300, 80]
            # cxcywh, [0-1]
            det_bboxes = det_bboxes[0] # [300, 4]
            det_logits = det_logits[0] # [300, 80]
            det_bboxes, det_scores, det_labels = self.rescale_boxes(det_bboxes, det_logits, img_meta)

            aug_bboxes.append(det_bboxes) # [n, 4]
            aug_scores.append(det_scores) # [n]
            aug_labels.append(det_labels) # [n]

        aug_bboxes = torch.cat((aug_bboxes[1], aug_scores[1].unsqueeze(1)), -1) # [300, 5]
        bbox_results = bbox2result(aug_bboxes, aug_labels[1], self.bbox_head.num_classes)
        return bbox_results