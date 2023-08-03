"""UnbiasedTeacher Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import functools

import numpy as np
import torch
from mmdet.models import DETECTORS, build_detector
from mmdet.models.detectors import BaseDetector
from mmdet.core.mask.structures import BitmapMasks
from mmdet.core import bbox2roi, multi_apply, bbox_overlaps, build_assigner
from mmcv.runner.fp16_utils import force_fp32

from otx.algorithms.common.utils.logger import get_logger

from .sam_detector_mixin import SAMDetectorMixin
from otx.algorithms.detection.adapters.mmdet.utils.paseco_utils import (Transform2D, filter_invalid,
                   resize_image)

logger = get_logger()

# TODO: Need to fix pylint issues
# pylint: disable=abstract-method, too-many-locals, unused-argument


@DETECTORS.register_module()
class MeanTeacher(SAMDetectorMixin, BaseDetector):
    """General mean teacher framework for instance segmentation."""
    def __init__(
        self,
        unlabeled_cls_loss_weight=1.0,
        unlabeled_reg_loss_weight=1.0,
        use_rpn_loss=True,
        pseudo_conf_thresh=0.7,
        enable_unlabeled_loss=False,
        bg_loss_weight=-1.0,
        min_pseudo_label_ratio=0.0,
        arch_type="CustomMaskRCNN",
        unlabeled_memory_bank=False,
        use_MSL=True,
        use_teacher_proposal=True,
        compute_mask_v2 = True,
        percentile=70,
        **kwargs
    ):
        super().__init__()
        self.unlabeled_cls_loss_weight = unlabeled_cls_loss_weight
        self.unlabeled_reg_loss_weight = unlabeled_reg_loss_weight
        self.unlabeled_loss_enabled = enable_unlabeled_loss
        self.use_MSL = use_MSL
        self.compute_mask_v2 = compute_mask_v2
        self.use_teacher_proposal = use_teacher_proposal
        self.unlabeled_memory_bank = unlabeled_memory_bank
        self.bg_loss_weight = bg_loss_weight
        self.min_pseudo_label_ratio = min_pseudo_label_ratio
        self.use_rpn_loss=use_rpn_loss
        self.percentile = percentile
        cfg = kwargs.copy()
        cfg["type"] = arch_type
        self.model_s = build_detector(cfg)
        self.model_t = copy.deepcopy(self.model_s)
        self.num_classes = cfg["roi_head"]["bbox_head"]["num_classes"]
        if self.unlabeled_memory_bank:
            self.memory_cat_bank = [[] for i in range(self.num_classes)]
            self.all_num_cat = 0
            self.pseudo_conf_thresh = None
        else:
            self.pseudo_conf_thresh = [pseudo_conf_thresh] * self.num_classes

        # initialize assignment to build condidate bags
        self.PLA_iou_thres = self.model_s.train_cfg.get("PLA_iou_thres", 0.4)
        initial_assigner_cfg=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=self.PLA_iou_thres,
            neg_iou_thr=self.PLA_iou_thres,
            match_low_quality=False,
            ignore_iof_thr=-1)
        self.initial_assigner = build_assigner(initial_assigner_cfg)

        # Hooks for super_type transparent weight load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    def extract_feat(self, imgs):
        """Extract features for UnbiasedTeacher."""
        return self.model_s.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        """Test from img with UnbiasedTeacher."""
        return self.model_s.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Aug Test from img with UnbiasedTeacher."""
        return self.model_s.aug_test(imgs, img_metas, **kwargs)

    def forward_dummy(self, img, **kwargs):
        """Dummy forward function for UnbiasedTeacher."""
        return self.model_s.forward_dummy(img, **kwargs)

    def enable_unlabeled_loss(self, mode=True):
        """Enable function for UnbiasedTeacher unlabeled loss."""
        self.unlabeled_loss_enabled = mode

    def turnoff_memory_bank(self):
        """Enable function for UnbiasedTeacher unlabeled loss."""
        self.unlabeled_memory_bank = False

    # def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
    #     """Test without augmentation."""
    #     # return self.model_s.simple_test(img, img_metas, **kwargs)
    #     model = self.model_s
    #     assert model.with_bbox, 'Bbox head must be implemented.'

    #     x = self.extract_feat(img, model, start_lvl=1)

    #     if proposals is None:
    #         proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
    #     else:
    #         proposal_list = proposals

    #     return model.roi_head.simple_test(
    #         x, proposal_list, img_metas, rescale=rescale)

    # def extract_feat(self, img, model="model_s", start_lvl=0):
    #     """Directly extract features from the backbone+neck."""
    #     # return self.model_s.extract_feat(img)
    #     model = self.model_s if model == "model_s" else model
    #     assert start_lvl in [0, 1], \
    #         f"start level {start_lvl} is not supported."
    #     x = model.backbone(img)
    #     # global feature -- [p2, p3, p4, p5, p6, p7]
    #     if model.with_neck:
    #         x = model.neck(x)
    #     if start_lvl == 0:
    #         return x[:-1]
    #     elif start_lvl == 1:
    #         return x[1:]

    def compute_dynamic_thrsh(self):
        """Enable function for UnbiasedTeacher unlabeled loss."""
        self.pseudo_conf_thresh = [0] * self.num_classes
        mediana = np.median([x for y in self.memory_cat_bank for x in y])
        for i, cat_scores in enumerate(self.memory_cat_bank):
            if len(cat_scores):
                coeff = np.percentile(np.array(cat_scores), self.percentile)
            else:
                coeff = mediana

            self.pseudo_conf_thresh[i] = coeff

        self.memory_cat_bank = None
        # per_cat_num_obj = np.array(self.memory_cat_bank) / self.all_num_cat
        # max_num_pbj = np.max(per_cat_num_obj)
        # range_of_variation = (max_num_pbj - 0.01) / 2 # 1%
        # # quadratic approximation. 0.01 -> 0.25, med -> 0.5, max -> 0.75
        # koeffs = np.polyfit([0.01, range_of_variation, max_num_pbj], [0.3, 0.5, 0.7], 2)
        # thrsh = [koeffs[0]*(x**2) + koeffs[1]*x + koeffs[2] for x in per_cat_num_obj]
        print(f"[*] Computed per class thresholds: {self.pseudo_conf_thresh}")

    def update_memory_bank(self, teacher_outputs, labeled_imgs, labeled_imgs_metas):

        with torch.no_grad():
            teacher_outputs_labeled = self.model_t.forward_test(
                [labeled_imgs],
                [labeled_imgs_metas],
                rescale=False,  # easy augmentation
            )

        for teacher_bboxes_labels in teacher_outputs_labeled:
            bboxes_l = teacher_bboxes_labels[0]
            for l, bb in enumerate(bboxes_l):
                confidences = bb[:, -1]
                if len(confidences):
                    self.memory_cat_bank[l].extend(confidences)

        for teacher_bboxes_labels in teacher_outputs:
            bboxes = teacher_bboxes_labels[0]
            for l, bb in enumerate(bboxes):
                confidences = bb[:, -1]
                if len(confidences):
                    self.memory_cat_bank[l].extend(confidences)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_masks, gt_bboxes_ignore=None, **kwargs):
        """Forward function for UnbiasedTeacher."""
        losses = {}
        # Supervised loss
        # TODO: check img0 only option (which is common for mean teacher method)
        sl_losses = self.model_s.forward_train(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore if gt_bboxes_ignore else None,
            gt_masks=gt_masks
        )
        losses.update(sl_losses)

        if not self.unlabeled_loss_enabled:
            return losses

        # Pseudo labels from teacher
        ul_args = kwargs.get("extra_0", {})  # Supposing ComposedDL([labeled, unlabeled]) data loader
        ul_img = ul_args.get("img")
        ul_img0 = ul_args.get("img0")
        ul_img_metas = ul_args.get("img_metas")
        if ul_img is None:
            return losses
        # with torch.no_grad():
        #     teacher_outputs = self.model_t.forward_test(
        #         [ul_img0],
        #         [ul_img_metas],
        #         rescale=False,  # easy augmentation
        #     )

        # current_device = ul_img0[0].device
        # pseudo_bboxes, pseudo_labels, pseudo_masks, pseudo_ratio = self.generate_pseudo_labels(
        #     teacher_outputs, device=current_device, img_meta=ul_img_metas, **kwargs
        # )
        # ps_recall = self.eval_pseudo_label_recall(pseudo_bboxes, ul_args.get("gt_bboxes", []))
        # losses.update(ps_recall=torch.tensor(ps_recall, device=current_device))
        # losses.update(ps_ratio=torch.tensor([pseudo_ratio], device=current_device))

        # Unsupervised loss
        # Compute only if min_pseudo_label_ratio is reached
        # if pseudo_ratio >= self.min_pseudo_label_ratio:
        #     if self.bg_loss_weight >= 0.0:
        #         self.model_s.bbox_head.bg_loss_weight = self.bg_loss_weight
        #     ul_losses = self.model_s.forward_train(ul_img, ul_img_metas, pseudo_bboxes, pseudo_labels, gt_masks=pseudo_masks)  # hard augmentation
        #     if self.bg_loss_weight >= 0.0:
        #         self.model_s.bbox_head.bg_loss_weight = -1.0

        ul_losses = self.foward_unsup_train(ul_img0, ul_img, ul_img_metas, ul_img_metas)  # hard augmentation

        for ul_loss_name in ul_losses.keys():
            if ul_loss_name.startswith("loss_"):
                # skip regression rpn loss
                if not self.use_rpn_loss and ul_loss_name == "loss_rpn_bbox":
                    # skip regression rpn loss
                    continue
                ul_loss = ul_losses[ul_loss_name]
                if "_bbox" in ul_loss_name:
                    if isinstance(ul_loss, list):
                        losses[ul_loss_name + "_ul"] = [loss * self.unlabeled_reg_loss_weight for loss in ul_loss]
                    else:
                        losses[ul_loss_name + "_ul"] = ul_loss * self.unlabeled_reg_loss_weight
                elif "_cls" in ul_loss_name:
                    # cls loss
                    if isinstance(ul_loss, list):
                        losses[ul_loss_name + "_ul"] = [loss * self.unlabeled_cls_loss_weight for loss in ul_loss]
                    else:
                        losses[ul_loss_name + "_ul"] = ul_loss * self.unlabeled_cls_loss_weight
                else:
                    # mask loss
                    if isinstance(ul_loss, list):
                        losses[ul_loss_name + "_ul"] = [loss * 1.0 for loss in ul_loss]
                    else:
                        losses[ul_loss_name + "_ul"] = ul_loss * 1.0
        return losses

    def forward_sup_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        forward training process for the labeled data.
        """
        losses = dict()
        # high resolution
        x = self.extract_feat(img, self.model_s, start_lvl=1)
        # RPN forward and loss
        if self.model_s.with_rpn:
            proposal_cfg = self.model_s.train_cfg.get('rpn_proposal',
                                              self.model_s.test_cfg.rpn)
            rpn_losses, proposal_list = self.model_s.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # RCNN forward and loss
        roi_losses = self.model_s.roi_head.forward_train(x, img_metas, proposal_list,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore, gt_masks,
                                                **kwargs)
        losses.update(roi_losses)

        return losses

    def foward_unsup_train(self, teacher_img, student_img, img_metas_teacher, img_metas_student):

        if len(img_metas_student) > 1:
            tnames = [meta["filename"] for meta in img_metas_teacher]
            snames = [meta["filename"] for meta in img_metas_student]
            tidx = [tnames.index(name) for name in snames]
            teacher_img = teacher_img[torch.Tensor(tidx).to(teacher_img.device).long()]
            img_metas_teacher = [img_metas_teacher[idx] for idx in tidx]

        det_bboxes, pseudo_labels, pseudo_masks, tea_proposals_tuple = self.extract_teacher_info(
                                teacher_img, img_metas_teacher)
        tea_proposals, tea_feats = tea_proposals_tuple
        tea_proposals_copy = copy.deepcopy(tea_proposals)    # proposals before geometry transform

        pseudo_bboxes = det_bboxes
        # pseudo_bboxes = self.convert_bbox_space(img_metas_teacher,img_metas_student, det_bboxes)
        # tea_proposals = self.convert_bbox_space(img_metas_teacher,img_metas_student, tea_proposals)
        loss = {}
        # RPN stage
        feats = self.model_s.extract_feat(student_img)
        stu_rpn_outs, rpn_losses = self.unsup_rpn_loss(
                feats, pseudo_bboxes, pseudo_labels, img_metas_student)
        loss.update(rpn_losses)

        if self.use_MSL:
            # construct View 2 to learn feature-level scale invariance
            img_ds = resize_image(student_img)   # downsampled images
            feats_ds = self.model_s.extract_feat(img_ds)
            img_metas_student2 = img_metas_student.copy()
            for metas in img_metas_student2:
                metas["img_shape"] = (*img_ds.shape[2:], 3)
            _, rpn_losses_ds = self.unsup_rpn_loss(feats_ds,
                                    pseudo_bboxes, pseudo_labels,
                                    img_metas_student2)
            for key, value in rpn_losses_ds.items():
                loss[key + "_V2"] = value

        # RCNN stage
        """ obtain proposals """
        if self.use_teacher_proposal:
            proposal_list = tea_proposals

        else :
            proposal_cfg = self.model_s.train_cfg.get(
                "rpn_proposal", self.model_s.test_cfg.rpn
            )
            proposal_list = self.model_s.rpn_head.get_bboxes(
                *stu_rpn_outs, img_metas=img_metas_student, cfg=proposal_cfg
            )

        """ obtain teacher predictions for all proposals """
        with torch.no_grad():
            rois_ = bbox2roi(tea_proposals_copy)
            tea_bbox_results = self.model_t.roi_head._bbox_forward(
                             tea_feats, rois_)

        teacher_infos = {
            "imgs": teacher_img,
            "cls_score": tea_bbox_results["cls_score"][:, :self.num_classes].softmax(dim=-1),
            "bbox_pred": tea_bbox_results["bbox_pred"],
            "feats": tea_feats,
            "img_metas": img_metas_teacher,
            "proposal_list": tea_proposals_copy}

        rcnn_losses = self.unsup_rcnn_cls_loss(
                            feats,
                            feats_ds if self.use_MSL else None,
                            img_metas_student,
                            proposal_list,
                            pseudo_bboxes,
                            pseudo_labels,
                            pseudo_masks=pseudo_masks,
                            teacher_infos=teacher_infos)

        loss.update(rcnn_losses)

        return loss

    def unsup_rpn_loss(self, stu_feats, pseudo_bboxes, pseudo_labels, img_metas):
        stu_rpn_outs = self.model_s.rpn_head(stu_feats)
        # rpn loss
        gt_bboxes_rpn = []
        for bbox, label in zip(pseudo_bboxes, pseudo_labels):
            bbox, label, _ = filter_invalid(
                bbox[:, :4],
                label=label,
                score=bbox[
                    :, 4
                ],  # TODO: replace with foreground score, here is classification score,
                thr=self.model_s.train_cfg.rpn_pseudo_threshold,
                min_size=self.model_s.train_cfg.min_pseduo_box_size,
            )
            gt_bboxes_rpn.append(bbox)

        stu_rpn_loss_inputs = stu_rpn_outs + ([bbox.float() for bbox in gt_bboxes_rpn], img_metas)
        rpn_losses = self.model_s.rpn_head.loss(*stu_rpn_loss_inputs)
        return stu_rpn_outs, rpn_losses

    def unsup_rcnn_cls_loss(self,
                        feat,
                        feat_V2,
                        img_metas,
                        proposal_list,
                        pseudo_bboxes,
                        pseudo_labels,
                        pseudo_masks,
                        teacher_infos=None):

        gt_bboxes, gt_labels, gt_masks = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            pseudo_masks,
            [image_meta["img_shape"][:-1] for image_meta in img_metas],
            thr=self.model_s.train_cfg.cls_pseudo_threshold,
            )

        sampling_results = self.prediction_guided_label_assign(
                    img_metas,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    teacher_infos=teacher_infos)

        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]

        bbox_targets = self.model_s.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, img_metas, rcnn_train_cfg=self.model_s.train_cfg.rcnn
        )
        mask_targets = self.model_s.roi_head.mask_head.get_targets(sampling_results, gt_masks,
                                                  rcnn_train_cfg=self.model_s.train_cfg.rcnn)
        labels = bbox_targets[0]

        rois = bbox2roi(selected_bboxes)
        bbox_results = self.model_s.roi_head._bbox_forward(feat, rois)
        mask_results = self.model_s.roi_head._mask_forward(feat, rois)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        bbox_weights = self.compute_PCV(
                bbox_results["bbox_pred"],
                labels,
                selected_bboxes,
                pos_gt_bboxes_list,
                pos_assigned_gt_inds_list)
        bbox_weights_ = bbox_weights.pow(2.0)
        pos_inds = (labels >= 0) & (labels < self.model_s.roi_head.bbox_head.num_classes)
        if pos_inds.any():
            reg_scale_factor = bbox_weights.sum() / bbox_weights_.sum()
        else:
            reg_scale_factor = 0.0

        # Focal loss
        loss = self.model_s.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *(bbox_targets[:3]),
            bbox_weights_,
            reduction_override="none",
        )
        loss_mask = self.model_s.roi_head.mask_head.loss(mask_results['mask_pred'][pos_inds], mask_targets, pos_labels)

        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = reg_scale_factor * loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0)
        loss["loss_mask"] = loss_mask["loss_mask"]

        if feat_V2 is not None:
            bbox_results_V2 = self.model_s.roi_head._bbox_forward(feat_V2, rois)
            if self.compute_mask_v2:
                mask_results_V2 = self.model_s.roi_head._mask_forward(feat_V2, rois)
            loss_V2 = self.model_s.roi_head.bbox_head.loss(
                bbox_results_V2["cls_score"],
                bbox_results_V2["bbox_pred"],
                rois,
                *(bbox_targets[:3]),
                bbox_weights_,
                reduction_override="none",
            )

            if self.compute_mask_v2:
                loss_mask_V2 = self.model_s.roi_head.mask_head.loss(mask_results_V2['mask_pred'][pos_inds],
                                        mask_targets, pos_labels)

            loss["loss_cls_V2"] = loss_V2["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
            loss["loss_bbox_V2"] = reg_scale_factor * loss_V2["loss_bbox"].sum() / max(
                bbox_targets[1].size()[0], 1.0)
            if self.compute_mask_v2:
                loss["loss_mask_V2"] = loss_mask_V2["loss_mask"]
            if "acc" in loss_V2:
                loss["acc_V2"] = loss_V2["acc"]

        return loss


    def extract_teacher_info(self, img, img_metas):
        feat = self.model_t.extract_feat(img)

        proposal_cfg = self.model_t.train_cfg.get(
            "rpn_proposal", self.model_t.test_cfg.rpn
        )
        rpn_out = list(self.model_t.rpn_head(feat))
        proposal_list = self.model_t.rpn_head.get_bboxes(*rpn_out, img_metas=img_metas, cfg=proposal_cfg)

        # teacher proposals
        proposals = copy.deepcopy(proposal_list)

        proposal_list, proposal_label_list = \
            self.model_t.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list,
            self.model_t.test_cfg.rcnn,
            rescale=False
        )   # obtain teacher predictions
        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        thr = self.model_s.train_cfg.pseudo_label_initial_score_thr

        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.model_s.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        segm_proposal2 = self.model_t.roi_head.simple_test_mask(
                feat, img_metas, det_bboxes, proposal_label_list, rescale=False)
        gt_masks = []
        for i, image_mask in enumerate(segm_proposal2):
            local_image_masks = []
            for mask_group in image_mask:
                if len(mask_group) > 0:
                    for m in mask_group:
                        local_image_masks.append(np.expand_dims(m, 0))

            if len(local_image_masks) > 0:
                gt_masks.append(BitmapMasks(np.concatenate(local_image_masks), *img_metas[i]["img_shape"][:-1]))
            else:
                gt_masks.append(BitmapMasks(local_image_masks, *img_metas[i]["img_shape"][:-1]))
        return det_bboxes, proposal_label_list, gt_masks, \
            (proposals, feat)

    @torch.no_grad()
    def compute_PCV(self,
                      bbox_preds,
                      labels,
                      proposal_list,
                      pos_gt_bboxes_list,
                      pos_assigned_gt_inds_list):
        """ Compute regression weights for each proposal according
            to Positive-proposal Consistency Voting (PCV).

        Args:
            bbox_pred (Tensors): bbox preds for proposals.
            labels (Tensors): assigned class label for each proposals.
                0-79 indicate fg, 80 indicates bg.
            propsal_list tuple[Tensor]: proposals for each image.
            pos_gt_bboxes_list, pos_assigned_gt_inds_list tuple[Tensor]: label assignent results

        Returns:
            bbox_weights (Tensors): Regression weights for proposals.
        """

        nums = [_.shape[0] for _ in proposal_list]
        labels = labels.split(nums, dim=0)
        bbox_preds = bbox_preds.split(nums, dim=0)

        bbox_weights_list = []

        for bbox_pred, label, proposals, pos_gt_bboxes, pos_assigned_gt_inds in zip(
                    bbox_preds, labels, proposal_list, pos_gt_bboxes_list, pos_assigned_gt_inds_list):

            pos_inds = ((label >= 0) &
                        (label < self.model_s.roi_head.bbox_head.num_classes)).nonzero().reshape(-1)
            bbox_weights = proposals.new_zeros(bbox_pred.shape[0], 4)
            pos_proposals = proposals[pos_inds]
            if len(pos_inds):
                pos_bbox_weights = proposals.new_zeros(pos_inds.shape[0], 4)
                pos_bbox_pred = bbox_pred.view(
                            bbox_pred.size(0), -1, 4)[
                                pos_inds, label[pos_inds]
                            ]
                decoded_bboxes = self.model_s.roi_head.bbox_head.bbox_coder.decode(
                        pos_proposals, pos_bbox_pred)

                gt_inds_set = torch.unique(pos_assigned_gt_inds)

                IoUs = bbox_overlaps(
                    decoded_bboxes,
                    pos_gt_bboxes,
                    is_aligned=True)

                for gt_ind in gt_inds_set:
                    idx_per_gt = (pos_assigned_gt_inds == gt_ind).nonzero().reshape(-1)
                    if idx_per_gt.shape[0] > 0:
                        pos_bbox_weights[idx_per_gt] = IoUs[idx_per_gt].mean()
                bbox_weights[pos_inds] = pos_bbox_weights

            bbox_weights_list.append(bbox_weights)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        return bbox_weights

    @torch.no_grad()
    def prediction_guided_label_assign(
                self,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                teacher_infos,
                gt_bboxes_ignore=None,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        # get teacher predictions (including cls scores and bbox ious)
        tea_proposal_list = teacher_infos["proposal_list"]
        tea_cls_score_concat = teacher_infos["cls_score"]
        tea_bbox_pred_concat = teacher_infos["bbox_pred"]
        num_per_img = [_.shape[0] for _ in tea_proposal_list]
        tea_cls_scores = tea_cls_score_concat.split(num_per_img, dim=0)
        tea_bbox_preds = tea_bbox_pred_concat.split(num_per_img, dim=0)

        decoded_bboxes_list = []
        for bbox_preds, cls_scores, proposals in zip(tea_bbox_preds, tea_cls_scores, tea_proposal_list):
            pred_labels = cls_scores.max(dim=-1)[1]

            bbox_preds_ = bbox_preds.view(
                bbox_preds.size(0), -1,
            4)[torch.arange(bbox_preds.size(0)), pred_labels]
            decode_bboxes = self.model_s.roi_head.bbox_head.bbox_coder.decode(proposals, bbox_preds_)
            decoded_bboxes_list.append(decode_bboxes)

        # decoded_bboxes_list = self.convert_bbox_space(
        #                         teacher_infos['img_metas'],
        #                         img_metas,
        #                         decoded_bboxes_list)

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.initial_assigner.assign(
                decoded_bboxes_list[i],
                gt_bboxes[i],
                gt_bboxes_ignore[i],
                gt_labels[i])

            gt_inds = assign_result.gt_inds
            pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).reshape(-1)

            assigned_gt_inds = gt_inds - 1
            pos_assigned_gt_inds = assigned_gt_inds[pos_inds]
            pos_labels = gt_labels[i][pos_assigned_gt_inds]

            tea_pos_cls_score = tea_cls_scores[i][pos_inds]

            tea_pos_bboxes = decoded_bboxes_list[i][pos_inds]
            ious = bbox_overlaps(tea_pos_bboxes, gt_bboxes[i])

            wh = proposal_list[i][:, 2:4] - proposal_list[i][:, :2]
            areas = wh.prod(dim=-1)
            pos_areas = areas[pos_inds]

            refined_gt_inds = self.assignment_refinement(gt_inds,
                                       pos_inds,
                                       pos_assigned_gt_inds,
                                       ious,
                                       tea_pos_cls_score,
                                       pos_areas,
                                       pos_labels)

            assign_result.gt_inds = refined_gt_inds + 1
            sampling_result = self.model_s.roi_head.bbox_sampler.sample(
                                assign_result,
                                proposal_list[i],
                                gt_bboxes[i],
                                gt_labels[i])
            sampling_results.append(sampling_result)
        return sampling_results

    @torch.no_grad()
    def assignment_refinement(self, gt_inds, pos_inds, pos_assigned_gt_inds,
                             ious, cls_score, areas, labels):
        # (PLA) refine assignment results according to teacher predictions
        # on each image
        refined_gt_inds = gt_inds.new_full((gt_inds.shape[0], ), -1)
        refined_pos_gt_inds = gt_inds.new_full((pos_inds.shape[0],), -1)

        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
            target_labels = labels[pos_idx_per_gt]
            target_scores = cls_score[pos_idx_per_gt, target_labels]
            target_areas = areas[pos_idx_per_gt]
            target_IoUs = ious[pos_idx_per_gt, gt_ind]

            cost = (target_IoUs * target_scores).sqrt()
            _, sort_idx = torch.sort(cost, descending=True)

            candidate_topk = min(pos_idx_per_gt.shape[0], self.model_s.train_cfg.PLA_candidate_topk)
            topk_ious, _ = torch.topk(target_IoUs, candidate_topk, dim=0)
            # calculate dynamic k for each gt
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            sort_idx = sort_idx[:dynamic_ks]
            # filter some invalid (area == 0) proposals
            sort_idx = sort_idx[
                target_areas[sort_idx] > 0
            ]
            pos_idx_per_gt = pos_idx_per_gt[sort_idx]

            refined_pos_gt_inds[pos_idx_per_gt] = pos_assigned_gt_inds[pos_idx_per_gt]

        refined_gt_inds[pos_inds] = refined_pos_gt_inds
        return refined_gt_inds

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def convert_bbox_space(self, img_metas_A, img_metas_B, bboxes_A):
        """
            function: convert bboxes_A from space A into space B
            Parameters:
                img_metas: list(dict); bboxes_A: list(tensors)
        """
        transMat_A = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                                for meta in img_metas_A]
        transMat_B = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                            for meta in img_metas_B]
        M = self._get_trans_mat(transMat_A, transMat_B)
        bboxes_B = self._transform_bbox(
            bboxes_A,
            M,
            [meta["img_shape"] for meta in img_metas_B],
        )
        return bboxes_B

    def generate_pseudo_labels(self, teacher_outputs, img_meta, **kwargs):
        """Generate pseudo label for UnbiasedTeacher."""
        device = kwargs.pop("device")
        all_pseudo_bboxes = []
        all_pseudo_labels = []
        all_pseudo_masks = []
        num_all_bboxes = 0
        num_all_pseudo = 0
        ori_image_shape = img_meta[0]["img_shape"][:-1]
        for teacher_bboxes_labels in teacher_outputs:
            pseudo_bboxes = []
            pseudo_labels = []
            pseudo_masks = []
            bboxes = teacher_bboxes_labels[0]
            masks = teacher_bboxes_labels[1]
            for label, teacher_bboxes_masks in enumerate(zip(bboxes, masks)):
                teacher_bboxes = teacher_bboxes_masks[0]
                teacher_masks = teacher_bboxes_masks[1]
                confidences = teacher_bboxes[:, -1]
                pseudo_indices = confidences > self.pseudo_conf_thresh[label]
                pseudo_bboxes.append(teacher_bboxes[pseudo_indices, :4])  # model output: [x y w h conf]
                pseudo_labels.append(np.full([sum(pseudo_indices)], label))
                if np.any(pseudo_indices):
                    teacher_masks = [np.expand_dims(mask, 0) for mask in teacher_masks]
                    pseudo_masks.append(np.concatenate(teacher_masks)[pseudo_indices])
                else:
                    pseudo_masks.append(np.array([]).reshape(0, *ori_image_shape))

                num_all_bboxes += teacher_bboxes.shape[0]
                if len(pseudo_bboxes):
                    num_all_pseudo += pseudo_bboxes[-1].shape[0]

            if len(pseudo_bboxes) > 0:
                all_pseudo_bboxes.append(torch.from_numpy(np.concatenate(pseudo_bboxes)).to(device))
                all_pseudo_labels.append(torch.from_numpy(np.concatenate(pseudo_labels)).to(device))
                all_pseudo_masks.append(BitmapMasks(np.concatenate(pseudo_masks), *ori_image_shape))

        pseudo_ratio = float(num_all_pseudo) / num_all_bboxes if num_all_bboxes > 0 else 0.0
        return all_pseudo_bboxes, all_pseudo_labels, all_pseudo_masks, pseudo_ratio

    def eval_pseudo_label_recall(self, all_pseudo_bboxes, all_gt_bboxes):
        """Eval pseudo label recall for test only."""
        from mmdet.core.evaluation.recall import _recalls, bbox_overlaps

        img_num = len(all_gt_bboxes)
        if img_num == 0:
            return [0.0]
        all_ious = np.ndarray((img_num,), dtype=object)
        for i in range(img_num):
            ps_bboxes = all_pseudo_bboxes[i]
            gt_bboxes = all_gt_bboxes[i]
            # prop_num = min(ps_bboxes.shape[0], 100)
            prop_num = ps_bboxes.shape[0]
            if gt_bboxes is None or gt_bboxes.shape[0] == 0:
                ious = np.zeros((0, ps_bboxes.shape[0]), dtype=np.float32)
            elif ps_bboxes is None or ps_bboxes.shape[0] == 0:
                ious = np.zeros((gt_bboxes.shape[0], 0), dtype=np.float32)
            else:
                ious = bbox_overlaps(gt_bboxes.detach().cpu().numpy(), ps_bboxes.detach().cpu().numpy()[:prop_num, :4])
            all_ious[i] = ious
        recall = _recalls(all_ious, np.array([100]), np.array([0.5]))
        return recall

    @staticmethod
    def state_dict_hook(module, state_dict, prefix, *args, **kwargs):  # pylint: disable=unused-argument
        """Redirect student model as output state_dict (teacher as auxilliary)."""
        logger.info("----------------- MeanTeacherSegmentor.state_dict_hook() called")
        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            if not prefix or key.startswith(prefix):
                key = key.replace(prefix, "", 1)
                if key.startswith("model_s."):
                    key = key.replace("model_s.", "", 1)
                elif key.startswith("model_t."):
                    continue
                key = prefix + key
            state_dict[key] = value
        return state_dict

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):  # pylint: disable=unused-argument
        """Redirect input state_dict to teacher model."""
        logger.info("----------------- MeanTeacherSegmentor.load_state_dict_pre_hook() called")
        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            state_dict["model_s." + key] = value
            state_dict["model_t." + key] = value
