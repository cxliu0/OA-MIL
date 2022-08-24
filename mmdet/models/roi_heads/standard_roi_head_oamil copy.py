import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from mmdet.models.losses import accuracy
import math
import mmdet.core.bbox.iou_calculators.iou2d_calculator as iou

import copy

@HEADS.register_module()
class StandardRoIHeadOAMIL(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """                
        
        def get_sampling_results(img_metas, proposal_list, gt_bboxes, gt_bboxes_ignore):
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            return sampling_results

        def get_new_gt_boxes(gt_bboxes, noisy_gt_boxes, pseudo_gt_boxes):
            # update noisy gt for anchor reassignment
            new_gt_bboxes = copy.deepcopy(gt_bboxes)
            if pseudo_gt_boxes is None:
                return new_gt_bboxes

            for ii, img_gt_boxes in enumerate(gt_bboxes):
                for jj, img_gt_box in enumerate(img_gt_boxes):
                    match_index = torch.where((noisy_gt_boxes - img_gt_box.view(1, -1)).sum(dim=1) == 0)[0][:1]
                    m_pseudo_gt_box = pseudo_gt_boxes[match_index]
                    
                    if len(m_pseudo_gt_box) > 0:
                        new_gt_bboxes[ii][jj] = m_pseudo_gt_box
            return new_gt_bboxes

        # Object-Aware Multiple Instance Learning
        losses = dict()
        loss_oamil = 0
        tmp_oamil = 0
        total_iter_num = self.bbox_head.oa_ie_num+1 if self.bbox_head.oa_ie_flag and self._epoch+1 >= self.bbox_head.oa_ie_epoch else 1
        for iter_num in range(total_iter_num):
            
            # update gt with best selected instance 
            # when performing Object-Aware Instance Extension (OA-IE)
            if iter_num > 0:
                gt_bboxes = new_gt_bboxes

            # assign gts and sample proposals
            if self.with_bbox or self.with_mask:
                sampling_results = get_sampling_results(img_metas, proposal_list, gt_bboxes, gt_bboxes_ignore)

            bbox_results, rois, bbox_targets = self._bbox_forward_only(x, sampling_results, gt_bboxes, gt_labels, img_metas, **kwargs)

            loss_oamil, pseudo_bbox_targets, noisy_gt_boxes, pseudo_gt_boxes = self._oamil_loss(bbox_targets, bbox_results, rois, x)

            # cls and loc loss
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets, pseudo_bbox_targets=pseudo_bbox_targets,)

            if iter_num == 0:
                losses.update(loss_bbox)
                losses.update(loss_oamil)
            else:
                #losses['loss_oais'] += loss_oamil['loss_oais']
                tmp_oamil += loss_oamil['loss_oais']

            # update noisy gt for multi-stage 
            new_gt_bboxes = get_new_gt_boxes(gt_bboxes, noisy_gt_boxes, pseudo_gt_boxes)
        
        #import ipdb; ipdb.set_trace()
        #if self.bbox_head.oa_ie_flag and (self._epoch+1 >= self.bbox_head.start_epoch):
        if self.bbox_head.oa_ie_flag and (self._epoch+1 >= self.bbox_head.oa_ie_epoch):
            #losses['loss_oais'] /= (self.bbox_head.oa_ie_num+1)
            
            #losses['loss_oais'] = (losses['loss_oais'] + tmp_oamil)/ (self.bbox_head.oa_ie_num + 1)
            
            losses['loss_oais'] = losses['loss_oais'] + tmp_oamil/self.bbox_head.oa_ie_num*2.0
        
        return losses


    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
    

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        #import ipdb; ipdb.set_trace()
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, **kwargs)

        #import ipdb; ipdb.set_trace()
        # perform offset constraints
        loss_oamil, pseudo_bbox_targets, noisy_gt_boxes, pseudo_gt_boxes = self._oamil_loss(bbox_targets, bbox_results, rois, x)

        re_assign = 0
        if re_assign:
            # *** get pseudo boxes
            pos_inds = (bbox_targets[0] >= 0) & (bbox_targets[0] < self.bbox_head.num_classes)
            clean_gt_targets = bbox_targets[2][pos_inds.type(torch.bool)] + bbox_targets[-1][pos_inds.type(torch.bool)]
            pseudo_boxes_list, clean_boxes_list = self._get_pseudo_box(rois[pos_inds.type(torch.bool)], pseudo_bbox_targets, clean_gt_targets, gt_bboxes, img_metas)

            #import ipdb; ipdb.set_trace()
            bbox_targets = self.bbox_head.get_targets(sampling_results, pseudo_boxes_list,
                                                  gt_labels, self.train_cfg, **kwargs)
            pseudo_bbox_targets = None


        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        rois,
                                        *bbox_targets,
                                        pseudo_bbox_targets=pseudo_bbox_targets,
                                        )
        loss_bbox.update(loss_oamil)

        # *** Set bbox loss to zero ***
        #loss_bbox['loss_bbox'] *= 0

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results


    def _bbox_forward_only(self, x, sampling_results, gt_bboxes, gt_labels, img_metas, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        #import ipdb; ipdb.set_trace()
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, **kwargs)

        return bbox_results, rois, bbox_targets


    #def _get_pseudo_box(self, rois, pseudo_bbox_targets, clean_gt_targets, gt_bboxes, img_metas):
    def _get_pseudo_box(self, rois, pseudo_bbox_targets, gt_bboxes, img_metas):
        pseudo_boxes_list = []
        clean_boxes_list = []
        for i, (img_meta, gt_bbox) in enumerate(zip(img_metas, gt_bboxes)):
            
            # pseudo boxes
            inds = torch.where(rois[:, 0] == i)[0]
            pseudo_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][inds], pseudo_bbox_targets[inds])
            coord_sum = pseudo_boxes.sum(dim=1).int()
            inds = torch.cat([torch.where(coord_sum == inst)[0][0].view(-1) for inst in torch.unique(coord_sum)])
            pseudo_boxes_list.append(pseudo_boxes[inds].detach())

            #print(torch.unique(coord_sum))
            #import ipdb; ipdb.set_trace()

            # clean boxes
            '''inds = torch.where(rois[:, 0] == i)[0]
            clean_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][inds], clean_gt_targets[inds])
            coord_sum = clean_boxes.sum(dim=1)
            inds = torch.cat([torch.where(coord_sum == inst)[0][0].view(-1) for inst in torch.unique(coord_sum)])
            clean_boxes_list.append(clean_boxes[inds].detach())'''
        return pseudo_boxes_list, clean_boxes_list


    def _oamil_loss(self, bbox_targets, bbox_results, rois, x):
        loss_bbox = dict()

        # *** score maximum  ***
        labels, label_weights, cur_bbox_targets = bbox_targets[0], bbox_targets[1], bbox_targets[2]
        pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
        neg_inds = labels == self.bbox_head.num_classes
        pos_labels = labels[pos_inds.type(torch.bool)]

        # indices of the same gt boxes
        pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]
        noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)
        uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)

        #import ipdb; ipdb.set_trace()
        # pred OA-IS cls loss
        if self.bbox_head.oais_coef > 0:
            pred_pos_scores_list, pseudo_bbox_targets, pseudo_gt_boxes = self._get_pred_scores(bbox_results, labels, rois, x, cur_bbox_targets)
            loss_bbox['loss_oais'] = 0
            for pred_pos_scores in pred_pos_scores_list:
                pred_cls_pos_scores, pred_cls_neg_scores = self._get_instance_cls_pos_scores(pos_labels, pred_pos_scores, uniq_inst, pos_indices)
                loss_bbox['loss_oais'] += (1 - pred_cls_pos_scores) * self.bbox_head.oais_coef
            loss_bbox['loss_oais'] /= len(pred_pos_scores_list)
        else:
            pseudo_bbox_targets = None
            pseudo_gt_boxes = None

        return loss_bbox, pseudo_bbox_targets, noisy_gt_boxes, pseudo_gt_boxes

    def _get_instance_cls_pos_scores(self, pos_labels, gt_pos_scores, uniq_inst, pos_indices):
        # instance mean score
        inst_labels = []
        inst_pos_scores = []
        inst_neg_scores = []
        for inst in torch.unique(pos_indices):
            inst_inds = torch.where(pos_indices == inst)[0]
            inst_pos_scores.append(gt_pos_scores[inst_inds].max().view(-1))
            inst_neg_scores.append(gt_pos_scores[inst_inds].min().view(-1))
            inst_labels.append(pos_labels[inst_inds[0]].view(-1))

        inst_labels = torch.cat(inst_labels, dim=0)
        inst_pos_scores = torch.cat(inst_pos_scores, dim=0)
        inst_neg_scores = torch.cat(inst_neg_scores, dim=0)

        # class-wise mean score
        gt_cls_pos_scores = 0
        for cls in torch.unique(inst_labels):
            cls_inds = torch.where(inst_labels == cls)[0]
            gt_cls_pos_scores += inst_pos_scores[cls_inds].mean()
            #gt_cls_pos_scores += inst_pos_scores[cls_inds].min()
        gt_cls_pos_scores /= len(torch.unique(inst_labels))

        # class-wise mean neg score
        gt_cls_neg_scores = 0
        for cls in torch.unique(inst_labels):
            cls_inds = torch.where(inst_labels == cls)[0]
            gt_cls_neg_scores += inst_neg_scores[cls_inds].mean()
            #gt_cls_pos_scores += inst_pos_scores[cls_inds].min()
        gt_cls_neg_scores /= len(torch.unique(inst_labels))

        return gt_cls_pos_scores, gt_cls_neg_scores

    def _get_pred_scores(self, bbox_results, labels, rois, x, cur_bbox_targets):

        # get object bag info from noisy gt
        pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
        inds = torch.ones(pos_inds.sum()).cuda()
        pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]
        noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)
        uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)
        aa = [torch.where(pos_indices == inst)[0] for inst in torch.unique(pos_indices)]

        # initialize bbox result
        new_bbox_results = {}
        new_bbox_results['cls_score'], new_bbox_results['bbox_pred'] = bbox_results['cls_score'], bbox_results['bbox_pred']
        
        # keep positive preds
        new_bbox_results['bbox_pred'] = new_bbox_results['bbox_pred'].view(new_bbox_results['bbox_pred'].size(0), -1, 4)[pos_inds.type(torch.bool)]
        ori_pos_scores = torch.softmax(new_bbox_results['cls_score'], dim=1)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
        
        #import ipdb; ipdb.set_trace()
        new_pos_scores_list = []
        new_pred_boxes_list = []
        for i in range(self.bbox_head.oamil_iter_num):

            # select pred cls
            bbox_pred = new_bbox_results['bbox_pred']
            inds = torch.ones(bbox_pred.size(0)).type(torch.bool).cuda()
            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[inds, labels[pos_inds.type(torch.bool)]]

            # decode
            #if i == 0:
            if 1:
                new_pred_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_pred)
                new_roi = rois[pos_inds.type(torch.bool)].clone()
                new_roi[:,1:] = new_pred_boxes
            else:
                new_pred_boxes = self.bbox_head.bbox_coder.decode(new_roi[:, 1:], pos_bbox_pred)
                new_roi[:,1:] = new_pred_boxes

            new_pred_boxes_list.append(new_pred_boxes)

            new_bbox_results = self._bbox_forward(x, new_roi)
            new_pos_scores = torch.softmax(new_bbox_results['cls_score'], dim=1)[inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
            new_pos_scores_list.append(new_pos_scores)

        use_pseudo = self.bbox_head.use_pseudo and (self._epoch+1 >= self.bbox_head.start_epoch)
        if use_pseudo:
            # *** For Pseudo Pred GT Taregts
            pseudo_bbox_targets = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()
            pseudo_gt_boxes = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()
            for index in aa:

                # if multi-stage
                if len(new_pos_scores_list) > 1:
                    ms_box = 0
                    if ms_box:
                        cur_pred_boxes, cur_pos_scores = 0, 0
                        for pred_boxes, pred_scores in zip(new_pred_boxes_list, new_pos_scores_list):
                            _, max_inds = torch.max(pred_scores[index], dim=0)
                            cur_pred_boxes += pred_boxes[index][max_inds]
                            cur_pos_scores += pred_scores[index][max_inds]
                        
                        cur_pred_boxes /= len(new_pred_boxes_list)
                        cur_pos_scores /= len(new_pos_scores_list)
                        cur_pred_boxes = cur_pred_boxes.view(1, -1)

                    else:
                        if self.bbox_head.pseudo_type == 'cat':
                            cur_pred_boxes = torch.cat([pred_boxes[index] for pred_boxes  in new_pred_boxes_list], dim=0)
                            cur_pos_scores = torch.cat([pred_scores[index] for pred_scores  in new_pos_scores_list], dim=0)
                        elif self.bbox_head.pseudo_type == 'first':
                            cur_pred_boxes = new_pred_boxes_list[0][index]
                            cur_pos_scores = new_pos_scores_list[0][index]
                else:
                    # get score maximimum pred box
                    cur_pred_boxes = new_pred_boxes[index]
                    cur_pos_scores = new_pos_scores[index]

                # select pred with maximum score
                _, max_inds = torch.max(cur_pos_scores, dim=0)
                max_pred_score = cur_pos_scores[max_inds].clone()
                max_pred_box = cur_pred_boxes[max_inds].clone()
                max_pred_box = max_pred_box.clamp(min=0.0).view(1, -1)

                #import ipdb; ipdb.set_trace()
                if self.bbox_head.score_type == 'thrs':
                    score_thrs = self.bbox_head.score_thrs
                    # generate new pseudo gt
                    if cur_pos_scores.max() > score_thrs:
                        alpha = 0.5
                        pseudo_gt_box = max_pred_box.detach() * alpha + noisy_gt_boxes[index[0]].view(1, -1) * (1 - alpha)
                        pseudo_gt_targets = self.bbox_head.bbox_coder.encode(rois[:, 1:][pos_inds.type(torch.bool)][index], pseudo_gt_box.repeat(len(index), 1))
                        pseudo_bbox_targets[index] = pseudo_gt_targets
                    else:
                        pseudo_bbox_targets[index] = pos_bbox_targets[index]
                elif self.bbox_head.score_type == 'exp':
                    alpha = (max_pred_score.detach())**self.bbox_head.score_alpha
                    alpha = alpha.clamp(max=self.bbox_head.max_alpha)
                    
                    pseudo_gt_box = max_pred_box.detach() * alpha + noisy_gt_boxes[index[0]].view(1, -1) * (1 - alpha)
                    pseudo_gt_targets = self.bbox_head.bbox_coder.encode(rois[:, 1:][pos_inds.type(torch.bool)][index], pseudo_gt_box.repeat(len(index), 1))
                    pseudo_bbox_targets[index] = pseudo_gt_targets
                    pseudo_gt_boxes[index] = pseudo_gt_box.repeat(len(index), 1)
        else:
            pseudo_bbox_targets = None
            pseudo_gt_boxes = None

        return new_pos_scores_list, pseudo_bbox_targets, pseudo_gt_boxes
    
    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.test_cfg is None:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, None, rescale=rescale)
            det_bboxes = [boxes.cpu().numpy() for boxes in det_bboxes]
            det_labels = [labels.cpu().numpy() for labels in det_labels]
            return det_bboxes, det_labels

        else:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes[i], det_labels[i],
                            self.bbox_head.num_classes)
                for i in range(len(det_bboxes))
            ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
