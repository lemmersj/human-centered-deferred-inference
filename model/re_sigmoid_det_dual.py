"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for RE model
"""
from collections import defaultdict

import torch
from torch import nn
import random
import numpy as np
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel

def computeIoU(all_boxes, target_boxes, num_boxes):
    # all_boxes comes in as list of numpy array (for some reason...)
    all_boxes_tensor = torch.zeros(len(all_boxes), num_boxes, all_boxes[0].shape[1])
    for i in range(len(all_boxes)):
        all_boxes_tensor[i, :all_boxes[i].shape[0], :] = torch.tensor(all_boxes[i])

    all_targets_tensor = torch.zeros(len(target_boxes), 4)
    for i in range(len(target_boxes)):
        all_targets_tensor[i, :] = torch.tensor(target_boxes[i])

    all_targets_tensor = all_targets_tensor.unsqueeze(1).repeat(
        1, all_boxes_tensor.shape[1], 1)

    inter_x1 = (all_targets_tensor[:, :, 0] >= all_boxes_tensor[:, :, 0]).float()*all_targets_tensor[:, :, 0] + (all_targets_tensor[:, :, 0] < all_boxes_tensor[:, :, 0]).float()*all_boxes_tensor[:, :, 0]
    inter_y1 = (all_targets_tensor[:, :, 1] >= all_boxes_tensor[:, :, 1]).float()*all_targets_tensor[:, :, 1] + (all_targets_tensor[:, :, 1] < all_boxes_tensor[:, :, 1]).float()*all_boxes_tensor[:, :, 1]

    br_corner_targets = all_targets_tensor[:, :, :2] + all_targets_tensor[:, :, 2:4]
    br_corner_detections = all_boxes_tensor[:, :, :2] + all_boxes_tensor[:, :, 2:4]

    inter_x2 = (br_corner_targets[:, :, 0] <= br_corner_detections[:, :, 0]).float()*br_corner_targets[:, :, 0] + (br_corner_targets[:, :, 0] > br_corner_detections[:, :, 0]).float()*br_corner_detections[:, :, 0]
    inter_y2 = (br_corner_targets[:, :, 1] <= br_corner_detections[:, :, 1]).float()*br_corner_targets[:, :, 1] + (br_corner_targets[:, :, 1] > br_corner_detections[:, :, 1]).float()*br_corner_detections[:, :, 1]
   
    intersection = ((inter_x2 - inter_x1)*(inter_y2 - inter_y1))*(inter_x2 - inter_x1 > 0).float() * (inter_y2 - inter_y1 > 0).float()
    union = all_targets_tensor[:, :, 2] * all_targets_tensor[:, :, 3] + all_boxes_tensor[:, :, 2] * all_boxes_tensor[:, :, 3] - intersection

    return intersection/union

class UniterForReferringExpressionComprehension(UniterPreTrainedModel):
    """ Finetune UNITER for RE
    """
    def __init__(self, config, img_dim, loss="cls",
                 margin=0.2, gamma=0.0, hard_ratio=0.3, mlp=1):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        if mlp == 1:
            self.re_output = nn.Linear(config.hidden_size, 2)
        elif mlp == 2:
            self.re_output = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                GELU(),
                LayerNorm(config.hidden_size, eps=1e-12),
                nn.Linear(config.hidden_size, 2)
            )
        else:
            raise ValueError("MLP restricted to be 1 or 2 layers.")
        self.loss = loss
        assert self.loss in ['cls', 'focal', 'cls_all', 'focal_all']
        self.gamma = gamma
        if self.loss == 'rank':
            self.margin = margin
            self.hard_ratio = hard_ratio
        else:
            self.crit = nn.CrossEntropyLoss(reduction='none')

        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True, mask=False, return_separate=False, loss_type="both", bce_scale=1):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        obj_masks = batch['obj_masks']
        txt_lens = batch["txt_lens"]
        num_bbs = batch["num_bbs"]
        targets = batch["tgt_box"]
        det_boxes = batch["obj_boxes"]
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)

        # get only the region part
        sequence_output = self._get_image_hidden(
            sequence_output, txt_lens, num_bbs)

        # Save last sequence output for pickling later.
        self.last_sequence_output = sequence_output.clone().detach().cpu()

        # re score (n, max_num_bb)
        task_output = self.re_output(sequence_output).squeeze(2)
        scores_for_sigmoid = task_output[:,:, 0]
        scores_for_softmax = task_output[:,:, 1]
        scores_for_sigmoid = scores_for_sigmoid.masked_fill(obj_masks.bool(), -1e4)  # mask out non-objects
        scores_for_softmax = scores_for_softmax.masked_fill(obj_masks.bool(), -1e4)  # mask out non-objects

        if compute_loss:
            sigmoid_scores = torch.sigmoid(scores_for_sigmoid)
            if batch['tgt_box'] == None:
                ious = torch.zeros(sigmoid_scores.shape).to(sigmoid_scores.device)
                ious[torch.arange(sigmoid_scores.shape[0]), batch['targets'].squeeze()] = 1
                no_target_mask = 1-(batch['targets'] == -1).float().repeat(1, ious.shape[1])
                ious = no_target_mask * ious
            else:
                ious = computeIoU(det_boxes, targets, scores_for_sigmoid.shape[1]).to(device=scores_for_sigmoid.device)
            sigmoid_targets = (ious > 0.5).float()

            loss_mask = (sigmoid_scores > -1e4).to(dtype=sigmoid_scores.dtype)
            
            bce_loss = nn.functional.binary_cross_entropy(
                sigmoid_scores, sigmoid_targets, reduction="none")*loss_mask
            
            # Softmax target does not always exist
            softmax_targets = ious.argmax(dim=1)
            # mask for when the softmax target DNE.
            softmax_target_exists = (sigmoid_targets.sum(dim=1) > 0).float()
            
            ce_loss = self.crit(scores_for_softmax, softmax_targets) * softmax_target_exists

            if return_separate:
                return bce_loss.sum(dim=1), ce_loss
            else:
                if loss_type == "cx":
                    return ce_loss
                elif loss_type == "bce":
                    return bce_loss.sum(dim=1)
                elif loss_type == "both":
                    return bce_loss.sum(dim=1)*bce_scale+ce_loss
                    #return ce_loss
        else:
            # (n, max_num_bb)
            return scores_for_sigmoid, scores_for_softmax

    def sample_neg_ix(self, scores, targets, num_bbs):
        """
        Inputs:
        :scores    (n, max_num_bb)
        :targets   (n, )
        :num_bbs   list of [num_bb]
        return:
        :neg_ix    (n, ) easy/hard negative (!= target)
        """
        neg_ix = []
        cand_ixs = torch.argsort(
            scores, dim=-1, descending=True)  # (n, num_bb)
        for i in range(len(num_bbs)):
            num_bb = num_bbs[i]
            if np.random.uniform(0, 1, 1) < self.hard_ratio:
                # sample hard negative, w/ highest score
                for ix in cand_ixs[i].tolist():
                    if ix != targets[i]:
                        assert ix < num_bb, f'ix={ix}, num_bb={num_bb}'
                        neg_ix.append(ix)
                        break
            else:
                # sample easy negative, i.e., random one
                ix = random.randint(0, num_bb-1)  # [0, num_bb-1]
                while ix == targets[i]:
                    ix = random.randint(0, num_bb-1)
                neg_ix.append(ix)
        neg_ix = torch.tensor(neg_ix).type(targets.type())
        assert neg_ix.numel() == targets.numel()
        return neg_ix

    def _get_image_hidden(self, sequence_output, txt_lens, num_bbs):
        """
        Extracting the img_hidden part from sequence_output.
        Inputs:
        - sequence_output: (n, txt_len+num_bb, hid_size)
        - txt_lens       : [txt_len]
        - num_bbs        : [num_bb]
        Output:
        - img_hidden     : (n, max_num_bb, hid_size)
        """
        outputs = []
        max_bb = max(num_bbs)
        hid_size = sequence_output.size(-1)
        for seq_out, len_, nbb in zip(sequence_output.split(1, dim=0),
                                      txt_lens, num_bbs):
            img_hid = seq_out[:, len_:len_+nbb, :]
            if nbb < max_bb:
                img_hid = torch.cat(
                        [img_hid, self._get_pad(
                            img_hid, max_bb-nbb, hid_size)],
                        dim=1)
            outputs.append(img_hid)

        img_hidden = torch.cat(outputs, dim=0)
        return img_hidden

    def _get_pad(self, t, len_, hidden_size):
        pad = torch.zeros(1, len_, hidden_size, dtype=t.dtype, device=t.device)
        return pad
