"""
A straightforward object-like interface to UNITER.
"""

import json
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from model.re_sigmoid_det_dual import UniterForReferringExpressionComprehension
from utils.misc import Struct
from utils.const import IMG_DIM

class Tokenizer:
    """A class that turns text into tokens that can be passed to UNITER."""
    def __init__(self):
        """Initialize the tokenizer

        args:
            none

        returns:
            none
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.separator_token = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.class_token = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]

    def tokenize_text(self, text):
        """Convert the text string to tokens.

        Much of this was copied from the prepro.py file in the UNITER directory.

        args:
            text: the text string

        returns:
            a torch tensor of tokenized text.
        """
        ids = [self.class_token]
        # Based on some early results it seems like I need to put a lower() here. 
        for word in text.lower().strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # special character, failed encoding
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        ids.append(self.separator_token)
        return torch.tensor(ids)

class UNITERInterface():
    """A class that provides an interface to the UNITER architecture."""
    def __init__(self, scenario_category):
        """Initializes the network.

        args:
            scenario_category: tells us where to look on the filesystem

        returns:
            none
        """
        self.checkpoint_dir = "net_weights"
        self.tokenizer = Tokenizer()
        self.scenario_category = scenario_category
        self.hps_file = f'{self.checkpoint_dir}/log/hps.json'
        model_opts = json.load(open(self.hps_file))
        if 'mlp' not in model_opts:
            model_opts['mlp'] = 1
        model_opts = Struct(model_opts)
        self.model = UniterForReferringExpressionComprehension.from_pretrained(
            f'{self.checkpoint_dir}/log/model.json', torch.load(f'{self.checkpoint_dir}/ckpt/model_epoch_best_acc.pt'),
            img_dim=IMG_DIM, mlp=model_opts.mlp)
        self.model.eval()
        self.model.to("cuda")

    def forward(self, expression, npz_name, dropout=False, return_all_boxes=False, return_raw_scores=False):
        """Makes a prediction.

        args:
            expression: the expression in text form.
            target: The name that points at the npz file.
            dropout: Do we use dropout sampling to produce our guess?
            return_all_boxes: if false, return the argmax box. Otherwise return all.

        returns:
            the predicted bbox.
        """
        data = np.load(f"scenarios/{self.scenario_category}/features/{npz_name}.npz")
        batch = {}
        batch['input_ids'] = self.tokenizer.tokenize_text(expression).unsqueeze(0)
        batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).unsqueeze(0)
        batch['img_feat'] = torch.tensor(data['x']).unsqueeze(0)
        batch['txt_lens'] = [batch['input_ids'].shape[1]]
        batch['num_bbs'] = [batch['img_feat'].shape[1]]

        # Img pos feat is tlx, tly, brx, bry, w, h, area
        normalized_bboxes = torch.tensor(data['bbox']).clone()
        normalized_bboxes[:, 0] = normalized_bboxes[:, 0]/data['image_w']
        normalized_bboxes[:, 1] = normalized_bboxes[:, 1]/data['image_h']
        normalized_bboxes[:, 2] = normalized_bboxes[:, 2]/data['image_w']
        normalized_bboxes[:, 3] = normalized_bboxes[:, 3]/data['image_h']

        batch['img_pos_feat'] = torch.zeros(1, normalized_bboxes.shape[0], 7)
        batch['img_pos_feat'][0, :, :4] = normalized_bboxes
        batch['img_pos_feat'][0, :, 4] = normalized_bboxes[:, 2]-normalized_bboxes[:,0]
        batch['img_pos_feat'][0, :, 5] = normalized_bboxes[:, 3]-normalized_bboxes[:,1]
        batch['img_pos_feat'][0, :, 6] = batch['img_pos_feat'][0, :, 4]*batch['img_pos_feat'][0, :, 5]

        # Attn_masks in the original script is the largest value of num_bbs + txt_lens, where any value below that is zero. Since we're running one image at a time, this is just ones all the way across. 
        batch['attn_masks'] = torch.ones((1, batch['num_bbs'][0]+batch['txt_lens'][0]))
        batch['gather_index'] = torch.arange(batch['attn_masks'].sum()).long().unsqueeze(0)
        batch['obj_masks'] = torch.zeros((1, batch['num_bbs'][0]))
        for key in batch:
            if type(batch[key]) == torch.Tensor:
                batch[key] = batch[key].to("cuda")

        # If we do dropout, set the model to train and project the batch
        if dropout:
            self.model.train()
            batch['input_ids'] = batch['input_ids'].repeat(50,1)
            batch['position_ids'] = batch['position_ids'].repeat(50,1)
            batch['img_feat'] = batch['img_feat'].repeat(50,1,1)
            batch['txt_lens'] = batch['txt_lens']*50
            batch['num_bbs'] = batch['num_bbs']*50
            batch['img_pos_feat'] = batch['img_pos_feat'].repeat(50, 1, 1)
            batch['attn_masks'] = batch['attn_masks'].repeat(50,1)
            batch['gather_index'] = batch['gather_index'].repeat(50,1)
            batch['obj_masks'] = batch['obj_masks'].repeat(50,1)
        _, scores = self.model(batch, compute_loss=False)

        # Softmax the scores unless we have an indication otherwise.
        if not return_raw_scores:
            scores = scores.softmax(dim=1)
        scores = scores.mean(dim=0)
        selection = scores.argmax(dim=0)
        chosen_bbox = data['bbox'][selection.cpu()]

        if return_all_boxes:
            return scores, data['bbox']
        return scores, chosen_bbox
