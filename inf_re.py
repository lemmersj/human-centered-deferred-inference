"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference of VQA for submission
"""
import argparse
import json
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader

from apex import amp
from cytoolz import concat

from data import (PrefetchLoader, DetectFeatLmdb, ReTxtTokLmdb,
                  ReEvalDataset, re_eval_collate)
from data.sampler import DistributedSampler
from model.re import UniterForReferringExpressionComprehension

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import IMG_DIM
import pickle
from IPython import embed

def calc_iou(bbox_1, bbox_2):
    max_left_x = max(bbox_1[0], bbox_2[0])
    min_right_x = min(bbox_1[0]+bbox_1[2], bbox_2[0]+bbox_2[2])
    max_top_y = max(bbox_1[1], bbox_2[1])
    min_bottom_y = min(bbox_1[1]+bbox_1[3], bbox_2[1]+bbox_2[3])

    if min_right_x < max_left_x or min_bottom_y < max_top_y:
        intersection = 0
    else:
        intersection = (min_right_x - max_left_x) * (min_bottom_y-max_top_y)

    union = bbox_1[2]*bbox_1[3] + bbox_2[2]*bbox_2[3] - intersection

    return intersection/union

def write_to_tmp(txt, tmp_file):
    if tmp_file:
        f = open(tmp_file, "a")
        f.write(txt)

def best_bbox_iou(candidates, target):
    best_iou = 0
    for box_idx in range(candidates.shape[0]):
        cur_box = candidates[box_idx, :]
        iou = calc_iou(cur_box, target)
        if iou > best_iou:
            best_iou = iou

    return best_iou

def main(opts):
    n_gpu = 1
    device = "cuda"
    rank = 0

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = json.load(open(hps_file))
    if 'mlp' not in model_opts:
        model_opts['mlp'] = 1
    model_opts = Struct(model_opts)
    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_epoch_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    model = UniterForReferringExpressionComprehension.from_pretrained(
        f'{opts.output_dir}/log/model.json', checkpoint,
        img_dim=IMG_DIM, mlp=model_opts.mlp)
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')

    # load DBs and image dirs
    img_db_type = "gt" if "coco_gt" in opts.img_db else "det"
    conf_th = -1 if img_db_type == "gt" else model_opts.conf_th
    num_bb = 100 if img_db_type == "gt" else model_opts.num_bb
    eval_img_db = DetectFeatLmdb(opts.img_db,
                                 conf_th, model_opts.max_bb,
                                 model_opts.min_bb, num_bb,
                                 opts.compressed_db)

    # Prepro txt_dbs
    txt_dbs = opts.txt_db.split(':')
    for txt_db in txt_dbs:
        print(f'Evaluating {txt_db}')
        eval_txt_db = ReTxtTokLmdb(txt_db, -1)
        eval_dataset = ReEvalDataset(
            eval_txt_db, eval_img_db, use_gt_feat=img_db_type == "gt")

        sampler = DistributedSampler(eval_dataset, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=sampler,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=re_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader)

        # evaluate
        val_log, results = evaluate(model, eval_dataloader)

        result_dir = f'{opts.output_dir}/results_test'
        if not exists(result_dir) and rank == 0:
            os.makedirs(result_dir)
        write_to_tmp(
            f"{txt_db.split('_')[1].split('.')[0]}-acc({img_db_type}): {results['acc']*100:.2f}% ",
            args.tmp_file)

        all_results = list(concat(results))

        if rank == 0:
            db_split = txt_db.split('/')[-1].split('.')[0]  # refcoco+_val
            img_dir = opts.img_db.split('/')[-1]  # re_coco_gt
            with open(f'{result_dir}/'
                    f'results_{opts.checkpoint}_{db_split}_on_{img_dir}_all.json', 'w') as f:
                json.dump(all_results, f)
        # print
        print(f'{opts.output_dir}/results_test')

    write_to_tmp(f'\n', args.tmp_file)


@torch.no_grad()
def evaluate(model, eval_loader):
    LOGGER.info("start running evaluation...")
    model.eval()
    rank = 0
    tot_score = 0
    n_ex = 0
    st = time()
    predictions = []
    for i, batch in enumerate(eval_loader):
        (tgt_box_list, obj_boxes_list, sent_ids) = (
            batch['tgt_box'], batch['obj_boxes'], batch['sent_ids'])
        # scores (n, max_num_bb)
        scores = model(batch, compute_loss=False)
        ixs = torch.argmax(scores, 1).cpu().detach().numpy()  # (n, )

        # ixs = predicted bboxes from frcnn (batch_size)
        # scores = output probabilities corresponding to pred bboxes
        # (batch_size x 31)
        # tgt_box_list = list of tgt bboxes (list of batch_size, containing
        # 4 dim array)
        batch_idx = 0
        for ix, obj_boxes, tgt_box, sent_id in \
                zip(ixs, obj_boxes_list, tgt_box_list, sent_ids):
            pred_box = obj_boxes[ix]
            predictions.append({'sent_id': int(sent_id),
                                'pred_box': pred_box.tolist(),
                                'tgt_box': tgt_box.tolist()})
            if eval_loader.loader.dataset.computeIoU(pred_box, tgt_box) > .5:
                tot_score += 1
            n_ex += 1
            pickle_dict = {}
            pickle_dict['detected_bboxes'] = obj_boxes
            pickle_dict['target_bbox'] = tgt_box
            pickle_dict['sent_id'] = sent_id
            pickle_dict['output_prob'] = scores[batch_idx, :]
            pickle_dict['best_iou'] = best_bbox_iou(obj_boxes, tgt_box)
            model_last_sequence = model.last_sequence_output[batch_idx, :, :]

            batch_idx += 1
            if args.split_dir:
                with open(f'{os.path.join(args.output_dir, args.split_dir, sent_id)}.pickle', 'wb') as pickle_file:
                    pickle.dump(pickle_dict, pickle_file)
        if i % 100 == 0 and rank == 0:
            n_results = len(predictions)
            LOGGER.info(f'{n_results}/{len(eval_loader.dataset)} '
                        'answers predicted')
    n_ex = n_ex
    tot_time = time()-st
    tot_score = tot_score
    val_acc = tot_score / n_ex
    val_log = {'valid/acc': val_acc, 'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation ({n_ex} sents) finished in"
                f" {int(tot_time)} seconds"
                f", accuracy: {val_acc*100:.2f}%")
    # summarizae
    results = {'acc': val_acc, 'predictions': predictions}
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="can be the path to binary or int number (step)")
    parser.add_argument("--batch_size",
                        default=256, type=int,
                        help="number of sentences per batch")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory of the training command")

    # device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    parser.add_argument('--split_dir', type=str, default=None,
                        help="write results to tmp file")
    # Write simple results to some tmp file
    parser.add_argument('--tmp_file', type=str, default=None,
                        help="write results to tmp file")

    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_dir, args.split_dir), exist_ok=True)

    main(args)
