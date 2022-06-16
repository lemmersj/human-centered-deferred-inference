"""This file is designed to evaluate performance of the webapp.

Notably, it seeks to compare accuracy when the webapp is used vs store
(database) accuracy.
"""
from util.database_object import DatabaseObject
import os
import shutil
import numpy as np
import subprocess
from IPython import embed
from uniter_interface import UNITERInterface
from util.calculation_utils import computeIoU
import torch
import pdb

def meets_criteria(counts, targets, accs):
    if counts < 5:
        return False

    if len(set(targets)) <= 3:
        return False

    if np.array(accs).mean() < 0.65:
        return False

    return True
split = "testB"

dbo = DatabaseObject()

UNITER_interface = UNITERInterface(split)

# Want to use UNITER-GT
model = dbo.get_model_ids("UNITER", "gt")[0]

# Now we want our temp table
temptable = dbo.get_temp_table(model, dbo.distribution_to_idx("softmax"), split)
dbo.cur.execute(f"SELECT sentence_target as target, outputs_detections as 'detections [detections]', outputs_sentence as sentence FROM {temptable}")
all_rows = dbo.cur.fetchall()

correct = 0
total = 0
all_raw_probs = []
all_ious = []

loc_string_accs = {}
loc_string_counts = {}
loc_string_targets = {}
# Collect the data.
for row in all_rows:
    total += 1
    dbo.cur.execute("SELECT phrase FROM sentences where id=?", (row['sentence'],))
    sentence_text = dbo.cur.fetchall()[0]['phrase']
    img_loc = dbo.get_image_loc_and_target_loc(row['target'])
    loc_string = img_loc['image_loc']
    loc_string = "_".join(loc_string.split("_")[:-1])
    loc_string_jpg = loc_string+".jpg"
    loc_string_npz = loc_string+".npz"

    with torch.no_grad():
        probs, guess = UNITER_interface.forward(sentence_text, loc_string, return_raw_scores=True, dropout=False)
        if loc_string not in loc_string_accs:
            loc_string_accs[loc_string] = []
            loc_string_targets[loc_string] = []
            loc_string_counts[loc_string] = len(probs)
        all_raw_probs.append(probs.cpu())
        these_ious = []
        data = np.load(f"../bottom-up-attention.pytorch/extracted_features/{split}/{loc_string}.npz")
        for cur_guess in range(data['bbox'].shape[0]):
            try:
                iou = computeIoU([img_loc['tlx'], img_loc['tly'], img_loc['brx'],
                              img_loc['bry']],
                                 data['bbox'][cur_guess])
            except:
                pdb.set_trace()
            these_ious.append(iou)
        loc_string_targets[loc_string].append(torch.tensor(these_ious).argmax().item())
        loc_string_accs[loc_string].append(these_ious[probs.argmax()] > 0.5)
        all_ious.append(these_ious)

for key in loc_string_accs:
    if meets_criteria(loc_string_counts[key], loc_string_targets[key], loc_string_accs[key]):
        shutil.copy(f"/z/dat/mscoco/images/train2014/{key}.jpg", f"candidate_images_testB/{key}.jpg")
pdb.set_trace()
# First, let's calculate accuracy.
correct = 0
for cur_refexp in range(len(all_ious)):
    guess = all_raw_probs[cur_refexp].argmax() 
    correct += float(all_ious[cur_refexp][guess] > 0.5)

print(f"Overall accuracy: {correct/len(all_ious)}")
pdb.set_trace()
# Now let's do the calibration error 
best_ece = 1e6
best_mce = 1e6
for scale_int in range(1, 1000):
    scale = scale_int/10.
    in_bins = [0]*10
    correct_in_bins = [0]*10
    bin_confidences = [0]*10
    for sample in range(len(all_ious)):
        probs = (all_raw_probs[sample]/scale).softmax(dim=0)
        for prob in range(len(probs)):
            if prob!=probs.argmax():
                continue
            cur_bin = int(np.floor(probs[prob].item()*10))
            cur_bin = min(cur_bin, 9)
            bin_confidences[cur_bin] = (bin_confidences[cur_bin]*in_bins[cur_bin] + probs[prob].item())/(in_bins[cur_bin]+1)
            in_bins[cur_bin] += 1
            if all_ious[sample][prob] > 0.5:
                correct_in_bins[cur_bin] += 1

    in_bins = np.array(in_bins)
    correct_in_bins = np.array(correct_in_bins)
    bin_confidences = np.array(bin_confidences)
    ECE = (in_bins/in_bins.sum() * np.abs(bin_confidences-(correct_in_bins/in_bins)))[np.where(in_bins!=0)].sum()
    MCE = np.abs(bin_confidences-(correct_in_bins/in_bins))[np.where(in_bins!=0)].max()
    if MCE < best_mce:
        best_mce = MCE
        best_mce_idx = scale_int
        best_mce_confidences = bin_confidences
        best_mce_counts = in_bins
        best_mce_correct = correct_in_bins
    if ECE < best_ece:
        best_ece = ECE
        best_ece_idx = scale_int
        best_ece_dist = probs
        best_ece_confidences = bin_confidences
        best_ece_counts = in_bins
        best_ece_correct = correct_in_bins

    print(f"{scale}: ECE: {ECE}, MCE: {MCE}")
import pdb
pdb.set_trace()
print(f"{correct} of {total} correct. ({correct/total*100})")
