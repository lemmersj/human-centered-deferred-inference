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

dbo = DatabaseObject()

UNITER_interface = UNITERInterface()

# Want to use UNITER-GT
model = dbo.get_model_ids("UNITER", "gt")[0]

# Now we want our temp table
temptable = dbo.get_temp_table(model, dbo.distribution_to_idx("softmax"), "val")
dbo.cur.execute(f"SELECT sentence_target as target, outputs_detections as 'detections [detections]', outputs_sentence as sentence FROM {temptable}")
all_rows = dbo.cur.fetchall()


correct = 0
total = 0
for row in all_rows:
    total += 1
    dbo.cur.execute("SELECT phrase FROM sentences where id=?", (row['sentence'],))
    sentence_text = dbo.cur.fetchall()[0]['phrase']
    img_loc = dbo.get_image_loc_and_target_loc(row['target'])
    loc_string = img_loc['image_loc']
    loc_string = "_".join(loc_string.split("_")[:-1])
    loc_string_jpg = loc_string+".jpg"
    loc_string_npz = loc_string+".npz"

    guess = UNITER_interface.forward(sentence_text, loc_string)
    iou = computeIoU([img_loc['tlx'], img_loc['tly'], img_loc['brx'],
                      img_loc['bry']], guess)
    if iou > 0.5:
        correct += 1


print(f"{correct} of {total} correct. ({correct/total*100})")
