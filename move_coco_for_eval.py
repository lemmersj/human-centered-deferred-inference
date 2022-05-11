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

dbo = DatabaseObject()

# Get a file list to check if already copied.
files_that_exist = os.listdir(
    "../bottom-up-attention.pytorch/images")

# Want to use UNITER-GT
model = dbo.get_model_ids("UNITER", "gt")[0]

# Now we want our temp table
temptable = dbo.get_temp_table(model, dbo.distribution_to_idx("softmax"), "val")
dbo.cur.execute(f"SELECT sentence_target as target, outputs_detections as 'detections [detections]'  FROM {temptable}")
all_rows = dbo.cur.fetchall()

for row in all_rows:
    img_loc = dbo.get_image_loc_and_target_loc(row['target'])
    loc_string = img_loc['image_loc']
    loc_string = "_".join(loc_string.split("_")[:-1])
    loc_string_jpg = loc_string+".jpg"
    loc_string_npz = loc_string+".npz"
    if loc_string_jpg in files_that_exist:
        print(f"File {loc_string} exists, skipping")
        continue
    shutil.copyfile(f"/z/dat/mscoco/images/train2014/{loc_string_jpg}",
                    f"../bottom-up-attention.pytorch/images/{loc_string_jpg}")
    formatted_detections = np.zeros(row['detections'].shape)
    formatted_detections[:, :2] = row['detections'][:, :2]
    formatted_detections[:, 2] = row['detections'][:, 2] + row['detections'][:, 0]
    formatted_detections[:, 3] = row['detections'][:, 3] + row['detections'][:, 1]
    
    np.savez(f"../bottom-up-attention.pytorch/boxes/{loc_string_npz}", bbox=formatted_detections)

