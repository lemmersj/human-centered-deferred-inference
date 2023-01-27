"""Calculates differences between initial query and deferral response.

Additionally outputs a tsv containing corresponding initial queries and
deferral responses.

Typical usage:
    python analysis/rq4_depth_diff.py
"""
import pickle
import os
import pdb
import numpy as np
from prettytable import PrettyTable
from scipy.stats import wilcoxon, fisher_exact

from util.calculation_utils import computeIoU
from util.database_object import DatabaseObject

# Unfortunately, we need to refer back to our database to find the target object.
dbo = DatabaseObject()

# All of the errors/scores for the initial query and deferral response.
errors_0_all = []
errors_1_all = []

depth_0_all = []
depth_1_all = []

length_0_all = []
length_1_all = []

scores_significant_count = 0
lengths_significant_count = 0
errors_significant_count = 0
phrases_0 = []
phrases_1 = []

# Loop through all of the users
trackers = os.listdir("user_trackers")
user_significance_table = PrettyTable(["User","Init Score", "Def Score",
                                       "p score", "Score Sig", "Init Len",
                                       "Def Len", "p len", "Len Sig",
                                       "Init Err", "Def Err", "p err", "Err Sig"])
user_significance_table.float_format = '.4'
for tracker in trackers:
    # keep data for this user.
    scores_0 = []
    scores_1 = []
    errors_0 = []
    errors_1 = []
    length_0 = []
    length_1 = []

    similarities = []
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)
    # Do not analyze empty pickles
    if len(data.inference_steps) < 120:
        continue

    # Iterate through every step.
    next_inf_step_idx = 0
    for inference_step in data.inference_steps:
        next_inf_step_idx += 1
        # Skip the initial setting, with no deferral.
        if inference_step['rqr'] == 0:
            continue

       # cacluate entropy.
        entropy = -(np.array(inference_step['scores'])*np.log(
            np.array(inference_step['scores']))).sum()

        # Is this an initial query?
        if inference_step['depth'] == 0:
            # Make sure there's a deferral response.
            if next_inf_step_idx == len(data.inference_steps):
                continue
            if data.inference_steps[next_inf_step_idx]['depth'] == 0:
                continue
            # Save all the data.
            scores_0.append(entropy)
            depth_0_all.append(entropy)
            errors_0_all.append(inference_step['inference_correct'])
            length_0.append(len(inference_step['phrase'].split(" ")))
            length_0_all.append(len(inference_step['phrase'].split(" ")))
            errors_0.append(inference_step['inference_correct'])
            phrases_0.append(inference_step['phrase'])
        # Save data if it's a deferral response.
        elif inference_step['depth'] == 1:
            scores_1.append(entropy)
            depth_1_all.append(entropy)
            length_1.append(len(inference_step['phrase'].split(" ")))
            length_1_all.append(len(inference_step['phrase'].split(" ")))
            phrases_1.append(inference_step['phrase'])

            # We have to do this because we don't save whether the inference
            # is correct without deferral.
            dbo.cur.execute(
                f"SELECT * FROM targets WHERE image_loc LIKE '%{inference_step['img']}%'")
            results = dbo.cur.fetchall()

            # Try to find an instance in the database with the same filename and
            # target.
            match = None
            for result in results:
                if int(result['tlx']) == int(inference_step['target'][0]) and\
                   int(result['tly']) == int(inference_step['target'][1]) and\
                   int(result['brx']) == int(inference_step['target'][2]) and\
                   int(result['bry']) == int(inference_step['target'][3]):
                    match = result
                    break
            # Get the detections.
            dbo.cur.execute("SELECT id FROM sentences WHERE target=?",(match['id'],))
            results = dbo.cur.fetchall()
            dbo.cur.execute("SELECT model, detections as 'detections [detections]', model FROM outputs WHERE sentence=? AND model=3",(results[0]['id'],))
            results = dbo.cur.fetchall()
            detections = results[0]['detections']

            # And convert the detections to a guess.
            try:
                guess = detections[inference_step['scores'].argmax()].copy()
            except:
                pdb.set_trace()

            # tlx tly w h -> tlx tly brx bry
            guess[2:] = guess[2:] + guess[:2]
            # Now we can save whether it was correct or not.
            correct = (computeIoU(guess, inference_step['target']) > 0.5)
            errors_1.append(correct)
            errors_1_all.append(correct)
    # test for this user, whether score and length are significant.
    p_value_scores = wilcoxon(scores_0, scores_1).pvalue
    p_value_length = wilcoxon(length_0, length_1).pvalue


    # Test for all users whether error is significant.
    contingency_table = np.zeros((2,2))
    contingency_table[0, 0] = np.array(errors_0).sum()
    contingency_table[0, 1] = np.array(errors_1).sum()
    contingency_table[1, 0] = (1-np.array(errors_0)).sum()
    contingency_table[1, 1] = (1-np.array(errors_1)).sum()
    statistic, p_value_error = fisher_exact(contingency_table)
    user_significance_table.add_row([tracker, float(np.array(scores_0).mean()),
                                     float(np.array(scores_1).mean()),
                                     p_value_scores, p_value_scores < 0.05,
                                     float(np.array(length_0).mean()),
                                     float(np.array(length_1).mean()),
                                     p_value_length, p_value_length < 0.05,
                                     float(np.array(errors_0).mean()),
                                     float(np.array(errors_1).mean()),
                                     p_value_error, p_value_error < 0.05])

print(user_significance_table)
max_entropy = max(np.array(depth_0_all).max(),np.array(depth_1_all).max())
min_entropy = min(np.array(depth_0_all).min(),np.array(depth_1_all).min())

table_all = PrettyTable(["Init Score", "Def Score", "p score", "Score Sig",
                         "Init Len", "Def Len", "p len", "Len Sig", "Init Err",
                         "Def Err", "p err", "Err Sig"])
table_all.float_format = '.4'

pval_score = wilcoxon(depth_0_all, depth_1_all).pvalue
pval_len = wilcoxon(length_0_all, length_1_all).pvalue

# Check if, for all users, error is significant.
contingency_table = np.zeros((2,2))
contingency_table[0, 0] = np.array(errors_0_all).sum()
contingency_table[0, 1] = np.array(errors_1_all).sum()
contingency_table[1, 0] = (1-np.array(errors_0_all)).sum()
contingency_table[1, 1] = (1-np.array(errors_1_all)).sum()
statistic, p_value_error = fisher_exact(contingency_table)

table_all.add_row([float(np.array(depth_0_all).mean()), float(np.array(depth_1_all).mean()), pval_score, pval_score < 0.05, float(np.array(length_0_all).mean()), float(np.array(length_1_all).mean()), pval_len, pval_len < 0.05, 1-float(np.array(errors_0_all).mean()), 1-float(np.array(errors_1_all).mean()), p_value_error, p_value_error < 0.05]) 
print(table_all)
# Save phrases.
with open("first_second_phrases.tsv", "w") as outfile:
    for i in range(len(phrases_0)):
        outfile.write(f"{phrases_0[i]}\t{phrases_1[i]}\n")
