"""Performs a permutation test to determine the importance of knowing the user.

Typical usage:
    python analysis/rq5_permutation_test.py
"""
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from EDGE.EDGE_4_3_1 import EDGE

# Each user gets saved in a dict
score_dict = {}
error_dict = {}

# And we also save non-user-associated data
all_error_list = []
all_score_list = []

# How many random samples we use to build our dist.
num_samples = 10000
# Burn-in period.
settle_point = 45

# Loop through all of the trackers
trackers = os.listdir("user_trackers")
for tracker in trackers:
    # Save errors and scores for this user.
    error_list = []
    score_list = []
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)

    # Skip blank files
    if len(data.inference_steps) == 0:
        continue

    # Have we reached the burn in?
    reached_settle = 0
    for inference_step in data.inference_steps:
        if inference_step['depth'] == 0:
            reached_settle += 1
        # if we haven't gotten through the burn-in, continue
        if reached_settle < settle_point:
            continue
        # I don't think this actually does anything, given the burn-in
        if inference_step['rqr'] == 0:
            continue
        # Also, select only initial queries.
        if inference_step['depth'] > 0:
            continue
        # Save the error and score for this user.
        error_list.append(1-inference_step['inference_correct'])
        score_list.append(-(np.array(inference_step['belief'])*np.log(
            np.array(inference_step['belief']))).sum())

    # Add this user data to the non-user-specified list.
    all_error_list = error_list + all_error_list
    all_score_list = score_list + all_score_list

    # Save this user data in the dictionary.
    error_dict[f'{tracker}'] = np.array(error_list)
    score_dict[f'{tracker}'] = np.array(score_list)

# Here we produce num_samples random sets of paired data.
# We can't calculate the MI here and just save that, because EDGE does some
# weird stuff with the RNG.
all_sets = []
print("Sampling")
for i in tqdm(range(num_samples)):
    all_sets.append(np.random.choice(
        [*range(len(all_score_list))], 120-settle_point))

# Now we can calculate the MI for all of our randomly chosen samples.
rand_mi_scores = []
print("Calculating MI")
for i in tqdm(range(len(all_sets))):
    x = np.array(all_score_list)[all_sets[i]]
    y = np.array(all_error_list)[all_sets[i]]
    rand_mi_scores.append(EDGE(x,y))

# And calculate the MI for individual users.
per_user_mis = []
for tracker in score_dict:
    if tracker not in score_dict:
        continue
    x = score_dict[tracker]
    y = error_dict[tracker]
    per_user_mis.append(EDGE(x, y))

# Calculate our threshold
rand_scores_sorted = sorted(rand_mi_scores)
ninetyfifth_percentile = rand_scores_sorted[len(rand_mi_scores)-len(rand_mi_scores)//100]

# Plot here to produce histogram without samples
rand_mi_scores = np.array(rand_mi_scores)
plt.hist(rand_mi_scores, 100, color="#1E88E5")
plt.xlabel("Mutual Information")
plt.axvline(ninetyfifth_percentile, color="#004D40")
ax = plt.gca()
ax.get_yaxis().set_visible(False)
plt.savefig("random_mi_distribution.pdf")

significant = 0
less_significant = 0
total = 0
for i in range(len(per_user_mis)):
    plt.axvline(per_user_mis[i], color="#D81B60")
    diff = (per_user_mis[i] > rand_mi_scores).mean()
    if diff > 0.95:
        significant += 1
    if diff > 0.9:
        less_significant += 1
    total += 1
    print(trackers[i], (per_user_mis[i] > rand_mi_scores).mean())
# And now plot here to produce histogram with lines.
plt.savefig("random_mi_distribution_withlines.pdf")
print(f"Out of {total} users, knowing the user provided additional information "\
      f"for {significant} users at significance threshold p < 0.05, and "\
      f"{less_significant} at significance threshold p < 0.1")
plt.show()
