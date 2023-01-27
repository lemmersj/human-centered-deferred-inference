"""Calculates if users scores are significantly different.

typical usage:
    python analysis/rq3_users_all_comparison.py
"""
import pickle
import os
import sys
import numpy as np
from scipy.stats import kruskal, mannwhitneyu

split_point = 45
manual_trim = 75 #45+75= 120

# stack stores all entropy lists as a list of lists.
stack = []

# Loop through all users
trackers = os.listdir("user_trackers")
for tracker in trackers:
    # Save all of this user's entropies
    entropy_list = []
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)
    if len(data.inference_steps) == 0:
        continue

    # Skip until we get past the burn-in
    used_steps = 0
    for inference_step in data.inference_steps:
        # Only consider initial query
        if inference_step['depth'] > 0:
            continue
        used_steps += 1
        if used_steps < split_point:
            continue
        # Save the entropy 
        entropy_list.append(-(np.array(inference_step['belief'])*np.log(np.array(inference_step['belief']))).sum())
    stack.append(np.array(entropy_list[:manual_trim]))

# Stack and run a kruskal wallis tests
stack=np.stack(stack)
print("Probability that user has no effect on score, Kruskal-Wallis:")
print(kruskal(*stack))

# Now do individual user comparisons
compared = 0
different = 0

# Iterate through every user combination and track when significant difference.
for i in range(len(stack)):
    for j in range(i+1, len(stack)):
        cur_result = mannwhitneyu(stack[i], stack[j])
        compared += 1
        different += int(cur_result.pvalue < 0.05)

print(f"Out of {compared} user combinations, {different} were different to a statistically significant degree.")
