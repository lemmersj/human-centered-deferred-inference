"""
Locates users who are judged as outliers to be from the study.

This is defined as any user whose error is greater than three standard
deviations from the mean, and is performed iteratively. That is, after a
user is removed, the distribution is recalculated and removal occurs again.
Note that this does not perform any file operations, only tells you what to
rm.

Typical usage:
    python analysis/find_outliers.py
"""
import pickle
import os
import numpy as np

# Load all the users.
trackers = os.listdir("user_trackers")

user_accs = {}

# Iterate through every user
for tracker in trackers:

    # user_correct tracks this user's accuracy
    user_correct = []
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)

    # If there's no data, skip.
    if len(data.inference_steps) == 0:
        continue

    # Loop through every task
    for inference_step in data.inference_steps:
        # If this is a deferral response, skip
        if inference_step['depth'] > 0:
            continue
        # If it's an initial query, save whether it is correct.
        user_correct.append(inference_step['inference_correct'])
    # Then save the user's accuracy in a dict.
    user_accs[tracker] = np.array(user_correct).mean()

# Now remove outliers.

# This is the list that's printed out.
removed = []

# Loop until no outliers remain.
while True:

    # turn accuracies into a list, then calculate statistics.
    all_acc_list = [user_accs[key] for key in user_accs]
    mean_acc = np.array(all_acc_list).mean()
    std_acc = np.array(all_acc_list).std()

    # change remaining participants to list for index-based accessing. 
    all_participant_list = [key for key in user_accs]

    # find indices that meet our outlier criteria.
    to_remove = np.where(all_acc_list < (mean_acc - 3*std_acc))

    # If there are none, end.
    if len(to_remove[0]) == 0:
        break

    # Remove outliers and place the id into the removed list.
    for idx in range(len(to_remove[0])):
        removed.append(all_participant_list[to_remove[0][idx]])
        del user_accs[removed[-1]]

# Output
print(removed)
