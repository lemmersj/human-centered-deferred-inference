"""Performs the significance tests to determine when mental models have settled.

typical use:
    python analysis/rq2_time_variance_tests.py"""
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import fisher_exact, mannwhitneyu

# Save data in arrays for processing
all_errors = np.zeros((0,120))
all_entropies = np.zeros((0,120))

# Iterate through all users
trackers = os.listdir("user_trackers")
for tracker in trackers:
    has_repeat = False
    step_list_error = []
    step_list_entropy = []

    # img_list is used to track and remove repeated tasks.
    # we start it off with zero so that we can access [-1].
    img_list = [0] 
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)

    # Loop through every step for this user. 
    for inference_step in data.inference_steps:
        # Only track initial queries.
        if inference_step['depth'] > 0:
            continue
        # If there was a hiccup that caused a repeat, erase the first one.
        if img_list[-1] == inference_step['img']:
            img_list = img_list[:-1]
            step_list_error = step_list_error[:-1]
            step_list_entropy = step_list_entropy[:-1]
            has_repeat = True
        
        # Save error and entropy.
        img_list.append(inference_step['img'])
        step_list_error.append(1-inference_step['inference_correct'])
        step_list_entropy.append(-(inference_step['scores']*np.log(inference_step['scores'])).sum())

    # Don't save users that don't have complete data.
    if len(step_list_entropy) < 120:
        continue

    # Save errors and entropy.
    all_errors = np.concatenate((all_errors, np.array(step_list_error).reshape(1,-1)),axis=0)
    all_entropies = np.concatenate((all_entropies, np.array(step_list_entropy).reshape(1,-1)),axis=0)

p_vals_errors_burnin = []
p_vals_errors_remain = []

p_vals_entropies_burnin = []
p_vals_entropies_remain = []
# Sort through all potential split points (i.e., burn-in points.)
for split_point in range(1,90):
    # Find these indices and split the data.
    second_point = (all_entropies.shape[1]-split_point)//2 + split_point
    burn_in_errors = np.array(all_errors[:, :split_point]).flatten()
    compare_to_errors = np.array(all_errors[:, split_point:second_point]).flatten()
    remaining_errors = np.array(all_errors[:, second_point:]).flatten()

    # build the contingency tables for error
    # starting with burn-in
    contingency_table = np.zeros((2,2))
    contingency_table[0, 0] = np.array(burn_in_errors).sum()
    contingency_table[0, 1] = np.array(compare_to_errors).sum()
    contingency_table[1, 0] = (1-burn_in_errors).sum()
    contingency_table[1, 1] = (1-compare_to_errors).sum()
    statistic, p_value = fisher_exact(contingency_table)
    p_vals_errors_burnin.append(p_value)

    # Then continuing with the divided second half
    contingency_table = np.zeros((2,2))
    contingency_table[0, 0] = np.array(compare_to_errors).sum()
    contingency_table[0, 1] = np.array(remaining_errors).sum()
    contingency_table[1, 0] = (1-compare_to_errors).sum()
    contingency_table[1, 1] = (1-remaining_errors).sum()
    statistic, p_value = fisher_exact(contingency_table)
    p_vals_errors_remain.append(p_value)

    # Do the same for entropy
    burn_in_entropies = np.array(all_entropies[:, :split_point]).flatten()

    compare_to_entropies = np.array(all_entropies[:, split_point:second_point]).flatten()
    remaining_entropies = np.array(all_entropies[:, second_point:]).flatten()
    statistic, p_value = mannwhitneyu(burn_in_entropies, compare_to_entropies)
    p_vals_entropies_burnin.append(p_value)
    statistic, p_value = mannwhitneyu(compare_to_entropies, remaining_entropies)
    p_vals_entropies_remain.append(p_value)

# Print points where the conditions are met.
print("Errors")
print(np.where((np.array(p_vals_errors_burnin) < 0.05)*(np.array(p_vals_errors_remain) > 0.05))[0])
print("Entropies")
print(np.where((np.array(p_vals_entropies_burnin) < 0.05)*(np.array(p_vals_entropies_remain) > 0.05))[0])

# Plot Figure 5
fig, axs = plt.subplots(2, 1)
axs[0].plot(p_vals_errors_burnin, label="p(burn in group = middle group)", color="#D81B60")
axs[0].plot(p_vals_errors_remain, label="p(middle group = final group)", color="#1E88E5")
axs[0].plot([0, len(p_vals_errors_burnin)],[0.05, 0.05], color="#004D40", label="Significance Threshold")
axs[0].set_ylabel("(A) Error")
axs[0].tick_params(axis='both', which='both', length=0)
axs[0].axvline(45, ls="--") 
axs[0].yaxis.set_ticks([0.0, 0.5, 1.0])
plt.setp(axs[0].get_xticklabels(), visible=False)
#plt.setp(axs[0].get_yticklabels(), visible=False)
axs[0].set_xlim((0, 90))
axs[0].legend(loc='upper left')

axs[1].plot(p_vals_entropies_burnin, label="burn in", color="#D81B60")
axs[1].plot(p_vals_entropies_remain, label="remaining", color="#1E88E5")
axs[1].plot([0, len(p_vals_errors_burnin)],[0.05, 0.05], color="#004D40", label="Significance Threshold")
axs[1].tick_params(axis='y', which='both', length=0)
axs[1].axvline(45, ls="--") 
#plt.setp(axs[1].get_yticklabels(), visible=False)
axs[1].yaxis.set_ticks([0.0, 0.5, 1.0])
axs[1].set_ylabel("(B) Deferral Scores")
axs[1].set_xlabel("Burn-In Length (# Tasks)")
axs[1].set_xlim((0, 90))
axs[1].set_xticks(list(axs[1].get_xticks()) + [45])
#axs[1].legend()
fig.set_size_inches(9, 4)
fig.tight_layout()
plt.savefig("p_different.pdf")
plt.show()
