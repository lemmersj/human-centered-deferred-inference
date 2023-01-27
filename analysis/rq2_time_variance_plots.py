"""Produces Figure 6---the relationship between task # and score, errors, length

Typical usage:
    python analysis/rq2_time_variance_plots.py
"""
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Data arrays.
all_errors = np.zeros((0,120))
all_entropies = np.zeros((0,120))
all_word_counts = np.zeros((0,120))

# Use an average filter for smoothing
avg_filter_length = 5
conv_filter = np.ones(avg_filter_length)/avg_filter_length

# Loop through every user
trackers = os.listdir("user_trackers")
for tracker in trackers:
    has_repeat = False
    step_list_error = []
    step_list_entropy = []
    word_counts = []
    # img_list is used to track and remove repeated tasks.
    # we start it off with zero so that we can access [-1].
    img_list = [0]

    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)

    for inference_step in data.inference_steps:
        # Only perform analysis on initial query
        if inference_step['depth'] > 0:
            continue
        # If there was a hiccup that caused a repeat, erase the first one.
        if img_list[-1] == inference_step['img']:
            has_repeat = True

        # Save data
        img_list.append(inference_step['img'])
        step_list_error.append(1-inference_step['inference_correct'])
        step_list_entropy.append(-(inference_step['scores']*np.log(inference_step['scores'])).sum())
        word_counts.append(len(inference_step['phrase'].split(" ")))

    # Don't save users that don't have complete data.
    if len(step_list_entropy) < 120:
        print(len(step_list_entropy))
        continue
    all_errors = np.concatenate((all_errors, np.array(step_list_error[:120]).reshape(1,-1)),axis=0)
    all_entropies = np.concatenate((all_entropies, np.array(step_list_entropy[:120]).reshape(1,-1)),axis=0)
    all_word_counts = np.concatenate((all_word_counts, np.array(word_counts[:120]).reshape(1,-1)),axis=0)

# Find statistics and plot.
fig, axs = plt.subplots(3, 1)
entropies_mean = all_entropies.mean(axis=0)
entropies_std = all_entropies.std(axis=0)/np.sqrt(all_entropies.shape[0])
x = np.arange(all_entropies.shape[1])+1
axs[0].plot(x, entropies_mean, color="#1E88E5")
axs[0].fill_between(x, entropies_mean-entropies_std, entropies_mean + entropies_std, alpha=0.5, color="#1E88E5")
axs[0].plot(np.convolve(x, conv_filter, 'valid'), np.convolve(entropies_mean, conv_filter, "valid"), color="#D81B60")
axs[0].axvline(45, ls="--")
axs[0].set_xlabel("$\\bf{(A)\ Deferral\ Score}$")
axs[0].set_xlim((0,121))
axs[0].tick_params(axis='x', which='both', length=0)
plt.setp(axs[0].get_xticklabels(), visible=False)
axs[0].set_ylabel("Deferral Score")

x = np.arange(all_errors.shape[1])+1
errors_mean = all_errors.mean(axis=0)
errors_std = all_errors.std(axis=0)/np.sqrt(all_errors.shape[0])
axs[1].plot(x, errors_mean, color="#1E88E5")
axs[1].plot(np.convolve(x, conv_filter, 'valid'), np.convolve(errors_mean, conv_filter, "valid"), color="#D81B60")
axs[1].tick_params(axis='x', which='both', length=0)
plt.setp(axs[1].get_xticklabels(), visible=False)
axs[1].axvline(45, ls="--")
axs[1].set_xlim((0,121))
axs[1].set_xlabel("$\\bf{(B)\ Errors}$")
axs[1].set_ylabel("p(error)")

x = np.arange(all_word_counts.shape[1])+1
word_counts_mean = all_word_counts.mean(axis=0)
word_counts_std = all_word_counts.std(axis=0)/np.sqrt(all_word_counts.shape[0])
axs[2].plot(x, word_counts_mean, color="#1E88E5")
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs[2].axvline(45, ls="--")
axs[2].fill_between(x, word_counts_mean-word_counts_std, word_counts_mean + word_counts_std, alpha=0.5, color="#1E88E5")
axs[2].plot(np.convolve(x, conv_filter, 'valid'), np.convolve(word_counts_mean, conv_filter, "valid"), color="#D81B60")
axs[2].set_xlabel("Task Number\n$\\bf{(C)\ Expression\ Length}$")
axs[2].set_ylabel("Words")
axs[2].set_xlim((0,121))
axs[2].set_xticks(list(axs[2].get_xticks())[:-1] + [45])

fig.set_size_inches(9, 6)
fig.tight_layout()
plt.savefig("time_plots.pdf", bbox_inches="tight")
plt.show()
