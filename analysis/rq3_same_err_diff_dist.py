"""Produces fig 7, comparing distributions of two users with the same error.

Typical usage:
    python analysis/rq3_same_err_diff_dist.py
"""
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from scipy.stats import gaussian_kde, mannwhitneyu, kruskal


user_accs = {}
user_scores = {}
burn_in = 45

# Load and iterate through trackers
trackers = os.listdir("user_trackers")
for tracker in trackers:
    user_correct = []
    user_scores[tracker] = []
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)

    # Skip empty pickles
    if len(data.inference_steps) == 0:
        continue

    # Count allows us to skip the burn-in period
    count = 0
    for inference_step in data.inference_steps:
        if inference_step['depth'] > 0:
            continue
        count += 1
        # Skip the burn-in period.
        if count < burn_in:
            continue

        # Save deferral score and if correct.
        user_correct.append(inference_step['inference_correct'])
        user_scores[tracker].append(
            -(inference_step['scores']*np.log(inference_step['scores'])).sum())
    # Save to a dict.
    user_accs[tracker] = np.array(user_correct).mean()

# create list of unique accuracies.
accs = np.array([user_accs[tracker] for tracker in user_accs])
unique_accs = set(accs)

# Loop through all the accuracies and find users with matching accs.
accs_to_plot = []
accs_to_compare = []
for acc in unique_accs:
    count = (accs == acc).sum()
    if count >= 2:
        accs_to_compare.append(acc)
        # These conditions meet the two that get plotted.
        if acc > 0.75 and count == 2:
            accs_to_plot.append(acc)

all_kdes = []
for_min_max = []
all_scores = {}
acc_colors = ["#1E88E5", "#D81B60"]
acc_color_dict = {}
labels = []

# Create a dict key for every relevant accuracy.
for acc in accs_to_compare:
    all_scores[acc] = []

color_count = 0
colors_list = []
style_list = []
# Save the data, with some extra magic for plotting with desired colors.
for key in user_accs:
    if user_accs[key] in accs_to_compare:
        all_scores[user_accs[key]].append(user_scores[key])
    if user_accs[key] in accs_to_plot:
        if user_accs[key] not in acc_color_dict:
            acc_color_dict[user_accs[key]] = acc_colors[color_count]
            labels.append(f"{int(np.floor((1-user_accs[key])*100))}% Error")
            style_list.append("-")
            color_count += 1
        else:
            labels.append("")
            style_list.append("--")
        colors_list.append(acc_color_dict[user_accs[key]])
        all_kdes.append(gaussian_kde(user_scores[key]))
        for_min_max = for_min_max + user_scores[key]
        #labels.append(np.floor(user_accs[key]*100))

# Loop through and find significance.
table = PrettyTable(["Accuracy", "Test", "p_val", "Is significant?"])
for key in all_scores:
    if len(all_scores[key]) == 2:
        test = "Mann Whitney"
        p_val = mannwhitneyu(all_scores[key][0], all_scores[key][1]).pvalue
    else:
        test = "Kruskal-Wallis"
        p_val = kruskal(*np.stack(all_scores[key])).pvalue
    table.add_row([key, test, p_val, p_val < 0.05])

print(table.get_string(sortby="Accuracy"))

# Plot
x_vals = np.arange(0, max(for_min_max), 1/10000)
for i in range(len(all_kdes)):
    kde = all_kdes[i]
    kde_result = kde(x_vals)
    kde_result = kde_result/kde_result.sum()
    plt.plot(x_vals, kde_result, color=colors_list[i], label=labels[i], ls=style_list[i])
plt.gca().tick_params(axis='y', which='both', length=0)
plt.setp(plt.gca().get_yticklabels(), visible=False)
plt.xlabel("Deferral Score")
plt.ylabel("Density")
plt.legend()
plt.gcf().set_size_inches(9, 4)
plt.gcf().tight_layout()
plt.savefig("score_dists.pdf")

plt.show()
