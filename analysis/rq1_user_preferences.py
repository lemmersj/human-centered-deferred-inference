"""Performs statistical analysis and plots user preferences.

This is done with respect to error and deferral rate.

Typical usage:
    python analysis/rq1_user_preferences.py
"""
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
from prettytable import PrettyTable

# Lists for storing data.
errors = []
satisfactions = []
drs = []
dr_satisfactions = []

# Load and iterate through all users
trackers = os.listdir("user_trackers")
for tracker in trackers:
    # 
    this_user_satisfactions = []
    this_dr_satisfactions = []
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)
    cur_survey_idx = 0

    # Skip empty files
    if len(data.inference_steps) == 0:
        continue

    # Loop through every step the user took.
    for i in range(len(data.inference_steps)):
        # Find the last inferences before the survey to save err and dr.
        # Are we at the absolute last inference step OR the 29th step for
        # a specific setting and the next step is not a deferral response?
        if i == len(data.inference_steps)-1 or\
           (data.inference_steps[i]['rqr_idx_count'] == 29 and\
            data.inference_steps[i+1]['rqr_idx_count'] == 0):
            # Make sure our survey data is for the correct condition
            if data.inference_steps[i]['rqr'] != data.surveys[cur_survey_idx]['rqr']:
                continue
            # Save the actual error, deferral rate, and satisfactions.
            errors.append(1-data.inference_steps[i]['correct_inferences']/\
                          data.inference_steps[i]['total_inferences'])
            this_user_satisfactions.append(
                data.surveys[cur_survey_idx]['acc_satisfaction'])
            drs.append(data.surveys[cur_survey_idx]['rqr'])
            this_dr_satisfactions.append(
                data.surveys[cur_survey_idx]['rqr_satisfaction'])
            cur_survey_idx += 1
            # Break if there are no more suveys.
            if cur_survey_idx == len(data.surveys):
                break
    
    # Add this user's satisfactions to the big list. This may be unnecessary.
    satisfactions += this_user_satisfactions
    dr_satisfactions += this_dr_satisfactions

# The first analysis we do is accuracy.
new_xs = []
new_ys = []
new_stds = []
counts = []
all_acc_dict = {}

# Sort the errors for placing in a histogram.
sorted_errs = sorted(errors)
hist = []
hist = [x/10. for x in range(12)]
for i in range(len(hist)-1):
    where_same = np.where((np.array(errors) >= hist[i]) * (np.array(errors) < hist[i+1]))[0]
    if len(where_same) == 0:
        continue
    counts.append(len(where_same))
    new_xs.append(np.array(errors)[where_same].mean())
    all_acc_dict[new_xs[-1]] = np.array(satisfactions)[where_same]
    new_ys.append(np.array(satisfactions)[where_same].mean())
    new_stds.append(np.array(satisfactions)[where_same].std())

# Histogram bins are compared to determine if satisfaction difference
# is significant.
pairs = []
table = PrettyTable(['Err 0','Satisfaction 0', 'Err 1', 'Satisfaction 1','p_different','Significant'])
for key in all_acc_dict:
    for key2 in all_acc_dict:
        if key == key2:
            continue
        if (key, key2) in pairs:
            continue
        pairs.append((key, key2))
        pairs.append((key2, key))
        mannwhitney_result = mannwhitneyu(all_acc_dict[key], all_acc_dict[key2]).pvalue
        table.add_row([key, np.array(all_acc_dict[key]).mean(), key2, np.array(all_acc_dict[key2]).mean(), mannwhitney_result, mannwhitney_result < 0.05])

# Print and plot
print(table.get_string(sortby="Err 0"))
fig, axs = plt.subplots(1, 2)
axs[0].set_title("I was satisfied with the accuracy I was able to achieve.", fontsize=10, style='italic')
axs[0].scatter(new_xs, new_ys, color="#1E88E5")
axs[0].errorbar(new_xs, new_ys, yerr=new_stds, color="#1E88E5")
axs[0].set_ylim(0, 8)
axs[0].set_yticks([1, 3, 5, 7])
axs[0].set_yticklabels(["Strongly Disagree", "3", "5", "Strongly Agree"])
axs[0].set_xlim(0, .62)
axs[0].set_xlabel("Error\n$\\bf{(A)}$")

# Now we repeat for the deferral rate.
dr_dict = {}
for i in range(len(drs)):
    if drs[i] not in dr_dict:
        dr_dict[drs[i]] = []
    dr_dict[drs[i]].append(dr_satisfactions[i])

to_plot_x_list = []
to_plot_y_list = []
to_plot_errbar = []
for dr in sorted(dr_dict):
    to_plot_x_list.append(dr)
    to_plot_y_list.append(np.array(dr_dict[dr]).mean())
    to_plot_errbar.append(np.array(dr_dict[dr]).std())

#print(f"DRs: {to_plot_x_list}, Satisfactions: {to_plot_y_list}, err: {to_plot_errbar}")
#pairs = []
#print("---")
#print("Deferral Rate")
table = PrettyTable(['DR 0','Dissatisfaction 0', 'DR 1', 'Dissatisfaction 1','p_different','Significant'])
for key in dr_dict:
    for key2 in dr_dict:
        if key == key2:
            continue
        if (key, key2) in pairs:
            continue
        pairs.append((key, key2))
        pairs.append((key2, key))
        mannwhitney_result = mannwhitneyu(dr_dict[key], dr_dict[key2]).pvalue
        table.add_row([key, np.array(dr_dict[key]).mean(), key2, np.array(dr_dict[key2]).mean(), mannwhitney_result, mannwhitney_result < 0.05])

# print and plot
print(table.get_string(sortby="DR 0"))
axs[1].set_title("The computer asked me to repeat myself on too many pictures.", fontsize=10, style='italic')
axs[1].scatter(to_plot_x_list, to_plot_y_list, color="#1E88E5")
axs[1].errorbar(to_plot_x_list, to_plot_y_list, yerr=to_plot_errbar, color="#1E88E5")
axs[1].set_ylim(0, 8)
axs[1].set_yticks([1, 3, 5, 7])
axs[1].set_xticks([0, 0.1, .2, .3])
axs[1].get_yaxis().set_visible(False)
axs[1].set_xlabel("Deferral Rate\n$\\bf{(B)}$")

fig.set_size_inches(11.5, 3)
fig.tight_layout()
plt.savefig("liekert_plots.pdf", bbox_inches="tight")
plt.show()
