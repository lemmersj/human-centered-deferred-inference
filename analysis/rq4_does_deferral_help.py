"""
Does accuracy calculations with deferral AND aggregation function.

Typical usage:
    python analysis/rq4_does_deferral_help.py
"""
import pickle
import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Enable/disable burnin
start_point = 45
#start_point = 0

# Track error before and after deferral.
error_0 = []
error_1 = []

# Loop through all users
trackers = os.listdir("user_trackers")
for tracker in trackers:
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)

    # Skip empty files
    if len(data.inference_steps) == 0:
        continue

    step_count = 0 # Count for burn-in
    # Loop through all steps.
    for i in range(len(data.inference_steps)):
        # Do burn-in
        if data.inference_steps[i]['depth'] == 0:
            # only update at depth 0
            step_count += 1
        if step_count < start_point:
            continue
        # If we're at the initial query, continue
        if data.inference_steps[i]['depth'] == 0:
            step_count += 1
        # if we're at the deferral response, add both.
        if data.inference_steps[i]['depth'] == 1:
            error_0.append(data.inference_steps[i-1]['inference_correct'])
            error_1.append(data.inference_steps[i]['inference_correct'])

# statology.org/mcnemars-test-python
error_0 = np.array(error_0)
error_1 = np.array(error_1)
correct_to_correct = (error_0 * error_1).sum()
correct_to_incorrect = (error_0 * (1-error_1)).sum()
incorrect_to_correct = ((1-error_0) * error_1).sum()
incorrect_to_incorrect = ((1-error_0) * (1-error_1)).sum()

contingency_table = np.stack([[correct_to_correct, incorrect_to_correct],
                              [correct_to_incorrect, incorrect_to_incorrect]])

print(f"The error before deferral was {1-error_0.mean()}, and after deferral "\
      f"it was {1-error_1.mean()}")
print(f"{correct_to_correct + correct_to_incorrect} tasks were correct after "\
      f"the first query, while {correct_to_correct} of those remained"\
      f"correct after the deferral response "\
      f"({correct_to_correct/(correct_to_correct+correct_to_incorrect)})")
print(f"{incorrect_to_correct + incorrect_to_incorrect} tasks were incorrect"\
      f"after the first query, while {incorrect_to_correct} of those were "\
      f"corrected by the deferral response "\
      f"({incorrect_to_correct/(incorrect_to_correct+incorrect_to_incorrect)})")
print(f"The probability that deferral had an effect was "\
      f"{mcnemar(contingency_table, exact=True).pvalue}")
