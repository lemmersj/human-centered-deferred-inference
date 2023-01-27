"""Target a deferral rate and measure it success based on various thresholds.

Specifically, there are three thresholds: 1) The threshold based on the RefCOCO dataset,
2) The threshold based on the full collected dataset.
3) The threshold based on individual users.
"""
import pickle
import os
import numpy as np
import scipy.stats

# Set the target deferral rate here.
target_dr = .3

# Where the calib and split tests are divided, AFTER the burn-in
split_point = 37

# length of the burn-in
start_idx = 45

# Violation percentage.
tol = 0.05

def get_simple_threshold_all_users(skip_user, full_entropy_dict, full_images_dict, target_dr):
    """Gets the minimal absolute error threshold for a user.

    args:
        skip_user: the user that we ignore (the val split).
        full_entropy_dict: all user entropies.
        full_images_dict: the dict containing all the images for every user.
        target_dr: the target deferral rate.

    returns:
        an upper bound threshold.
    """
    # We set the threshold based on the OTHER users.
    other_user_entropies = []
    for user in full_entropy_dict:
        # don't pull data from the current user.
        if user == skip_user:
            continue
        else:
            # If it's not the current user, add all entropies that aren't
            # related an image in the current user's dataset.
            for i in range(len(full_images_dict[user])):
                if full_images_dict[user][i] in full_images_dict[skip_user]:
                    continue
                other_user_entropies.append(full_entropy_dict[user][i])

    # Sort and return the threshold.
    sorted_entropy = sorted(other_user_entropies)

    return sorted_entropy[int(len(sorted_entropy)*(1-target_dr))]

def get_ub_threshold_all_users(skip_user, full_entropy_dict, full_images_dict, target_dr):
    """Gets the upper bound threshold for all users.

    args:
        skip_user: the user that we ignore (the val split).
        full_images_dict: the dict containing all the images for every user.
        full_entropy_dict: all user entropies.
        target_dr: the target deferral rate.

    returns:
        an upper bound threshold.
    """
    # Get all entropies from not the current user
    other_user_entropies = []

    # Loop through all users
    for user in full_entropy_dict:
        # But don't use the user from our val set.
        if user == skip_user:
            continue
        else:
            # Add the entropy for any image that's not in our target user's set.
            for i in range(len(full_images_dict[user])):
                if full_images_dict[user][i] in full_images_dict[skip_user]:
                    continue
                other_user_entropies.append(full_entropy_dict[user][i])

    # Now we perform a binary search, as in the geifman+el-yaniv paper.
    sorted_entropy = sorted(other_user_entropies)
    top = len(sorted_entropy)
    bottom = 0
    # Will converge by log2 of length.
    for i in range(int(np.ceil(np.log2(len(sorted_entropy))))):
        split_idx = int(np.ceil((top+bottom)/2))
        split_entropy = sorted_entropy[split_idx]
        p_deferred = (np.array(sorted_entropy) > split_entropy).mean()
        result = binary_minimize(len(sorted_entropy), p_deferred, tol)

        if result > target_dr or result == -1:
            bottom = split_idx
        else:
            top = split_idx
            best_split = split_idx

    return sorted_entropy[best_split]

def binary_minimize(m, p_deferred, delta):
    """Performs a binary search to solve our constraint.

    args:
        m: the number of samples.
        p_deferred: the probability of a deferral given the threshold.
        delta: our acceptable error tolerance.

    returns: the upper bound deferral rate.
    """
    bottom = 0
    top = 1
    precision = 1e-7

    def inv_binom(b):
        return (-1*delta)+scipy.stats.binom.cdf(int(m*p_deferred), m, b)

    b = (bottom+top)/2
    funcval = inv_binom(b)
    while abs(funcval) > precision:
        if top == 1.0 and bottom == 1.0:
            b = -1.0
            break
        if funcval > 0:
            bottom = b
        else:
            top = b
        b = (top+bottom)/2
        funcval = inv_binom(b)
    return b

print("---")
print(f"The target Deferral Rate is: {target_dr}")

# Get all the data, and store it in a dict with users as the key
full_entropy_dict = {}
full_images_dict = {}

# Loop through all the users.
trackers = os.listdir("user_trackers")
for tracker in trackers:
    entropy_list = []
    image_list = []
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)
    counter = 0 # Used for burn-in
    for inference_step in data.inference_steps:
        # Only use initial query
        if inference_step['depth'] == 0:
            counter += 1
        else:
            continue

        # Make sure to account for burn-in
        if counter < start_idx:
            continue
        entropy = -(np.array(inference_step['belief'])*np.log(
            np.array(inference_step['belief']))).sum()
        counter += 1
        entropy_list.append(entropy)

        image_list.append(f"{inference_step['img']}")

    # Save this user's data to the dict.
    full_entropy_dict[tracker] = entropy_list
    full_images_dict[tracker] = image_list

# Now for every user, figure out how applicable the fit is.
err_full = []
err_custom = []
err_dataset = []

# valid_scores.pkl is a bit of a kludge---it's output directly from the UNITER
# model in another repository where tasks match the requirements of the paper.
with open("valid_scores.pkl", "rb") as in_pickle:
    pickle_data = pickle.load(in_pickle)
    for i in range(len(pickle_data['names'])):
        pickle_data['names'][i] = pickle_data['names'][i].split("-")[0]

# Now we calculate the results for every user.
for tracker in full_entropy_dict:
    # Start by setting the entropy list and ignoring empty pickles.
    entropy_list = full_entropy_dict[tracker]
    if len(entropy_list) == 0:
        continue

    # Set train (calibration) and test examples.
    train_entropy = entropy_list[:split_point]
    test_entropy = entropy_list[split_point:]

    # Sort
    te_argsort = np.argsort(train_entropy)
    sorted_entropy = np.array(train_entropy)[te_argsort]

    top = len(sorted_entropy)-1
    bottom = 0
    last_match = -1
    # set the upper bound threshold on only this user's train data.
    for i in range(int(np.ceil(np.log2(len(sorted_entropy))))):
        split_idx = int(np.ceil((top+bottom)/2))
        split_entropy = sorted_entropy[split_idx]
        p_deferred = (np.array(sorted_entropy) > split_entropy).mean()
        result = binary_minimize(len(sorted_entropy), p_deferred, tol)

        if result > target_dr or result == -1:
            bottom = split_idx
        else:
            top = split_idx
            best_split = split_idx
    try:
        threshold = sorted_entropy[best_split]
    except NameError:
        print("Threshold not found")
        threshold = -1

    # Now cacluate the actual deferral rate.
    dr_test = len(np.where(np.array(test_entropy) > threshold)[0])/len(test_entropy)

    # Calculate the UB threshold using all test participants.
    all_users_threshold = get_ub_threshold_all_users(
        tracker, full_entropy_dict, full_images_dict, target_dr)
    # Calculate the dataset-based threshold using RefCOCO.
    dataset_ub_threshold = get_ub_threshold_all_users(
        tracker, {'-1':pickle_data['scores'],
                  tracker: full_entropy_dict[tracker]},
        {'-1':pickle_data['names'], tracker: full_images_dict[tracker]},
        target_dr)
    # Calculate the actual deferral rates
    dr_test_full = len(np.where(np.array(test_entropy) > all_users_threshold)[0])/len(test_entropy)
    dr_test_dataset = len(
        np.where(
            np.array(test_entropy) > dataset_ub_threshold)[0])/len(test_entropy)
    err_full.append(dr_test_full - target_dr)
    err_custom.append(dr_test - target_dr)
    err_dataset.append(dr_test_dataset-target_dr)
    dr_test_full = len(
        np.where(np.array(test_entropy) > all_users_threshold)[0])/len(
            test_entropy)
    if err_custom[-1] > 0:
        print(tracker)
    continue

# Print results
print(f"UB with dataset: {np.abs(np.array(err_dataset)).mean()}, "\
      f"{(np.array(err_dataset) > 0).sum()} violations")
print(f"UB with all examples: {np.abs(np.array(err_full)).mean()}, "\
      f"{(np.array(err_full) > 0).sum()} violations")
print(f"UB with only self: {np.abs(np.array(err_custom)).mean()}, "\
      f"{(np.array(err_custom) > 0).sum()} violations")

# Now do things the simple way, attempting to minimize mean absolute error.
errors_dataset = []
errors_full = []
errors_custom = []
for tracker in full_entropy_dict:
    # Set the user's entropy list
    # remember we didn't add data prior to the burn-in
    entropy_list = full_entropy_dict[tracker]
    if len(entropy_list) == 0:
        continue
    train_entropy = entropy_list[:split_point]
    test_entropy = np.array(entropy_list[split_point:])

    # Find the split point.
    this_user_split_point = sorted(train_entropy)[int(
        len(train_entropy)*(1-target_dr))]
    # Get the threshold for users in study
    all_users_split_point = get_simple_threshold_all_users(
        tracker, full_entropy_dict, full_images_dict, target_dr)

    # Get the threshold for RefCOCO
    dataset_simple_threshold = get_simple_threshold_all_users(
        tracker, {'-1':pickle_data['scores'],
                  tracker: full_entropy_dict[tracker]},
        {'-1':pickle_data['names'], tracker: full_images_dict[tracker]},
        target_dr)

    # Calculate error
    errors_dataset.append(np.abs((
        test_entropy > dataset_simple_threshold).mean()-target_dr))
    errors_full.append(np.abs((
        test_entropy > all_users_split_point).mean()-target_dr))
    errors_custom.append(np.abs((
        test_entropy > this_user_split_point).mean()-target_dr))

# Print results
print(f"minimize with dataset: {np.array(errors_dataset).mean()}"\
      f"+-{np.array(errors_dataset).std()/np.sqrt(len(errors_dataset))}")
print(f"minimize with all examples: {np.array(errors_full).mean()}"\
      f"+-{np.array(errors_full).std()/np.sqrt(len(errors_dataset))}")
print(f"minimize with only self: {np.array(errors_custom).mean()}"\
      f"+-{np.array(errors_custom).std()/np.sqrt(len(errors_dataset))}")
