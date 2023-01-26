"""Calculation utils used across many scripts.

Import as independent functions.
"""
import sys
import numpy as np

def computeIoU(box1, box2):
    """Computes the IoU of two boxes.

    Boxes are tuples of form tlx, tly, brx, bry.

    Args:
        box1: The first box. (tlx, tly, brx, bry)
        box2: The second box. (tlx, tly, brx, bry)

    Returns:
        The IoU
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) +\
            (box2[2]-box2[0])*(box2[3]-box2[1]) - inter

    return float(inter)/union

def compute_all_IoUs(target, candidate_boxes):
    """calculates the IoU between a set of boxes and a target.

    args:
        target: the target box (tlx, tly, brx, bry)
        candidate_boxes: a list of candidate boxes [(tlx, tly, brx, bry)]

    returns:
        list containing iou between target box and candidate boxes
    """
    iou_list = []
    for box in candidate_boxes:
        iou_list.append(computeIoU(box,target))

    return iou_list

def calc_rmae(draw):
    """Calculates the accuracy of a set of examples.

    args:
        draw: a dict containing key failure_modes

    returns:
        a floating point rmae (0->1)
    """
    return ((np.array(draw['failure_modes']) != 5)*(
        np.array(draw['failure_modes']) != 2)).mean()

def calc_error(draw):
    """Calculates the accuracy of a set of examples.

    args:
        draw: a dict containing key failure_modes

    returns:
        a floating point accuracy (0->1)
    """
    return (np.array(draw['failure_modes']) != 5).mean()

def calc_rejection_score(distribution, method):
    """Calculates the rejection score.

    args:
        distribution: the softmax distribution.
        method: the method used for calculating score from this distribution.

    returns:
        a float representing the model's prediction confidence.
    """
    # Remember higher score corresponds to higher predicted AE
    if method == "entropy":
        to_return = (-distribution*np.log(distribution+1e-32)).sum()
    elif method == "sr":
        to_return = -distribution.max()
    else:
        print("Invalid scoring method selected")
        sys.exit()
    if np.isnan(to_return.sum()) or np.isinf(to_return.sum()):
        embed()
    return to_return

def get_which_to_requery(draw, scoring_method):
    """Returns the index of the referring expression to requery.

    args:
        draw: a dict with keys probabilities
        scoring_method: the scoring function applied to the distribution

    returns:
        index of the element to re-query and a dict with the additional
        column score
    """

    # if the scores haven't been calculated, calculate them.
    if 'score' not in draw.keys():
        draw['score'] = []
        for i in range(len(draw['probabilities'])):
            draw['score'].append(
                calc_rejection_score(
                    draw['probabilities'][i], scoring_method))
    return np.array(draw['score']).argmax(), draw
