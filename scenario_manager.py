"""a manager for scenarios and tracking participants.

Within this manager are several different classes that I found it necessary to
develop.
"""
import os
import csv
import random
import pickle
import numpy as np
from IPython import embed

class InferenceStep():
    """A convenience class to hold data targeting a specific pick and place"""
    def __init__(self, scenario, step, bboxes, pick_target, place_target):
        """Initializes the class
        
        args:
            scenario: the scenario id
            step: the step
            bboxes: all bboxes in the image
            pick_target: the bbox for the pick object
            place_target: the bbox for the place target

        returns: 
            nothing
        """
        self.scenario = scenario
        self.step = step
        self.bboxes = bboxes
        self.pick_target = pick_target
        self.place_target = place_target

        self.unparsed_strings = []

        self.pick_strs = []
        self.pick_indiv_guesses = []
        self.pick_aggregated_guesses = []

        self.place_strs = []
        self.place_indiv_guesses = []
        self.place_aggregated_guesses = []


class UserTracker():
    """A class that tracks user-specific data.
    
    This includes information such as the current scenario and step
    (in case of mid-experiment crashes) and all of the information that will
    need to be done for analysis.
    """
    def __init__(self, user_id):
        """Initialize the tracker.

        args:
            user_id: the user id.

        returns:
            none
        """
        self.user_id = user_id
        self.cur_scenario = 0
        self.cur_step = 0
        self.inference_steps = []

class ScenarioManager():
    """a class for managing scenarios."""
    def __init__(self, scenario_category):
        """Initializes the scenario manager

        args:
            scenario_category: string used to find data in filesystem

        returns:
            None
        """
        self.scenario_dict = {}
        self.image_dict = {}
        self.user_trackers = {}
        self.scenario_category = scenario_category

        # load existing user sessions
        all_trackers = os.listdir('user_trackers')
        for tracker in all_trackers:
            with open(f"user_trackers/{tracker}", "rb") as infile:
                cur_tracker = pickle.load(infile)
                self.user_trackers[cur_tracker.user_id] = cur_tracker
        # load scenarios
        with open(f"scenarios/{self.scenario_category}/scenarios.csv", "r") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                self.scenario_dict[int(row['id'])] = []
                self.image_dict[int(row['id'])] = row['image']

        for key in self.scenario_dict.keys():
            with open(f"scenarios/{scenario_category}/{key}.csv") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    for key2 in row:
                        try:
                            row[key2] = float(row[key2])
                        except:
                            continue
                    self.scenario_dict[key].append(row)

    def get_new_user_id(self):
        """Gets a new user id

        args:
            none
        
        returns:
            a new unique user id 
        """
        new_user_id = 0

        if len(self.user_trackers.keys()) > 0:
            new_user_id = max(self.user_trackers.keys())+1

        self.user_trackers[new_user_id] = UserTracker(new_user_id)
        cur_idx = 0
        while len(self.scenario_dict[cur_idx]) == 0:
            cur_idx += 1
            continue
        bboxes = np.load(f"../bottom-up-attention.pytorch/extracted_features/{self.scenario_category}/{self.image_dict[cur_idx]}.npz")['bbox']
        pick_target = [self.scenario_dict[cur_idx][0]['pick_tlx'],
                       self.scenario_dict[cur_idx][0]['pick_tly'],
                       self.scenario_dict[cur_idx][0]['pick_brx'],
                       self.scenario_dict[cur_idx][0]['pick_bry']]
        place_target = [self.scenario_dict[cur_idx][0]['place_tlx'],
                       self.scenario_dict[cur_idx][0]['place_tly'],
                       self.scenario_dict[cur_idx][0]['place_brx'],
                       self.scenario_dict[cur_idx][0]['place_bry']]

        new_inference_step = InferenceStep(cur_idx,
                                           0,
                                           bboxes,
                                           pick_target,
                                           place_target)

        self.user_trackers[new_user_id].inference_steps.append(new_inference_step)
        self.user_trackers[new_user_id].cur_scenario = cur_idx

        return new_user_id

    def get_targets(self, user_id):
        """get the next bounding boxes and image location.
        
        args:
            user_id: the user id

        returns:
            tuple: str image object, tensor pick bbox, tensor place bbox
        """
        scenario = self.user_trackers[user_id].cur_scenario
        step = self.user_trackers[user_id].cur_step

        # if the session is ended, return -1 on all values
        if scenario >= len(self.scenario_dict):
            return -1, -1, -1

        image = self.image_dict[scenario]
        bbox_unformatted = self.scenario_dict[scenario][step]

        pick_bbox = [bbox_unformatted['pick_tlx'],
                     bbox_unformatted['pick_tly'],
                     bbox_unformatted['pick_brx'],
                     bbox_unformatted['pick_bry']]

        place_bbox = [bbox_unformatted['place_tlx'],
                     bbox_unformatted['place_tly'],
                     bbox_unformatted['place_brx'],
                     bbox_unformatted['place_bry']]
        return image, pick_bbox, place_bbox

    def add_inference(self, user_id, unparsed_string, pick_string, place_string, pick_individual, pick_aggregate, place_individual, place_aggregate):
        """Adds inference data to the InferenceStep

        args:
            user_id: the user ID
            unparsed_string: the unparsed string input
            pick_string: the processed pick string
            place_string: the processed place string
            pick_individual: the distribution for this instance of the pick object
            pick_aggregate: the aggregate pick distribution
            place_individual: this place distribution
            place_aggregate: the aggregate place distribution

        returns:
            None
        """
        self.user_trackers[user_id].inference_steps[-1].unparsed_strings.append(unparsed_string)
        self.user_trackers[user_id].inference_steps[-1].pick_strs.append(pick_string)
        self.user_trackers[user_id].inference_steps[-1].place_strs.append(place_string)
        self.user_trackers[user_id].inference_steps[-1].pick_indiv_guesses.append(pick_individual)
        self.user_trackers[user_id].inference_steps[-1].pick_aggregated_guesses.append(pick_aggregate)
        self.user_trackers[user_id].inference_steps[-1].place_indiv_guesses.append(place_individual)
        self.user_trackers[user_id].inference_steps[-1].place_aggregated_guesses.append(place_aggregate)


    def step(self, user_id):
        """Increments to the next step.

        args:
            user_id: the user id

        returns:
            False if out of steps and scenarios. True otherwise.
        """
        self.user_trackers[user_id].cur_step += 1
        if self.user_trackers[user_id].cur_step >= len(self.scenario_dict[self.user_trackers[user_id].cur_scenario]):
            self.user_trackers[user_id].cur_step = 0
            self.user_trackers[user_id].cur_scenario += 1

        with open(f"user_trackers/{user_id}.pkl", "wb") as f:
            pickle.dump(self.user_trackers[user_id], f)

        # Making the assumption that the IDs are a contiguous list.
        if self.user_trackers[user_id].cur_scenario >= len(self.scenario_dict):
            return False
        cur_scen = self.user_trackers[user_id].cur_scenario
        cur_step = self.user_trackers[user_id].cur_step
        bboxes = np.load(f"../bottom-up-attention.pytorch/extracted_features/{self.scenario_category}/{self.image_dict[cur_scen]}.npz")['bbox']
        pick_target = [self.scenario_dict[cur_scen][cur_step]['pick_tlx'],
                       self.scenario_dict[cur_scen][cur_step]['pick_tly'],
                       self.scenario_dict[cur_scen][cur_step]['pick_brx'],
                       self.scenario_dict[cur_scen][cur_step]['pick_bry']]
        place_target = [self.scenario_dict[cur_scen][cur_step]['place_tlx'],
                       self.scenario_dict[cur_scen][cur_step]['place_tly'],
                       self.scenario_dict[cur_scen][cur_step]['place_brx'],
                       self.scenario_dict[cur_scen][cur_step]['place_bry']]

        new_inference_step = InferenceStep(cur_scen,
                                           cur_step,
                                           bboxes,
                                           pick_target,
                                           place_target)

        self.user_trackers[user_id].inference_steps.append(new_inference_step)
        with open(f"user_trackers/{user_id}.pkl", "wb") as f:
            pickle.dump(self.user_trackers[user_id], f)
        return True
