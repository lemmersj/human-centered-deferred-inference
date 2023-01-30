"""a manager for scenarios and tracking participants.

Within this manager are several different classes that I found it necessary to
develop.
"""
import csv
import random
import string
import pickle
import time
import numpy as np
from requery import QuasiRandomRequery

class UserTracker():
    """A class that tracks user-specific data.

    This includes information such as the current scenario and step
    (in case of mid-experiment crashes) and all of the information that will
    need to be done for analysis.
    """
    def __init__(self, user_id, scenarios):
        """Initialize the tracker.

        args:
            user_id: the user id.

        returns:
            none
        """
        self.user_id = user_id
        self.cur_scenario = 0
        self.inference_steps = []
        self.surveys = []
        self.scenarios = [*scenarios]
        random.shuffle(self.scenarios)

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
        self.scenario_category = scenario_category

        self.rqrs = [0.1, 0.2, 0.3]
        self.targets_per_rqr = 30
        self.current_rqr_idx = 0
        self.current_rqr_idx_count = 0
        self.correct_inferences = 0
        self.total_inferences = 0
        self.user_id = None

        # Shuffle the deferral rates and set the first setting to DR=0.0
        random.shuffle(self.rqrs)
        self.rqrs = [0.0] + self.rqrs

        # Inference times are set relative, to better anonymize.
        self.start_time = time.time()

        # Defer according to the quasi-random function described in the paper.
        self.requery_fn = QuasiRandomRequery(self.rqrs[0], self.targets_per_rqr)
        print(f"Setting RQR to {self.rqrs[0]}")

        # load scenarios
        # a scenario is a csv file consisting of filenames and target bboxes.
        # we load it in a dict so that it's grouped by images. In other words,
        # no user receives multiple targets on the same image.
        with open(f"scenarios/{self.scenario_category}.csv", "r") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row['filename'] not in self.scenario_dict.keys():
                    self.scenario_dict[row['filename']] = []
                self.scenario_dict[row['filename']].append((float(row['tlx']),
                                                      float(row['tly']),
                                                      float(row['brx']),
                                                      float(row['bry'])))

    def load_user(self, user_id):
        """Loads a user from the pickle file after a restored session.

        args:
            user_id: the user id string.

        returns:
            none
        """
        with open(f"user_trackers/{user_id}.pkl", "rb") as infile:
            data = pickle.load(infile)
        self.user_tracker = data
        self.user_id = user_id

        # If the user has done anything, load the data. If not, just set the
        # ids. I believe self.user_tracker should have state and survey info.
        try:
            self.rqd_constraint = data.inference_steps[-1]['rqd_constraint']
            self.rqrs = data.inference_steps[-1]['shuffled_rqr']
            self.current_rqr_idx = data.inference_steps[-1]['rqr_idx']
            self.current_rqr_idx_count = data.inference_steps[-1]['rqr_idx_count']+1
            self.correct_inferences = data.inference_steps[-1]['correct_inferences']
            self.total_inferences = data.inference_steps[-1]['total_inferences']
        except IndexError:
            pass

    def log(self, in_dict):
        """Logs a step in the inference process.

        args:
            in_dict: a dict containing info to log.

        returns:
            None
        """
        in_dict['rqr'] = self.rqrs[self.current_rqr_idx]
        in_dict['rqr_idx'] = self.current_rqr_idx
        in_dict['rqr_idx_count'] = self.current_rqr_idx_count
        in_dict['shuffled_rqr'] = self.rqrs
        in_dict['time'] = time.time() - self.start_time
        in_dict['correct_inferences'] = self.correct_inferences
        in_dict['total_inferences'] = self.total_inferences
        self.user_tracker.inference_steps.append(in_dict)

    def log_initial_survey(self, in_dict):
        """Logs a step in the inference process.

        args:
            in_dict: a dict containing info to log.

        returns:
            None
        """
        in_dict['time'] = time.time() - self.start_time
        self.user_tracker.initial_survey = in_dict

    def log_survey(self, in_dict):
        """Logs a step in the inference process.

        args:
            in_dict: a dict containing info to log.

        returns:
            None
        """
        in_dict['rqr'] = self.rqrs[self.current_rqr_idx-1]
        in_dict['rqr_idx'] = self.current_rqr_idx-1
        in_dict['shuffled_rqr'] = self.rqrs
        in_dict['time'] = time.time() - self.start_time
        self.user_tracker.surveys.append(in_dict)

        with open(f"user_trackers/{self.user_id}.pkl", "wb") as f:
            pickle.dump(self.user_tracker, f)

        # Reset inferences for every condition
        self.correct_inferences = 0
        self.total_inferences = 0

    def get_new_user_id(self):
        """Gets a new user id

        args:
            none

        returns:
            a new unique user id. Additionally initializes the user tracker.
        """
        # randomly create a UUID.
        self.user_id = ''.join(random.choices(
            string.ascii_letters + string.digits, k=16))

        print("Creating user tracker")

        # Instantiate this in a user_tracker object.
        self.user_tracker = UserTracker(self.user_id, self.scenario_dict)

        # Initialize some parameters in the user tracker.
        self.user_tracker.cur_scenario = 0
        self.user_tracker.cur_image = self.user_tracker.scenarios[0]
        self.user_tracker.target_bbox = random.choice(
            self.scenario_dict[self.user_tracker.cur_image])

        # Save the user tracker
        with open(f"user_trackers/{self.user_id}.pkl", "wb") as f:
            pickle.dump(self.user_tracker, f)
        return self.user_id

    def get_targets(self, user_id):
        """get the next bounding boxes and image location.

        args:
            user_id: the user id

        returns:
            tuple: str image object, tensor pick bbox, tensor place bbox
        """
        scenario = self.user_tracker.cur_scenario

        # if the session is ended, return -1 on all values
        if scenario >= len(self.scenario_dict):
            return -1, -1, -1

        # Get the target from the user tracker
        image = self.user_tracker.cur_image
        target_bbox = self.user_tracker.target_bbox

        print(f"Count: {self.current_rqr_idx_count}")

        # If we have reached the end of the treatment, update the rqr_idx and
        # rqr_idx_count (which treatment and how many tasks have been done
        # in the treatment).
        if self.current_rqr_idx_count == self.targets_per_rqr:
            self.current_rqr_idx_count = 0
            self.current_rqr_idx += 1

            # If there are no more treatments, we're done.
            if self.current_rqr_idx == len(self.rqrs):
                return "COMPLETE", "COMPLETE"

            # Otherwise, we re-initialize the deferral function and tell the
            # webapp to start a new setting.
            self.requery_fn = QuasiRandomRequery(
                self.rqrs[self.current_rqr_idx], self.targets_per_rqr)
            print(f"rqr idx is now {self.current_rqr_idx}")
            return "NEW_RQR", "NEW_RQR"

        return image, target_bbox

    def step(self, user_id):
        """Increments to the next step (task within a treatment).

        args:
            user_id: the user id

        returns:
            False if out of steps and scenarios. True otherwise.
        """
        self.user_tracker.cur_scenario += 1
        self.current_rqr_idx_count += 1

        self.user_tracker.cur_image = \
                self.user_tracker.scenarios[self.user_tracker.cur_scenario]
        self.user_tracker.target_bbox = random.choice(
            self.scenario_dict[self.user_tracker.cur_image])

        with open(f"user_trackers/{user_id}.pkl", "wb") as f:
            pickle.dump(self.user_tracker, f)

        # Making the assumption that the IDs are a contiguous list.
        if self.user_tracker.cur_scenario >= len(self.scenario_dict):
            return False
        cur_scen = self.user_tracker.cur_scenario
        bboxes = np.load(
            f"scenario_data/{self.scenario_category}/"\
            f"{self.user_tracker.scenarios[cur_scen]}.npz")['bbox']

        with open(f"user_trackers/{user_id}.pkl", "wb") as f:
            pickle.dump(self.user_tracker, f)
        return True
