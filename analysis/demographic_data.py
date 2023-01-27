"""Calculates demographic data across test participants.

Typical usage:
    python analysis/demongraphic_data.py
"""
import pickle
import os
import numpy as np

# Lists for storing participant data.
ages = []
genders = []
tech_competence = []
cva_competence = []

# Load and iterate through all the users.
trackers = os.listdir("user_trackers")
for tracker in trackers:
    # Save the data for every user
    with open(f"user_trackers/{tracker}","rb") as in_pickle:
        data = pickle.load(in_pickle)
        try:
            ages.append(int(data.initial_survey['age']))
            genders.append(data.initial_survey['gender'])
            tech_competence.append(int(data.initial_survey['tech_competence']))
            cva_competence.append(int(data.initial_survey['cva_competence']))
        except Exception as e:
            continue

# Print results.
print(f"{len(ages)} users\n mean age {np.array(ages).mean()}+-{np.array(ages).std()} \n, tech competence: {np.array(tech_competence).mean()}+-{np.array(tech_competence).std()}\ncva competence: {np.array(cva_competence).mean()}+-{np.array(cva_competence).std()})")
genders = np.array(genders)
print(f"{(genders == 'male').sum()} male\n{(genders == 'female').sum()}\nfemale, {(genders == 'nb').sum()} nb/other\n{(genders == 'np').sum()} prefer not to state.")
