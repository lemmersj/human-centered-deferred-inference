"""Re-Query classes"""
import random
import torch
import numpy as np

class RandomRequery():
    def __init__(self, threshold):
        """The initializer.

        args:
            threshold: threshold for re-query. Since we're using uniform random, this is also the probability that the query is accepted.
        """
        self.threshold = threshold

    def should_requery(self, distribution):
        """Should we re-query?

        args:
            distribution: the output distribution. Unused, but required for inheritance.

        returns:
            boolean. True for re-query, false for not.
        """
        random_val = random.random()
        print(random_val)
        return random_val > self.threshold

class AcceptFirstRequery():
    def __init__(self, threshold):
        """The initializer.

        args:
            threshold: threshold for re-query. Unused, but necessary for consistent calls.
        """
        pass

    def should_requery(self, distribution):
        """Should we re-query?

        args:
            distribution: the output distribution. Unused, but required for inheritance.

        returns:
            False
        """
        return False

class EntropyRequery():
    def __init__(self, threshold):
        """The initializer.

        args:
            threshold: threshold for re-query.
        """
        self.threshold = threshold

    def should_requery(self, distribution):
        """Should we re-query?

        args:
            distribution: the output distribution.

        returns:
            bool with whether or not the entropy is above the threshold
        """
        entropy = -(distribution*np.log(distribution)).sum()
        return entropy > self.threshold
