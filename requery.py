"""Re-Query classes"""
import random
import torch
import numpy as np

class QuasiRandomRequery():
    def __init__(self, threshold, num_samples=50):
        """The initializer.

        args:
            threshold: threshold for re-query. Since we're using uniform random, this is also the probability that the query is accepted.
        """
        self.threshold = threshold
        self.num_samples = num_samples
        self.count = 0
        self.deferred = 0
        self.required = float(threshold*num_samples)

        if not self.required.is_integer():
            print(f"Division is not even: {self.required}")

    def should_requery(self, distribution):
        """Should we re-query?

        args:
            distribution: the output distribution. Unused, but required for inheritance.

        returns:
            boolean. True for re-query, false for not.
        """
        random_val = random.random()
        remaining_chances = self.num_samples-self.count
        deferrals_remaining = float(self.required - self.deferred)
        if deferrals_remaining == 0:
            actual_p = 0
        else:
            actual_p = max(self.threshold, deferrals_remaining/remaining_chances)
        print("---")
        print(remaining_chances, deferrals_remaining, actual_p)
        print("---")
        if random_val < actual_p:
            self.deferred += 1
        self.count += 1
        return random_val < actual_p

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
