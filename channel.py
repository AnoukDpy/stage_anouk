from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson



class Channel:

    def __init__(self, loss_coef: float, visibility: float):

        self.loss_coef = loss_coef
        self.visibility = visibility

    def transmittance(self, distance):

        return 10**(-self.loss_coef*distance/10)

    def probability_hitting_wrong_detector(self):

        return (1-self.visibility)/2