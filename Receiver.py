from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson

class Receiver:

    def __init__(self, transmittance: float):

        self.transmittance = transmittance

    @property
    def transmittance(self) -> float:
        """ Return the transmittance of the detector.

        Must be non-negative and less than 1
        """
        return self._transmittance

    @transmittance.setter
    def transmittance(self, value: float) -> None:
        if value < 0 or value >1:
            raise ValueError(f"transmittance must be non-negative and less than 1, got {value}")

        self._transmittance = float(value)
