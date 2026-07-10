from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math

## Detector

class Detector(ABC):

    def __init__(self, dark_count_rate: float, efficiency: float, time_window: float, transmittance: float):

        self.dark_count_rate = dark_count_rate
        self.efficiency = efficiency
        self.time_window = time_window
        self.transmittance = transmittance

    @property
    def dark_count_rate(self) -> float:
        """ Return the dark count rate in Hz.

        Must be non-negative
        """
        return self._dark_count_rate

    @dark_count_rate.setter
    def dark_count_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"dark_count_rate must be non-negative, got {value}")

        self._dark_count_rate = float(value)

    @property
    def efficiency(self) -> float:
        """ Return the efficiency of the detector.

        Must be non-negative and less than 1
        """
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value: float) -> None:
        if value < 0 or value >1:
            raise ValueError(f"efficiency must be non-negative and less than 1, got {value}")

        self._efficiency = float(value)

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

    @property
    def time_window(self) -> float:
        """ Return the time window of the detector in s.

        Must be non-negative
        """
        return self._time_window

    @time_window.setter
    def time_window(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"time_window must be non-negative, got {value}")

        self._time_window = float(value)

    @abstractmethod

    def dark_count_probability(self) -> float:

        """Probability of a dark count event"""

## Classes detector filles

class Threshold_detector(Detector):

    def __init__(self, dark_count_rate: float, efficiency: float, time_window: float, transmittance):

        self.dark_count_rate = dark_count_rate
        self.efficiency = efficiency
        self.time_window = time_window
        self.transmittance = transmittance

        super().__init__(dark_count_rate, efficiency, time_window, transmittance)

    def dark_count_probability(self):

        dark_count_probability = 2*self.dark_count_rate*self.time_window
        return min(1,dark_count_probability)