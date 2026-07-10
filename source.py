from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson

## Source

class Source(ABC):

    def __init__(self, source_intensity: float):

        self.source_intensity = source_intensity


    @property
    def source_intensity(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._source_intensity

    @source_intensity.setter
    def source_intensity(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"source_intensity must be non-negative, got {value}")

        self._source_intensity = float(value)


    @abstractmethod

    def probability_sending_i_photons(self,i) -> float:
        """ Probability of sending an i-photon state """


## Classes Sources filles

class Attenuated_Laser(Source):

    def __init__(self, source_intensity: float):

        self.source_intensity = source_intensity

        super().__init__(source_intensity)

    def probability_sending_i_photons(self,i) -> float:
        return poisson.pmf(i, self.source_intensity)


class Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, source_intensity: float, sources_num: float):

        self.source_intensity = source_intensity
        self.sources_num = sources_num

        super().__init__(source_intensity)

    def probability_sending_i_photons(self,i):

        if i==0:
            return np.exp(-self.source_intensity*self.sources_num)

        else:
            return poisson.pmf(i, self.source_intensity)*(1-np.exp(-self.source_intensity*self.sources_num))/np.exp(-self.source_intensity)