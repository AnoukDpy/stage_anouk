from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson

## Source

class Source(ABC):
    def __init__(self, source_intensity: float, repetition_rate: float):
        
        self.source_intensity = source_intensity
        
        self.repetition_rate = repetition_rate


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

    @property
    def repetition_rate(self) -> float:
        """ Return the pulse number per second.

        Must be non-negative
        """
        return self._repetition_rate

    @repetition_rate.setter
    def repetition_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"repetition_rate must be non-negative, got {value}")

        self._after_pulse = float(value)

    
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

    def __init__(self, source_intensity: float, sources_num: int):

        self.source_intensity = source_intensity
        self.sources_num = sources_num

        super().__init__(source_intensity)

        @property
        def sources_num(self) -> int:
            """ Return the HS units number.
    
            Must be non-negative
            """
            return self._sources_num
    
        @sources_num.setter
        def sources_num(self, value: int) -> None:
            if value <= 0 :
                raise ValueError(f"sources_num must be non-negative, got {value}")
            self._sources_num = int(value)
    
        def probability_sending_i_photons(self,i):
    
            if i==0:
                return np.exp(-self.source_intensity*self.sources_num)
    
            else:
                return poisson.pmf(i, self.source_intensity)*(1-np.exp(-self.source_intensity*self.sources_num))/np.exp(-self.source_intensity)

class Symmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, source_intensity: float, sources_num: int, transmittance: float, efficiency: float):

        self.source_intensity = source_intensity
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

        super().__init__(source_intensity)

    @property
    def sources_num(self) -> int:
        """ Return the HS units number.

        Must be non-negative
        """
        return self._sources_num

    @sources_num.setter
    def sources_num(self, value: int) -> None:
        if value <= 0 or (value & (value - 1)) != 0:
            raise ValueError(f"sources_num must be non-negative and a power of 2  , got {value}")
        self._sources_num = int(value)

    def probability_sending_i_photons(self,i):
        k = math.log2(self.sources_num)
        return (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.source_intensity)*np.exp(-self.efficiency*self.source_intensity*(2/self.transmittance)**k)/math.factorial(i)+poisson.pmf(i, self.source_intensity)*(1-((1-self.efficiency)**i)*np.exp(-self.efficiency*self.source_intensity*(-1+1/self.transmittance**i)))*(1-np.exp(-self.efficiency*self.source_intensity*(2/self.transmittance)**k))/(1-np.exp(-self.efficiency*self.source_intensity/(self.transmittance**k)))

















