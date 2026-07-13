from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson

## Source

class Source(ABC):
    def __init__(self, mean_photon_number: float, repetition_rate: float):

        self.mean_photon_number = mean_photon_number

        self.repetition_rate = repetition_rate


    @property
    def mean_photon_number(self) -> float:
        """ Return the mean photon number per pulse.

        Must be non-negative
        """
        return self._mean_photon_number

    @mean_photon_number.setter
    def mean_photon_number(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"mean_photon_number must be non-negative, got {value}")

        self._mean_photon_number = float(value)

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

        self._repetition_rate = float(value)


    @abstractmethod

    def probability_sending_i_photons(self,i) -> float:
        """ Probability of sending an i-photon state """


## Classes Sources filles

class Attenuated_Laser(Source):

    def __init__(self, mean_photon_number: float, repetition_rate: float):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

        super().__init__(mean_photon_number, repetition_rate)

    def probability_sending_i_photons(self,i) -> float:
        return poisson.pmf(i, self.mean_photon_number)


class Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, mean_photon_number: float, repetition_rate: float, sources_num: int):

        self.mean_photon_number = mean_photon_number
        self.sources_num = sources_num
        self.repetition_rate = repetition_rate

        super().__init__(mean_photon_number, repetition_rate)

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
            return np.exp(-self.mean_photon_number*self.sources_num)

        else:
            return poisson.pmf(i, self.mean_photon_number)*(1-np.exp(-self.mean_photon_number*self.sources_num))/np.exp(-self.mean_photon_number)

class Symmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, mean_photon_number: float,repetition_rate: float, sources_num: int, transmittance: float, efficiency: float):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

        super().__init__(mean_photon_number, repetition_rate)

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
        return (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k)/math.factorial(i)+poisson.pmf(i, self.mean_photon_number)*(1-((1-self.efficiency)**i)*np.exp(-self.efficiency*self.mean_photon_number*(-1+1/self.transmittance**i)))*(1-np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k))/(1-np.exp(-self.efficiency*self.mean_photon_number/(self.transmittance**k)))

class Asymmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, mean_photon_number: float, repetition_rate: float, sources_num: int, transmittance: float, efficiency: float):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

        super().__init__(mean_photon_number, repetition_rate)

    @property
    def sources_num(self) -> int:
        """ Return the HS units number.

        Must be non-negative
        """
        return self._sources_num

    @sources_num.setter
    def sources_num(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"sources_num must be non-negative, got {value}")
        self._sources_num = int(value)

    def probability_sending_i_photons(self,i):
        sum = 0
        for k in range(1, self.sources_num):
            if k == self.sources_num:
                k = self.sources_num -1
            sum += np.exp(-self.efficiency*self.mean_photon_number*((self.transmittance**(1-k)-1)/(1-self.transmittance)))*(1-((1-self.efficiency)**k))*np.exp(self.efficiency*self.mean_photon_number-self.efficiency*self.mean_photon_number/self.transmittance**k)

        return poisson.pmf(i, self.mean_photon_number)*sum + (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(((2-self.transmittance)*self.transmittance**(1-self.sources_num))-1)/(1-self.transmittance))/math.factorial(i)

















