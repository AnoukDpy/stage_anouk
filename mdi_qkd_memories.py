from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson
from typing import Optional
from scipy.integrate import quad
from typing import Callable
from scipy.special import voigt_profile
import itertools as its
import scipy
from mpmath import mp

mp.dps = 80
##
class Heralding_Quantum_Memory:

    def __init__(self,*, writing_efficiency: float, reading_efficiency_0: float, storage_time: float, writing_time: float, reading_time: float):

        self.writing_efficiency = writing_efficiency
        self.reading_efficiency_0 = reading_efficiency_0
        self.storage_time = storage_time
        self.writing_time = writing_time
        self.reading_time = reading_time

    def extra_rounds_reading(self, rep_period):
        return np.ceil((self.writing_time+self.reading_time)/rep_period)-1

##
class Source(ABC):
    pass


class Attenuated_Laser(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, pulse_duration: Optional[float] = None, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

        if pulse_duration is None:
            self.pulse_duration = 0
        else:
            self.pulse_duration = pulse_duration

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

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


    def probability_sending_i_state(self,i) -> float:
        return poisson.pmf(i, self.mean_photon_number)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)

    def repetition_period(self):
        return 1/repetition_rate
##

class FiberChannel:
    """A fiber optic communication channel."""

    def __init__(self,*, loss_per_km: float, distance_km: float, detection_error: float, channel_background: float):
        """Initialize the fiber channel with the given parameters.

        Parameters
        ----------
        loss_per_km : float
            Attenuation coefficient in dB/km.
        distance_km : float
            Fiber length
        detection_error : float
        """
        self.loss_per_km = loss_per_km
        self.distance_km = distance_km
        self.detection_error = detection_error
        self.channel_background = channel_background

    @property
    def loss_per_km(self) -> float:
        """Return the attenuation coefficient of the fiber channel in dB/km. Must be positive."""
        return self._loss_per_km

    @loss_per_km.setter
    def loss_per_km(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"loss_per_km must be non-negative, got {value}")
        self._loss_per_km = float(value)

    @property
    def distance_km(self) -> float:
        """Return the fiber length in km. Must be positive."""
        return self._distance_km

    @distance_km.setter
    def distance_km(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"distance_km must be non-negative, got {value}")
        self._distance_km = float(value)

    @property
    def detection_error(self) -> float:
        """Return the detection error. Must be positive and less than 1."""
        return self._detection_error

    @detection_error.setter
    def detection_error(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError(f"detection_error must be non-negative and less than 1, got {value}")
        self._detection_error = float(value)

    def compute_channel_losses(self):
        return self.distance_km*self.loss_per_km

    def transmittance(self):
        return 10**(-self.compute_channel_losses()/10)

    def channel_bg_rate(self, pulse_duration):
        return 2*self.channel_background*pulse_duration

##
class Memory_Assisted_MDI_Protocol(Protocol):

    def __init__(self, *, source_1: Source,  detector: Detector, channel_1: FiberChannel, receiver: Receiver, correction_efficiency: float, memory: Heralding_Quantum_Memory, source_2: Source, channel_2: Optional[FiberChannel] = None):

        self.source_1 = source_1
        self.detector = detector
        self.channel_1 = channel_1
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.memory = memory

        if source_2 is None:
            self.source_2 = source_1
        else:
            self.source_2 = source_2

        if channel_2 is None:
            self.channel_2 = channel_1
        else:
            self.channel_2 = channel_2

## Alice's side, z basis

    def transmission_1(self, basis):
        if
        return self.detector.efficiency*self.receiver.z_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency()







