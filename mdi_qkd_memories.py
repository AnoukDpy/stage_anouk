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

    def __init__(self,*, writing_efficiency: float, reading_efficiency_0: float, storage_time: float, coherence_time: float, writing_time: float, reading_time: float):

        self.writing_efficiency = writing_efficiency
        self.reading_efficiency_0 = reading_efficiency_0
        self.storage_time = storage_time
        self.coherence_time = coherence_time
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

## Detector

class Detector(ABC):

    def __init__(self, dark_count_rate: float, efficiency: float, time_window: float, after_pulsing: float):

        self.dark_count_rate = dark_count_rate
        self.efficiency = efficiency
        self.time_window = time_window
        self.after_pulsing = after_pulsing

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

    @property
    def after_pulsing(self) -> float:
        """ Return the after pulsing probability of the detector.

        Must be non-negative
        """
        return self._after_pulsing

    @after_pulsing.setter
    def after_pulsing(self, value: float) -> None:
        if value < 0 or value >1:
            raise ValueError(f"after_pulsing must be non-negative and less than 1, got {value}")

        self._after_pulsing = float(value)

    @abstractmethod

    def dark_count_probability(self) -> float:

        """Dark count probability"""

    def background_rate(self) -> float:

        """Overall back ground rate"""

## Classes detector filles

class Threshold_detector(Detector):

    def __init__(self,*, dark_count_rate: float, efficiency: float, time_window: float, after_pulsing: float):

        self.dark_count_rate = dark_count_rate
        self.efficiency = efficiency
        self.time_window = time_window
        self.after_pulsing = after_pulsing

        super().__init__(dark_count_rate, efficiency, time_window, after_pulsing)

    def dark_count_probability(self):

        dark_count_proba = self.dark_count_rate*self.time_window
        return min(1,dark_count_proba)

    def background_rate(self):

        background_rate = 2*self.dark_count_probability()*(1+self.after_pulsing)
        return min(1,background_rate)

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

""" Sources must have the same repetition rate """

##

    def loading_probability_z(self, node):
        if node == "1":
            return 1- np.exp(-self.receiver.z_basis_transmittance()*self.channel_1.transmittance()*self.memory.writing_efficiency*self.source_1.mean_photon_number-self.memory.writing_efficiency*self.channel_1.channel_bg_rate(self.source_1.pulse_duration))

        else:
            return 1- np.exp(-self.receiver.z_basis_transmittance()*self.channel_2.transmittance()*self.memory.writing_efficiency*self.source_2.mean_photon_number-self.memory.writing_efficiency*self.channel_2.channel_bg_rate(self.source_2.pulse_duration))

    def single_photon_loading_proba_z(self, node):
        if node == "1":
            return 1-(1-self.receiver.z_basis_transmittance()*self.channel_1.transmittance()*self.memory.writing_efficiency)*np.exp(-self.memory.writing_efficiency*channel_bg_rate(self.source_1.pulse_duration))

        else:
            return 1- (1-self.receiver.z_basis_transmittance()*self.channel_2.transmittance()*self.memory.writing_efficiency)*np.exp(-self.memory.writing_efficiency*channel_bg_rate(self.source_2.pulse_duration))

    def proba_synchronised_loading(self):
        return loading_probability_z(1)*loading_probability_z(2)/(loading_probability_z(1)+loading_probability_z(2)-loading_probability_z(1)*loading_probability_z(2))

    def proba_advance_loading(self, node):

        den = 1-(1-self.loading_probability_z(1))*(1-self.loading_probability_z(2))

        if node == "1":
            return 1-self.loading_probability_z(1)/den
        else:
            return 1-self.loading_probability_z(2)/den

    def success_proba_advance(self, node):
        den = 1-(1-self.loading_probability_z(1))*(1-self.loading_probability_z(2))
        num = self.loading_probability_z(1)*self.loading_probability_z(2)
        time_term = np.exp(self.source_1.repetition_period()/self.memory.coherence_time)

        if node == "1":
            return (1-self.loading_probability_z(1))*num*time_term/(den*(1-(1-self.loading_probability_z(1))*time_term))

        else:
            return (1-self.loading_probability_z(2))*num*time_term/(den*(1-(1-self.loading_probability_z(2))*time_term))


    def expected_loading_time(self):
        return 1/(1-np.exp(self.source_1.repetition_period()/self.memory.storage_time)*(1-loading_probability_z(1)))+1/(1-np.exp(self.source_1.repetition_period()/self.memory.storage_time)*(1-loading_probability_z(2)))-1

    def effective_measurement_efficiency(self):
        return self.detector.efficiency*self.memory.reading_efficiency_0

    def average_measurement_efficiency(self):
        return self.detector.efficiency*self.memory.reading_efficiency_0*self.proba_synchronised_loading()*self.expected_loading_time()

    def single_photon_succes_bsm_proba(self, eff_1, eff_2):
        return ((1-self.detector.background_rate())**2)*(eff_1*eff_2/2+(2*eff_1+2*eff_2-3*eff_1*eff_2)*self.detector.background_rate()+4*(1-eff_1)*(1-eff_2)*self.detector.background_rate()**2)

    def loading_memories_rate_z(self):
        return self.single_photon_succes_bsm_proba()*(1/(self.expected_loading_time()+self.memory.extra_rounds_reading()))

    def single_photon_rate_z(self):
        return self.loading_memories_rate_z()*self.single_photon_loading_proba_z(1)*self.single_photon_loading_proba_z(2)*self.source_1.mean_photon_number*self.source_2.mean_photon_number*np.exp(-self.source_1.mean_photon_number-self.source_2.mean_photon_number)/(self.loading_probability_z(1)*self.loading_probability_z(2))

## Misalignement terms


    def bg_loading(self, efficiency, node):
        if node == "1":
            return (1-np.exp(-self.memory.writing_efficiency*self.channel_1.channel_bg_rate(self.source_1.pulse_duration)))/efficiency
        else:
            return (1-np.exp(-self.memory.writing_efficiency*self.channel_2.channel_bg_rate(self.source_2.pulse_duration)))/efficiency

    def beta(self, efficiency, node):
        if node == "1":
            return (1-2*self.channel_1.detection_error)*(1-self.bg_loading(efficiency, 1))
        else:
            return (1-2*self.channel_2.detection_error)*(1-self.bg_loading(efficiency, 2))

    def misalignement_z(self, efficiency, node):
        if node == "1":
            return self.channel_1.detection_error*(1-self.bg_loading(efficiency,1))+self.bg_loading(efficiency,1)/2
        else:
            return self.channel_2.detection_error*(1-self.bg_loading(efficiency,2))+self.bg_loading(efficiency,2)/2

    def misalignement_deph(self, node):
        if node == "1":
            return (self.proba_advance_loading(1)-self.success_proba_advance(1))/2

        else:
            return (self.proba_advance_loading(2)-self.success_proba_advance(2))/2


    def expected_x_misalignement(self, efficiency, node):
        if node == "1":
            return self.misalignement_z(efficiency, 1)+self.beta(efficiency, 1)*self.misalignement_deph(1)

        else:
            return self.misalignement_z(efficiency, 2)+self.beta(efficiency, 2)**self.misalignement_deph(2)

    def expected_x_misalignement_product(self, efficiency):
        return self.misalignement_z(efficiency, 1)*self.misalignement_z(efficiency, 2)+self.misalignement_z(efficiency, 2)*self.beta(efficiency, 1)*self.misalignement_deph(1)+self.misalignement_z(efficiency, 1)*self.beta(efficiency, 2)*self.misalignement_deph(2)

    def error_single_photon(self, eff_1, eff_2, misalignement, basis):
        if basis == "x":
            return 1/2*self.single_photon_succes_bsm_proba(eff_1, eff_2)-(1/2-misalignement)*((1-self.detector.background_rate())**2)*eff_1*eff_2/2

        else:
            return 1/2*self.single_photon_succes_bsm_proba(eff_1, eff_2)-(1/2-misalignement)*((1-self.detector.background_rate())**2)*(1-2*self.detector.background_rate())*eff_1*eff_2/2


    def qm_error_z(self, eff_1, eff_2):
        return self.misalignement_z(self, eff_1, 1)*(1-self.misalignement_z(self, eff_2, 2))+self.misalignement_z(self, eff_2, 2)*(1-self.misalignement_z(self, eff_1, 1))

    def expected_x_qm_misalignement(self, eff_1, eff_2):
        return self.expected_x_misalignement(eff_1, 1)+self.expected_x_misalignement(eff_2, 2)-2*self.expected_x_misalignement(eff_1, 1)*self.expected_x_misalignement(eff_2, 2)

##

    def qm_z_single_photon_error(self):
        return self.error_single_photon(, , self.qm_error_z(self.single_photon_loading_proba_z(1), self.single_photon_loading_proba_z(2)),z)

    def qber_z_qm(self):
        return self.error_single_photon(self.effective_measurement_efficiency(), self.average_measurement_efficiency(), self.qm_error_z(self.loading_probability_z(1), self.loading_probability_z(2)), z)


    def qm_single_qm_x(self):
        return self.error_single_photon(eff_1, eff_2, self.expected_x_qm_misalignement(self, eff_1, eff_2), x)









