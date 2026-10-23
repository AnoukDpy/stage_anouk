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

def binary_shannon_entropy(x):
    if x<=0 or x>=1:
        return 0
    return -x*math.log2(x)-(1-x)*math.log2(1-x)

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
##

class Receiver:

    def __init__(self,*, transmittance: float, x_basis_loss: Optional[float] = None, z_basis_loss: Optional[float] = None):

        self.transmittance = transmittance


        if x_basis_loss is None:
            self.x_basis_loss = 0

        else:
            self.x_basis_loss = x_basis_loss

        if z_basis_loss is None:
            self.z_basis_loss = self.x_basis_loss

        else:
            self.z_basis_loss = z_basis_loss

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
    def x_basis_loss(self) -> float:
        """ Return the optical loss in the X basis.

        Must be non-negative.
        """
        return self._x_basis_loss

    @x_basis_loss.setter
    def x_basis_loss(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"x_basis_loss must be non-negative, got {value}")

        self._x_basis_loss = float(value)

    @property
    def z_basis_loss(self) -> float:
        """ Return the optical loss in the Z basis.

        Must be non-negative.
        """
        return self._z_basis_loss

    @z_basis_loss.setter
    def z_basis_loss(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"z_basis_loss must be non-negative, got {value}")

        self._z_basis_loss = float(value)

    def x_basis_transmittance(self):

        return 10**(-self.x_basis_loss/10)*self.transmittance

    def z_basis_transmittance(self):

        return 10**(-self.z_basis_loss/10)*self.transmittance


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
class Memory_Assisted_MDI_Protocol:

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

    #Sources must have the same repetition rate

## efficiencies

    def loading_probability_z(self, node):
        if node == 1:
            return 1- np.exp(-self.receiver.z_basis_transmittance()*self.channel_1.transmittance()*self.memory.writing_efficiency*self.source_1.mean_photon_number-self.memory.writing_efficiency*self.channel_1.channel_bg_rate(self.source_1.pulse_duration))

        else:
            return 1- np.exp(-self.receiver.z_basis_transmittance()*self.channel_2.transmittance()*self.memory.writing_efficiency*self.source_2.mean_photon_number-self.memory.writing_efficiency*self.channel_2.channel_bg_rate(self.source_2.pulse_duration))

    def single_photon_loading_proba_z(self, node):
        if node == 1:
            return 1-(1-self.receiver.z_basis_transmittance()*self.channel_1.transmittance()*self.memory.writing_efficiency)*np.exp(-self.memory.writing_efficiency*self.channel_1.channel_bg_rate(self.source_1.pulse_duration))

        else:
            return 1- (1-self.receiver.z_basis_transmittance()*self.channel_2.transmittance()*self.memory.writing_efficiency)*np.exp(-self.memory.writing_efficiency*self.channel_1.channel_bg_rate(self.source_2.pulse_duration))

## Loading proba, time

    def proba_synchronised_loading(self, eff_1, eff_2):
        return eff_1*eff_2/(eff_1+eff_2-eff_1*eff_2)

    def proba_advance_loading(self, node, eff_1, eff_2):

        den = 1-(1-eff_1)*(1-eff_2)

        if node == 1:
            return 1-eff_1/den
        else:
            return 1-eff_2/den

    def success_proba_advance(self, node, eff_1, eff_2):
        den = 1-(1-eff_1)*(1-eff_2)
        num = eff_1*eff_2
        time_term = np.exp(-self.source_1.repetition_period()/self.memory.coherence_time)

        if node == 1:
            return (1-eff_1)*num*time_term/(den*(1-(1-eff_1)*time_term))

        else:
            return (1-eff_2)*num*time_term/(den*(1-(1-eff_2)*time_term))

    def expected_loading_time(self, eff_1, eff_2):
        u = eff_1 + eff_2-eff_1*eff_2
        return (1/eff_1+1/eff_2+(eff_1*(1-eff_2)/(eff_2*u))+(eff_2*(1-eff_1)/(eff_1*u)))/2


    def exp_expected_loading_time(self, eff_1, eff_2):
        return 1/(1-np.exp(-self.source_1.repetition_period()/self.memory.storage_time)*(1-eff_1))+1/(1-np.exp(-self.source_1.repetition_period()/self.memory.storage_time)*(1-eff_2))-1

## measurement efficiencies

    def effective_measurement_efficiency(self):
        return self.detector.efficiency*self.memory.reading_efficiency_0

    def average_measurement_efficiency(self, eff_1, eff_2):
        return self.detector.efficiency*self.memory.reading_efficiency_0*self.proba_synchronised_loading(eff_1, eff_2)*self.exp_expected_loading_time(eff_1, eff_2)

    def single_photon_succes_bsm_proba(self, eff_1, eff_2):
        return ((1-self.detector.background_rate())**2)*(eff_1*eff_2/2+(2*eff_1+2*eff_2-3*eff_1*eff_2)*self.detector.background_rate()+4*(1-eff_1)*(1-eff_2)*self.detector.background_rate()**2)

## Gains

    def loading_memories_rate_z(self):
        return self.single_photon_succes_bsm_proba(self.effective_measurement_efficiency(),self.average_measurement_efficiency(self.loading_probability_z(1),self.loading_probability_z(2)))*(1/(self.expected_loading_time(self.loading_probability_z(1),self.loading_probability_z(2))+self.memory.extra_rounds_reading(self.source_1.repetition_period())))

    def qm_single_photon_rate_z(self):
        return self.loading_memories_rate_z()*self.single_photon_loading_proba_z(1)*self.single_photon_loading_proba_z(2)*self.source_1.mean_photon_number*self.source_2.mean_photon_number*np.exp(-self.source_1.mean_photon_number-self.source_2.mean_photon_number)/(self.loading_probability_z(1)*self.loading_probability_z(2))

## Misalignement terms

    def bg_loading(self, efficiency, node):
        if node == 1:
            return (1-np.exp(-self.memory.writing_efficiency*self.channel_1.channel_bg_rate(self.source_1.pulse_duration)))/efficiency
        else:
            return (1-np.exp(-self.memory.writing_efficiency*self.channel_2.channel_bg_rate(self.source_2.pulse_duration)))/efficiency

    def beta(self, efficiency, node):
        if node == 1:
            return (1-2*self.channel_1.detection_error)*(1-self.bg_loading(efficiency, 1))
        else:
            return (1-2*self.channel_2.detection_error)*(1-self.bg_loading(efficiency, 2))

    def misalignement_z(self, efficiency, node):
        if node == 1:
            return self.channel_1.detection_error*(1-self.bg_loading(efficiency,1))+self.bg_loading(efficiency,1)/2
        else:
            return self.channel_2.detection_error*(1-self.bg_loading(efficiency,2))+self.bg_loading(efficiency,2)/2

    def misalignement_deph(self, node, eff_1, eff_2):
        if node == 1:
            return (self.proba_advance_loading(2,eff_1, eff_2)-self.success_proba_advance(2,eff_1, eff_2))/2

        else:
            return (self.proba_advance_loading(1,eff_1, eff_2)-self.success_proba_advance(1,eff_1, eff_2))/2


    def expected_x_misalignement(self, eff_1, eff_2, node):
        if node == 1:
            return self.misalignement_z(eff_1, 1)+self.beta(eff_1, 1)*self.misalignement_deph(1,eff_1, eff_2)

        else:
            return self.misalignement_z(eff_2, 2)+self.beta(eff_2, 2)*self.misalignement_deph(2,eff_1, eff_2)

    def expected_x_misalignement_product(self, eff_1, eff_2):
        return self.misalignement_z(eff_1, 1)*self.misalignement_z(eff_2, 2)+self.misalignement_z(eff_2, 2)*self.beta(eff_1, 1)*self.misalignement_deph(1,eff_1, eff_2)+self.misalignement_z(eff_1, 1)*self.beta(eff_2, 2)*self.misalignement_deph(2,eff_1, eff_2)

    def error_single_photon(self, eff_1, eff_2, misalignement, basis):
        y11 = self.single_photon_succes_bsm_proba(eff_1, eff_2)
        if basis == "x":
            return (1/2*self.single_photon_succes_bsm_proba(eff_1, eff_2)-(1/2-misalignement)*((1-self.detector.background_rate())**2)*eff_1*eff_2/2)/y11

        else:
            return (1/2*self.single_photon_succes_bsm_proba(eff_1, eff_2)-(1/2-misalignement)*((1-self.detector.background_rate())**2)*(1-2*self.detector.background_rate())*eff_1*eff_2/2)/y11


    def qm_error_z(self, eff_1, eff_2):
        return self.misalignement_z(eff_1, 1)*(1-self.misalignement_z(eff_2, 2))+self.misalignement_z(eff_2, 2)*(1-self.misalignement_z(eff_1, 1))

    def expected_x_qm_misalignement(self, eff_1, eff_2):
        return self.expected_x_misalignement(eff_1, eff_2, 1)+self.expected_x_misalignement(eff_1, eff_2, 2)-2*self.expected_x_misalignement_product(eff_1, eff_2)

##

#    def qm_z_single_photon_error(self):
#        return self.error_single_photon(, , self.qm_error_z(self.single_photon_loading_proba_z(1), self.single_photon_loading_proba_z(2)),z)

    def qber_z_qm(self):
        return self.error_single_photon(self.effective_measurement_efficiency(), self.average_measurement_efficiency(self.loading_probability_z(1),self.loading_probability_z(2)), self.qm_error_z(self.loading_probability_z(1), self.loading_probability_z(2)), "z")


    def qm_single_qm_x(self):
        return self.error_single_photon(self.effective_measurement_efficiency(), self.average_measurement_efficiency(self.single_photon_loading_proba_z(1), self.single_photon_loading_proba_z(2)),self.expected_x_qm_misalignement(self.single_photon_loading_proba_z(1), self.single_photon_loading_proba_z(2)), "x")

##

    def secret_key_rate(self):
        return self.source_1.repetition_rate*(self.qm_single_photon_rate_z()*(1-binary_shannon_entropy(self.qm_single_qm_x()))-self.correction_efficiency*self.loading_memories_rate_z()*binary_shannon_entropy(self.qber_z_qm()))



###

def key_rate_loss(*, min, max, values_number, memory: Heralding_Quantum_Memory, source_1: Source, source_2: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min/2, max/2, values_number)
    horiz_axis = np.linspace(min * channel.loss_per_km, max * channel.loss_per_km, values_number)
    y1_values = []
    for x in x_values:
        channel.distance_km = x
        protocol = Memory_Assisted_MDI_Protocol(source_1=source_1,  detector=detector, channel_1=channel, receiver=receiver, correction_efficiency=correction_efficiency, memory=memory, source_2=source_2)
        y1 = protocol.secret_key_rate()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color='blue', label="With decoy state")
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in b/s")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


## warm vapor memory (Lorenzo)

wv_memory = Heralding_Quantum_Memory(writing_efficiency=0.05, reading_efficiency_0=0.05, storage_time=120*10**(-6), coherence_time=120*10**(-6), writing_time=1.43*10**(-9), reading_time=1.43*10**(-9))

##

repetition_rate = 518*10**6

channel = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.005, channel_background=0)

detector = Threshold_detector(dark_count_rate=1/repetition_rate, efficiency=0.93, time_window=1/repetition_rate, after_pulsing=0)

source = Attenuated_Laser(mean_photon_number=0.48, repetition_rate=repetition_rate, pulse_duration=1/repetition_rate)

receiver= Receiver(transmittance=1)

###

key_rate_loss(min=0, max=500, values_number=2000, memory=wv_memory, source_1=source, source_2=source, detector=detector, channel=channel, receiver=receiver, correction_efficiency=1.22, title="key rate MDI-QKD")















