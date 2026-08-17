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
import qutip as qt
import itertools as its
import scipy
from mpmath import mp

mp.dps = 80

## Source

class Source(ABC):

    @abstractmethod

    def probability_sending_i_state(self,i) -> float:
        """ Probability of sending an i-photon state """

## Classes Sources filles

## Pulsed sources


class Attenuated_Laser(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

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


class Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, sources_num: int, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.sources_num = sources_num
        self.repetition_rate = repetition_rate

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

    def probability_sending_i_state(self,i):

        if i==0:
            return np.exp(-self.mean_photon_number*self.sources_num)

        else:
            return poisson.pmf(i, self.mean_photon_number)*(1-np.exp(-self.mean_photon_number*self.sources_num))/np.exp(-self.mean_photon_number)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Symmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float,repetition_rate: float, sources_num: int, transmittance: float, efficiency: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

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

    def probability_sending_i_state(self,i):
        k = math.log2(self.sources_num)
        return (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k)/math.factorial(i)+poisson.pmf(i, self.mean_photon_number)*(1-((1-self.efficiency)**i)*np.exp(-self.efficiency*self.mean_photon_number*(-1+1/self.transmittance**i)))*(1-np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k))/(1-np.exp(-self.efficiency*self.mean_photon_number/(self.transmittance**k)))

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Asymmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, sources_num: int, transmittance: float, efficiency: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

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

    def probability_sending_i_state(self,i):
        sum = 0
        for k in range(1, self.sources_num):
            if k == self.sources_num:
                k = self.sources_num -1
            sum += np.exp(-self.efficiency*self.mean_photon_number*((self.transmittance**(1-k)-1)/(1-self.transmittance)))*(1-((1-self.efficiency)**k))*np.exp(self.efficiency*self.mean_photon_number-self.efficiency*self.mean_photon_number/self.transmittance**k)

        return poisson.pmf(i, self.mean_photon_number)*sum + (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(((2-self.transmittance)*self.transmittance**(1-self.sources_num))-1)/(1-self.transmittance))/math.factorial(i)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Single_Photon_Source(Source):

    def __init__(self,*, repetition_rate: float, brightness: float, g2: float, optical_losses: Optional[float] = None):

        self.repetition_rate = repetition_rate
        self.brightness = brightness
        self.g2 = g2

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
    def brightness(self) -> float:
        """ Return the brightness, the probability of a detection.

        Must be non-negative and less than one.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        if value <= 0 or value>1 :
            raise ValueError(f"brightness must be non-negative and less than 1, got {value}")
        self._brightness = float(value)

    @property
    def g2(self) -> float:
        """ Return the g2(0).

        Must be non-negative
        """
        return self._g2

    @g2.setter
    def g2(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"g2 must be non-negative, got {value}")
        self._g2 = float(value)


    def probability_sending_i_state(self,i):
        if i<0 or i>2:
            return 0

        elif i==0:
            return 1-self.brightness

        else:
            p2 = (1-self.g2*self.brightness-np.sqrt(1-2*self.g2*self.brightness))/self.g2

            if i==1:
                return self.brightness-p2

            else:
                return p2

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Sagnac_Sources(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

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

    def brightness_parameter(self):
        return self.mean_photon_number/2


    def probability_sending_i_state(self,i) -> float:
        return ((i+1)*self.brightness_parameter()**i)/(self.brightness_parameter()+1)**(i+2)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)

    def two_modes_squeezed_vacuum_states(self, sources_number):
        c_1 = 2*self.mean_photon_number+1
        c_2 = 2*np.sqrt(self.mean_photon_number*(self.mean_photon_number+1))

        partial_x_quadrature_matrix = np.array([[c_1,c_2],
                                                [c_2,c_1]])

        partial_p_quadrature_matrix = np.array([[c_1,-c_2],
                                                [-c_2,c_1]])

        x_quadrature = np.kron(np.eye(2),partial_x_quadrature_matrix)

        p_quadrature = np.kron(np.eye(2),partial_p_quadrature_matrix)

        overall_x = np.kron(np.eye(sources_number), x_quadrature)

        overall_p = np.kron(np.eye(sources_number), p_quadrature)

        return scipy.linalg.block_diag(overall_x, overall_p)

    def emitted_pairs_covariance_matrix(self, sources_number):

        partial_swapping_matrix = np.array([[1,0,0,0],
                                            [0,0,0,1],
                                            [0,0,1,0],
                                            [0,1,0,0]])

        swapping_matrix = np.kron(np.eye(2*sources_number), partial_swapping_matrix)

        return np.dot(swapping_matrix.T, np.dot(self.two_modes_squeezed_vacuum_states(sources_number),swapping_matrix))




## Continuous wave pumped

class Continuous_Wave_Pumped_Source(Source):

    def __init__(self,*, brightness: float, g2_profile: Callable, optical_losses: Optional[float] = None):

        self.brightness = brightness
        self.g2_profile = g2_profile

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses


    @property
    def brightness(self) -> float:
        """ Return the brightness, the probability of a detection.

        Must be non-negative and less than one.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        if value <= 0 :
            raise ValueError(f"brightness must be non-negative, got {value}")
        self._brightness = float(value)


    def probability_sending_i_state(self,i, coincidence_time) -> float:
        return poisson.pmf(i, self.brightness*coincidence_time)

    def coincidence_window_efficiency(self, coincidence_time):

        return quad(self.g2_profile, -coincidence_time, coincidence_time)[0]

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


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


## Functions

def binary_shannon_entropy(x):
    if x<=0 or x>=1:
        return 0
    return -x*math.log2(x)-(1-x)*math.log2(1-x)

def no_x_event_i_time(x,i):

    return (1-x)**i

## FiberChannel

class FiberChannel:
    """A fiber optic communication channel."""

    def __init__(self,*, loss_per_km: float, distance_km: float, detection_error: float):
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



## Receiver

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


## Protocol

class Protocol(ABC):

    def __init__(self, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency


class BB84(Protocol):

    def __init__(self,*, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency

## X basis

    def transmittance_i_photon_state_x(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.x_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency())**i

    def yield_i_photon_state_x(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state_x(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state_x(self, i):
        return self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i)

    def overall_gain_x(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state_x(i)
        return gain

    def quantum_bit_error_rate_x(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state_x(i))/self.yield_i_photon_state_x(i)

    def overall_quantum_bit_error_rate_x(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate_x(i)*self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain_x()
        return qber

## Z basis

    def transmittance_i_photon_state_z(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.z_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency())**i

    def yield_i_photon_state_z(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state_z(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state_z(self, i):
        return self.yield_i_photon_state_z(i)*self.source.probability_sending_i_state(i)

    def overall_gain_z(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state_z(i)
        return gain

    def quantum_bit_error_rate_z(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state_z(i))/self.yield_i_photon_state_z(i)

    def overall_quantum_bit_error_rate_z(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate_z(i)*self.yield_i_photon_state_z(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain_z()
        return qber

## Key rates
    def gain(self):
        return (self.overall_gain_x()+self.overall_gain_z())/2

    def key_rate_decoy_state_inf_key(self):
        return self.source.probability_sending_i_state(0)*self.detector.background_rate() + self.source.probability_sending_i_state(1)*(self.yield_i_photon_state_x(1)+self.yield_i_photon_state_z(1))/2*(1-binary_shannon_entropy(self.quantum_bit_error_rate_x(1)))-self.overall_gain_z()*self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z())

    def key_rate_no_decoy_state_inf_key(self):

        delta = (1-self.source.probability_sending_i_state(0)-self.source.probability_sending_i_state(1))/self.gain()

        return self.gain()*((1-delta)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate_x()/(1-delta)))-self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z()))


class Pulsed_BBM92(Protocol):

    def __init__(self, *, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):

        self.source = source
        self.detector1 = detector1

        if detector2 is None:
                    self.detector2 = detector1
        else:
            self.detector2 = detector2

        self.channel_1 = channel_1

        if channel_2 is None:
            self.channel_2 = channel_1
        else:
            self.channel_2 = channel_2

        self.receiver1 = receiver1

        if receiver2 is None:
            self.receiver2 = receiver1
        else:
            self.receiver2 = receiver2

        self.correction_efficiency = correction_efficiency

## Basis Z
    def transmittance_i_photon_state1_Z(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.z_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_Z(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.z_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_stateZ(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1_Z(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2_Z(i)))

    def gain_i_photon_stateZ(self, i):

        return self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i)

    def overall_gainZ(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_stateZ(i)
        return gain


    def entanglement_errorZ(self,n,m):

        return 1/2-((1/2-((self.channel_1.detection_error+self.channel_2.detection_error+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateZ(n))*(-self.transmittance_i_photon_state1_Z(n-m)+self.transmittance_i_photon_state1_Z(m))*(-self.transmittance_i_photon_state2_Z(n-m)+self.transmittance_i_photon_state2_Z(m))


    def quantum_bit_error_rateZ(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_errorZ(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rateZ(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rateZ(i)*self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gainZ()
        return max(0,qber)

## Basis X
    def transmittance_i_photon_state1_X(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.x_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_X(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.x_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_stateX(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1_X(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2_X(i)))

    def gain_i_photon_stateX(self, i):

        return self.yield_i_photon_stateX(i)*self.source.probability_sending_i_state(i)

    def overall_gainX(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_stateX(i)
        return gain


    def entanglement_errorX(self,n,m):

        return 1/2-((1/2-((self.channel_1.detection_error+self.channel_2.detection_error+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateX(n))*(-self.transmittance_i_photon_state1_X(n-m)+self.transmittance_i_photon_state1_X(m))*(-self.transmittance_i_photon_state2_X(n-m)+self.transmittance_i_photon_state2_X(m))


    def quantum_bit_error_rateX(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_errorX(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rateX(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rateX(i)*self.yield_i_photon_stateX(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gainX()
        return max(0,qber)

## Final gain and key rate
    def final_gain(self):
        return (self.overall_gainX()+self.overall_gainZ())/2

    def key_rate(self):

        return (self.final_gain()/2)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rateX())-binary_shannon_entropy(self.overall_quantum_bit_error_rateZ())*self.correction_efficiency)


## BBM92 with continuous-wave pumped entangled photon sources

class BBM92_continuous_wave_pumped_source(Protocol):

    def __init__(self, *, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, coincidence_time: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):

        self.source = source
        self.detector1 = detector1

        if detector2 is None:
                    self.detector2 = detector1
        else:
            self.detector2 = detector2

        self.channel_1 = channel_1

        if channel_2 is None:
            self.channel_2 = channel_1
        else:
            self.channel_2 = channel_2

        self.receiver1 = receiver1

        if receiver2 is None:
            self.receiver2 = receiver1
        else:
            self.receiver2 = receiver2

        self.correction_efficiency = correction_efficiency

        self.coincidence_time = coincidence_time



## Overall detector error

    def overall_detector_error(self):
        return (self.channel_1.detection_error+self.channel_2.detection_error)/2

## Basis X

    def heralding_efficiency_1_x(self):
        return self.detector1.efficiency*self.receiver1.x_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()

    def heralding_efficiency_2_x(self):
        return self.detector2.efficiency*self.receiver2.x_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()

    def true_coincidence_rate_i_n_x(self,i,n):
        return self.heralding_efficiency_1_x()*self.heralding_efficiency_2_x()*no_x_event_i_time(self.heralding_efficiency_1_x(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_x(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_x()/2,n-i)*no_x_event_i_time(self.heralding_efficiency_1_x()/2,n-i)*no_x_event_i_time(self.detector1.background_rate()/2,1)*no_x_event_i_time(self.detector2.background_rate()/2,1)

    def true_coincidence_rate_x_i(self,i):
        sum = 0
        for n in range(1,i+1):
            sum = sum + self.true_coincidence_rate_i_n_x(n,i)
        return sum

    def true_coincidence_rate_x(self):
        sum = 0
        for i in range(1,5):
            sum = sum+self.true_coincidence_rate_x_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum/self.coincidence_time

    def miss_coincidence_x_i(self,i):
        return no_x_event_i_time(self.heralding_efficiency_1_x(),i)+no_x_event_i_time(self.heralding_efficiency_2_x(),i)-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*no_x_event_i_time(self.heralding_efficiency_2_x(),i)-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*(1-no_x_event_i_time(self.heralding_efficiency_2_x(),i))*self.detector1.background_rate()-no_x_event_i_time(self.heralding_efficiency_2_x(),i)*(1-no_x_event_i_time(self.heralding_efficiency_1_x(),i))*self.detector2.background_rate()-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*no_x_event_i_time(self.heralding_efficiency_2_x(),i)*self.detector1.background_rate()*self.detector2.background_rate()

    def miss_coincidence_x(self):
        sum = 0
        for i in range(1,5):
            sum = sum + self.miss_coincidence_x_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum

    def accidental_coincidence_rate_x(self):

        return (1-self.true_coincidence_rate_x()*self.coincidence_time-self.miss_coincidence_x()-self.source.probability_sending_i_state(0, self.coincidence_time) *(1-self.detector1.background_rate()*self.detector2.background_rate()))/self.coincidence_time

    def measured_coincidence_rate_x(self):

        return (self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_x()+self.accidental_coincidence_rate_x())

    def coincidence_error_rate_x(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_x()*self.overall_detector_error()+self.accidental_coincidence_rate_x()/2

    def qber_x(self):

        return self.coincidence_error_rate_x()/self.measured_coincidence_rate_x()

## Basis Z

    def heralding_efficiency_1_z(self):
        return self.detector1.efficiency*self.receiver1.z_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()

    def heralding_efficiency_2_z(self):
        return self.detector2.efficiency*self.receiver2.z_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()

    def true_coincidence_rate_i_n_z(self,i,n):
        return self.heralding_efficiency_1_z()*self.heralding_efficiency_2_z()*no_x_event_i_time(self.heralding_efficiency_1_z(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_z(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_z()/2,n-i)*no_x_event_i_time(self.heralding_efficiency_1_z()/2,n-i)*no_x_event_i_time(self.detector1.background_rate()/2,1)*no_x_event_i_time(self.detector2.background_rate()/2,1)

    def true_coincidence_rate_z_i(self,i):
        sum = 0
        for n in range(1,i+1):
            sum = sum + self.true_coincidence_rate_i_n_z(n,i)
        return sum

    def true_coincidence_rate_z(self):
        sum = 0
        for i in range(1,5):
            sum = sum+self.true_coincidence_rate_z_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum/self.coincidence_time

    def miss_coincidence_z_i(self,i):
        return no_x_event_i_time(self.heralding_efficiency_1_z(),i)+no_x_event_i_time(self.heralding_efficiency_2_z(),i)-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*no_x_event_i_time(self.heralding_efficiency_2_z(),i)-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*(1-no_x_event_i_time(self.heralding_efficiency_2_z(),i))*self.detector1.background_rate()-no_x_event_i_time(self.heralding_efficiency_2_z(),i)*(1-no_x_event_i_time(self.heralding_efficiency_1_z(),i))*self.detector2.background_rate()-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*no_x_event_i_time(self.heralding_efficiency_2_z(),i)*self.detector1.background_rate()*self.detector2.background_rate()

    def miss_coincidence_z(self):
        sum = 0
        for i in range(1,5):
            sum = sum + self.miss_coincidence_z_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum

    def accidental_coincidence_rate_z(self):

        return (1-self.true_coincidence_rate_z()*self.coincidence_time-self.miss_coincidence_z()-self.source.probability_sending_i_state(0, self.coincidence_time) *(1-self.detector1.background_rate()*self.detector2.background_rate()))/self.coincidence_time

    def measured_coincidence_rate_z(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_z()+self.accidental_coincidence_rate_z()

    def coincidence_error_rate_z(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_z()*self.overall_detector_error()+self.accidental_coincidence_rate_z()/2

    def qber_z(self):

        return self.coincidence_error_rate_z()/self.measured_coincidence_rate_z()

## Key rate

    def overall_measured_coincidence(self):

        return (self.measured_coincidence_rate_z()+self.measured_coincidence_rate_x())/2

    def key_rate(self):
        return (self.overall_measured_coincidence()/2)*(1-binary_shannon_entropy(self.qber_x())-self.correction_efficiency*binary_shannon_entropy(self.qber_z()))

## Entanglement swapping

## Functions

def conditionnal_detection_probability(no_dark_count_proba, channel_efficiency, x_value, n_photons_sent):
    if x_value == 0:
        return no_dark_count_proba*(1-channel_efficiency*no_dark_count_proba)**n_photons_sent

    else:
        return 1-(no_dark_count_proba*(1-channel_efficiency*no_dark_count_proba)**n_photons_sent)


## Protocol

class Continuous_Entanglement_swapping:

    def __init__(self,*, bell_measurement_number: int, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver):

        self.bell_measurement_number = bell_measurement_number
        self.source_1 = source_1
        self.channel_1 = channel_1
        self.detector_1 = detector_1
        self.receiver_1 = receiver_1

        self.covariance_matrix = self.source_1.emitted_pairs_covariance_matrix(bell_measurement_number+1)

        self.beam_splitter_matrix = self.beam_splitter_matrix(1/2)

        self.losses_matrix = self.losses_matrix()

        self.beam_splitter_entanglement = self.beam_splitter_entanglement()

    def beam_splitter_matrix(self, t):

        A = np.sqrt(np.clip(t, 0.0, 1.0))*np.eye(2)
        B = np.sqrt(np.clip(1-t, 0.0, 1.0))*np.eye(2)

        partial_matrix = np.block([[A,B],[-B,A]])

        return np.kron(np.eye(2), scipy.linalg.block_diag(np.eye(2),np.kron(np.eye(self.bell_measurement_number),partial_matrix),np.eye(2)))

    def beam_splitter_entanglement(self):

        return np.dot(self.beam_splitter_matrix.T, np.dot(self.covariance_matrix, self.beam_splitter_matrix))

    def polarizers_matrix(self, polarizer_angle_1, polarizer_angle_2):
        a_1 = np.cos(polarizer_angle_1)
        a_2 = np.sin(polarizer_angle_1)

        A = np.array([[a_1,a_2],
                    [-a_2,a_1]])

        b_1 = np.cos(polarizer_angle_2)
        b_2 = np.sin(polarizer_angle_2)

        B = np.array([[b_1,b_2],
                    [-b_2,b_1]])

        return np.kron(np.eye(2), scipy.linalg.block_diag(A, np.eye(4*self.bell_measurement_number),B))


    def apply_polarizer(self, polarizer_angle_1, polarizer_angle_2):

        pol_matrix = self.polarizers_matrix(polarizer_angle_1, polarizer_angle_2)

        return np.dot(pol_matrix.T, np.dot(self.beam_splitter_entanglement, pol_matrix))


    def losses_matrix(self):

        n = self.source_1.optical_efficiency()*self.detector_1.efficiency*self.channel_1.transmittance()*self.receiver_1.transmittance*self.receiver_1.x_basis_transmittance()

        return n*np.eye(8*(self.bell_measurement_number+1))

    def apply_losses_matrix(self, polarizer_angle_1, polarizer_angle_2):

        K = np.sqrt(self.losses_matrix)
        A = np.eye(8*(self.bell_measurement_number+1))-self.losses_matrix

        return np.dot(K.T, np.dot(self.apply_polarizer(polarizer_angle_1, polarizer_angle_2), K))+A

    def overall_coincidence_probability(self, polarizer_angle_1, polarizer_angle_2):

        n = self.bell_measurement_number

        M = self.apply_losses_matrix(polarizer_angle_1, polarizer_angle_2)

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
import qutip as qt
import itertools as its
import scipy
from mpmath import mp

mp.dps = 80

## Source

class Source(ABC):

    @abstractmethod

    def probability_sending_i_state(self,i) -> float:
        """ Probability of sending an i-photon state """

## Classes Sources filles

## Pulsed sources


class Attenuated_Laser(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

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


class Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, sources_num: int, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.sources_num = sources_num
        self.repetition_rate = repetition_rate

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

    def probability_sending_i_state(self,i):

        if i==0:
            return np.exp(-self.mean_photon_number*self.sources_num)

        else:
            return poisson.pmf(i, self.mean_photon_number)*(1-np.exp(-self.mean_photon_number*self.sources_num))/np.exp(-self.mean_photon_number)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Symmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float,repetition_rate: float, sources_num: int, transmittance: float, efficiency: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

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

    def probability_sending_i_state(self,i):
        k = math.log2(self.sources_num)
        return (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k)/math.factorial(i)+poisson.pmf(i, self.mean_photon_number)*(1-((1-self.efficiency)**i)*np.exp(-self.efficiency*self.mean_photon_number*(-1+1/self.transmittance**i)))*(1-np.exp(-self.efficiency*self.mean_photon_number*(2/self.transmittance)**k))/(1-np.exp(-self.efficiency*self.mean_photon_number/(self.transmittance**k)))

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Asymmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, sources_num: int, transmittance: float, efficiency: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

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

    def probability_sending_i_state(self,i):
        sum = 0
        for k in range(1, self.sources_num):
            if k == self.sources_num:
                k = self.sources_num -1
            sum += np.exp(-self.efficiency*self.mean_photon_number*((self.transmittance**(1-k)-1)/(1-self.transmittance)))*(1-((1-self.efficiency)**k))*np.exp(self.efficiency*self.mean_photon_number-self.efficiency*self.mean_photon_number/self.transmittance**k)

        return poisson.pmf(i, self.mean_photon_number)*sum + (1-self.efficiency)*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(((2-self.transmittance)*self.transmittance**(1-self.sources_num))-1)/(1-self.transmittance))/math.factorial(i)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Single_Photon_Source(Source):

    def __init__(self,*, repetition_rate: float, brightness: float, g2: float, optical_losses: Optional[float] = None):

        self.repetition_rate = repetition_rate
        self.brightness = brightness
        self.g2 = g2

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
    def brightness(self) -> float:
        """ Return the brightness, the probability of a detection.

        Must be non-negative and less than one.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        if value <= 0 or value>1 :
            raise ValueError(f"brightness must be non-negative and less than 1, got {value}")
        self._brightness = float(value)

    @property
    def g2(self) -> float:
        """ Return the g2(0).

        Must be non-negative
        """
        return self._g2

    @g2.setter
    def g2(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"g2 must be non-negative, got {value}")
        self._g2 = float(value)


    def probability_sending_i_state(self,i):
        if i<0 or i>2:
            return 0

        elif i==0:
            return 1-self.brightness

        else:
            p2 = (1-self.g2*self.brightness-np.sqrt(1-2*self.g2*self.brightness))/self.g2

            if i==1:
                return self.brightness-p2

            else:
                return p2

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


class Sagnac_Sources(Source):

    def __init__(self,*, mean_photon_number: float, repetition_rate: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

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

    def brightness_parameter(self):
        return self.mean_photon_number/2


    def probability_sending_i_state(self,i) -> float:
        return ((i+1)*self.brightness_parameter()**i)/(self.brightness_parameter()+1)**(i+2)

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)

    def two_modes_squeezed_vacuum_states(self, sources_number):
        c_1 = 2*self.mean_photon_number+1
        c_2 = 2*np.sqrt(self.mean_photon_number*(self.mean_photon_number+1))

        partial_x_quadrature_matrix = np.array([[c_1,c_2],
                                                [c_2,c_1]])

        partial_p_quadrature_matrix = np.array([[c_1,-c_2],
                                                [-c_2,c_1]])

        x_quadrature = np.kron(np.eye(2),partial_x_quadrature_matrix)

        p_quadrature = np.kron(np.eye(2),partial_p_quadrature_matrix)

        overall_x = np.kron(np.eye(sources_number), x_quadrature)

        overall_p = np.kron(np.eye(sources_number), p_quadrature)

        return scipy.linalg.block_diag(overall_x, overall_p)

    def emitted_pairs_covariance_matrix(self, sources_number):

        partial_swapping_matrix = np.array([[1,0,0,0],
                                            [0,0,0,1],
                                            [0,0,1,0],
                                            [0,1,0,0]])

        swapping_matrix = np.kron(np.eye(2*sources_number), partial_swapping_matrix)

        return np.dot(swapping_matrix.T, np.dot(self.two_modes_squeezed_vacuum_states(sources_number),swapping_matrix))




## Continuous wave pumped

class Continuous_Wave_Pumped_Source(Source):

    def __init__(self,*, brightness: float, g2_profile: Callable, optical_losses: Optional[float] = None):

        self.brightness = brightness
        self.g2_profile = g2_profile

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses


    @property
    def brightness(self) -> float:
        """ Return the brightness, the probability of a detection.

        Must be non-negative and less than one.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        if value <= 0 :
            raise ValueError(f"brightness must be non-negative, got {value}")
        self._brightness = float(value)


    def probability_sending_i_state(self,i, coincidence_time) -> float:
        return poisson.pmf(i, self.brightness*coincidence_time)

    def coincidence_window_efficiency(self, coincidence_time):

        return quad(self.g2_profile, -coincidence_time, coincidence_time)[0]

    def optical_efficiency(self):
        return 10**(-self.optical_losses/10)


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


## Functions

def binary_shannon_entropy(x):
    if x<=0 or x>=1:
        return 0
    return -x*math.log2(x)-(1-x)*math.log2(1-x)

def no_x_event_i_time(x,i):

    return (1-x)**i

## FiberChannel

class FiberChannel:
    """A fiber optic communication channel."""

    def __init__(self,*, loss_per_km: float, distance_km: float, detection_error: float):
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



## Receiver

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


## Protocol

class Protocol(ABC):

    def __init__(self, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency


class BB84(Protocol):

    def __init__(self,*, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency

## X basis

    def transmittance_i_photon_state_x(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.x_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency())**i

    def yield_i_photon_state_x(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state_x(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state_x(self, i):
        return self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i)

    def overall_gain_x(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state_x(i)
        return gain

    def quantum_bit_error_rate_x(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state_x(i))/self.yield_i_photon_state_x(i)

    def overall_quantum_bit_error_rate_x(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate_x(i)*self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain_x()
        return qber

## Z basis

    def transmittance_i_photon_state_z(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.z_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency())**i

    def yield_i_photon_state_z(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state_z(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state_z(self, i):
        return self.yield_i_photon_state_z(i)*self.source.probability_sending_i_state(i)

    def overall_gain_z(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state_z(i)
        return gain

    def quantum_bit_error_rate_z(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state_z(i))/self.yield_i_photon_state_z(i)

    def overall_quantum_bit_error_rate_z(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate_z(i)*self.yield_i_photon_state_z(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain_z()
        return qber

## Key rates
    def gain(self):
        return (self.overall_gain_x()+self.overall_gain_z())/2

    def key_rate_decoy_state_inf_key(self):
        return self.source.probability_sending_i_state(0)*self.detector.background_rate() + self.source.probability_sending_i_state(1)*(self.yield_i_photon_state_x(1)+self.yield_i_photon_state_z(1))/2*(1-binary_shannon_entropy(self.quantum_bit_error_rate_x(1)))-self.overall_gain_z()*self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z())

    def key_rate_no_decoy_state_inf_key(self):

        delta = (1-self.source.probability_sending_i_state(0)-self.source.probability_sending_i_state(1))/self.gain()

        return self.gain()*((1-delta)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate_x()/(1-delta)))-self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z()))


class Pulsed_BBM92(Protocol):

    def __init__(self, *, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):

        self.source = source
        self.detector1 = detector1

        if detector2 is None:
                    self.detector2 = detector1
        else:
            self.detector2 = detector2

        self.channel_1 = channel_1

        if channel_2 is None:
            self.channel_2 = channel_1
        else:
            self.channel_2 = channel_2

        self.receiver1 = receiver1

        if receiver2 is None:
            self.receiver2 = receiver1
        else:
            self.receiver2 = receiver2

        self.correction_efficiency = correction_efficiency

## Basis Z
    def transmittance_i_photon_state1_Z(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.z_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_Z(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.z_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_stateZ(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1_Z(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2_Z(i)))

    def gain_i_photon_stateZ(self, i):

        return self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i)

    def overall_gainZ(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_stateZ(i)
        return gain


    def entanglement_errorZ(self,n,m):

        return 1/2-((1/2-((self.channel_1.detection_error+self.channel_2.detection_error+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateZ(n))*(-self.transmittance_i_photon_state1_Z(n-m)+self.transmittance_i_photon_state1_Z(m))*(-self.transmittance_i_photon_state2_Z(n-m)+self.transmittance_i_photon_state2_Z(m))


    def quantum_bit_error_rateZ(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_errorZ(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rateZ(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rateZ(i)*self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gainZ()
        return max(0,qber)

## Basis X
    def transmittance_i_photon_state1_X(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.x_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_X(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.x_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_stateX(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1_X(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2_X(i)))

    def gain_i_photon_stateX(self, i):

        return self.yield_i_photon_stateX(i)*self.source.probability_sending_i_state(i)

    def overall_gainX(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_stateX(i)
        return gain


    def entanglement_errorX(self,n,m):

        return 1/2-((1/2-((self.channel_1.detection_error+self.channel_2.detection_error+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateX(n))*(-self.transmittance_i_photon_state1_X(n-m)+self.transmittance_i_photon_state1_X(m))*(-self.transmittance_i_photon_state2_X(n-m)+self.transmittance_i_photon_state2_X(m))


    def quantum_bit_error_rateX(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_errorX(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rateX(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rateX(i)*self.yield_i_photon_stateX(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gainX()
        return max(0,qber)

## Final gain and key rate
    def final_gain(self):
        return (self.overall_gainX()+self.overall_gainZ())/2

    def key_rate(self):

        return (self.final_gain()/2)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rateX())-binary_shannon_entropy(self.overall_quantum_bit_error_rateZ())*self.correction_efficiency)


## BBM92 with continuous-wave pumped entangled photon sources

class BBM92_continuous_wave_pumped_source(Protocol):

    def __init__(self, *, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, coincidence_time: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):

        self.source = source
        self.detector1 = detector1

        if detector2 is None:
                    self.detector2 = detector1
        else:
            self.detector2 = detector2

        self.channel_1 = channel_1

        if channel_2 is None:
            self.channel_2 = channel_1
        else:
            self.channel_2 = channel_2

        self.receiver1 = receiver1

        if receiver2 is None:
            self.receiver2 = receiver1
        else:
            self.receiver2 = receiver2

        self.correction_efficiency = correction_efficiency

        self.coincidence_time = coincidence_time



## Overall detector error

    def overall_detector_error(self):
        return (self.channel_1.detection_error+self.channel_2.detection_error)/2

## Basis X

    def heralding_efficiency_1_x(self):
        return self.detector1.efficiency*self.receiver1.x_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()

    def heralding_efficiency_2_x(self):
        return self.detector2.efficiency*self.receiver2.x_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()

    def true_coincidence_rate_i_n_x(self,i,n):
        return self.heralding_efficiency_1_x()*self.heralding_efficiency_2_x()*no_x_event_i_time(self.heralding_efficiency_1_x(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_x(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_x()/2,n-i)*no_x_event_i_time(self.heralding_efficiency_1_x()/2,n-i)*no_x_event_i_time(self.detector1.background_rate()/2,1)*no_x_event_i_time(self.detector2.background_rate()/2,1)

    def true_coincidence_rate_x_i(self,i):
        sum = 0
        for n in range(1,i+1):
            sum = sum + self.true_coincidence_rate_i_n_x(n,i)
        return sum

    def true_coincidence_rate_x(self):
        sum = 0
        for i in range(1,5):
            sum = sum+self.true_coincidence_rate_x_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum/self.coincidence_time

    def miss_coincidence_x_i(self,i):
        return no_x_event_i_time(self.heralding_efficiency_1_x(),i)+no_x_event_i_time(self.heralding_efficiency_2_x(),i)-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*no_x_event_i_time(self.heralding_efficiency_2_x(),i)-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*(1-no_x_event_i_time(self.heralding_efficiency_2_x(),i))*self.detector1.background_rate()-no_x_event_i_time(self.heralding_efficiency_2_x(),i)*(1-no_x_event_i_time(self.heralding_efficiency_1_x(),i))*self.detector2.background_rate()-no_x_event_i_time(self.heralding_efficiency_1_x(),i)*no_x_event_i_time(self.heralding_efficiency_2_x(),i)*self.detector1.background_rate()*self.detector2.background_rate()

    def miss_coincidence_x(self):
        sum = 0
        for i in range(1,5):
            sum = sum + self.miss_coincidence_x_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum

    def accidental_coincidence_rate_x(self):

        return (1-self.true_coincidence_rate_x()*self.coincidence_time-self.miss_coincidence_x()-self.source.probability_sending_i_state(0, self.coincidence_time) *(1-self.detector1.background_rate()*self.detector2.background_rate()))/self.coincidence_time

    def measured_coincidence_rate_x(self):

        return (self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_x()+self.accidental_coincidence_rate_x())

    def coincidence_error_rate_x(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_x()*self.overall_detector_error()+self.accidental_coincidence_rate_x()/2

    def qber_x(self):

        return self.coincidence_error_rate_x()/self.measured_coincidence_rate_x()

## Basis Z

    def heralding_efficiency_1_z(self):
        return self.detector1.efficiency*self.receiver1.z_basis_transmittance()*self.channel_1.transmittance()*self.source.optical_efficiency()

    def heralding_efficiency_2_z(self):
        return self.detector2.efficiency*self.receiver2.z_basis_transmittance()*self.channel_2.transmittance()*self.source.optical_efficiency()

    def true_coincidence_rate_i_n_z(self,i,n):
        return self.heralding_efficiency_1_z()*self.heralding_efficiency_2_z()*no_x_event_i_time(self.heralding_efficiency_1_z(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_z(),i-1)*no_x_event_i_time(self.heralding_efficiency_2_z()/2,n-i)*no_x_event_i_time(self.heralding_efficiency_1_z()/2,n-i)*no_x_event_i_time(self.detector1.background_rate()/2,1)*no_x_event_i_time(self.detector2.background_rate()/2,1)

    def true_coincidence_rate_z_i(self,i):
        sum = 0
        for n in range(1,i+1):
            sum = sum + self.true_coincidence_rate_i_n_z(n,i)
        return sum

    def true_coincidence_rate_z(self):
        sum = 0
        for i in range(1,5):
            sum = sum+self.true_coincidence_rate_z_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum/self.coincidence_time

    def miss_coincidence_z_i(self,i):
        return no_x_event_i_time(self.heralding_efficiency_1_z(),i)+no_x_event_i_time(self.heralding_efficiency_2_z(),i)-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*no_x_event_i_time(self.heralding_efficiency_2_z(),i)-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*(1-no_x_event_i_time(self.heralding_efficiency_2_z(),i))*self.detector1.background_rate()-no_x_event_i_time(self.heralding_efficiency_2_z(),i)*(1-no_x_event_i_time(self.heralding_efficiency_1_z(),i))*self.detector2.background_rate()-no_x_event_i_time(self.heralding_efficiency_1_z(),i)*no_x_event_i_time(self.heralding_efficiency_2_z(),i)*self.detector1.background_rate()*self.detector2.background_rate()

    def miss_coincidence_z(self):
        sum = 0
        for i in range(1,5):
            sum = sum + self.miss_coincidence_z_i(i)*self.source.probability_sending_i_state(i, self.coincidence_time)
        return sum

    def accidental_coincidence_rate_z(self):

        return (1-self.true_coincidence_rate_z()*self.coincidence_time-self.miss_coincidence_z()-self.source.probability_sending_i_state(0, self.coincidence_time) *(1-self.detector1.background_rate()*self.detector2.background_rate()))/self.coincidence_time

    def measured_coincidence_rate_z(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_z()+self.accidental_coincidence_rate_z()

    def coincidence_error_rate_z(self):

        return self.source.coincidence_window_efficiency(self.coincidence_time)*self.true_coincidence_rate_z()*self.overall_detector_error()+self.accidental_coincidence_rate_z()/2

    def qber_z(self):

        return self.coincidence_error_rate_z()/self.measured_coincidence_rate_z()

## Key rate

    def overall_measured_coincidence(self):

        return (self.measured_coincidence_rate_z()+self.measured_coincidence_rate_x())/2

    def key_rate(self):
        return (self.overall_measured_coincidence()/2)*(1-binary_shannon_entropy(self.qber_x())-self.correction_efficiency*binary_shannon_entropy(self.qber_z()))

## Entanglement swapping

## Functions

def conditionnal_detection_probability(no_dark_count_proba, channel_efficiency, x_value, n_photons_sent):
    if x_value == 0:
        return no_dark_count_proba*(1-channel_efficiency*no_dark_count_proba)**n_photons_sent

    else:
        return 1-(no_dark_count_proba*(1-channel_efficiency*no_dark_count_proba)**n_photons_sent)


## Protocol

class Continuous_Entanglement_swapping:

    def __init__(self,*, bell_measurement_number: int, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver):

        self.bell_measurement_number = bell_measurement_number
        self.source_1 = source_1
        self.channel_1 = channel_1
        self.detector_1 = detector_1
        self.receiver_1 = receiver_1

        self.covariance_matrix = self.source_1.emitted_pairs_covariance_matrix(bell_measurement_number+1)

        self.beam_splitter_matrix = self.beam_splitter_matrix(1/2)

        self.losses_matrix = self.losses_matrix()

        self.beam_splitter_entanglement = self.beam_splitter_entanglement()

    def beam_splitter_matrix(self, t):

        A = np.sqrt(np.clip(t, 0.0, 1.0))*np.eye(2)
        B = np.sqrt(np.clip(1-t, 0.0, 1.0))*np.eye(2)

        partial_matrix = np.block([[A,B],[-B,A]])

        return np.kron(np.eye(2), scipy.linalg.block_diag(np.eye(2),np.kron(np.eye(self.bell_measurement_number),partial_matrix),np.eye(2)))

    def beam_splitter_entanglement(self):

        return np.dot(self.beam_splitter_matrix.T, np.dot(self.covariance_matrix, self.beam_splitter_matrix))

    def polarizers_matrix(self, polarizer_angle_1, polarizer_angle_2):
        a_1 = np.cos(polarizer_angle_1)
        a_2 = np.sin(polarizer_angle_1)

        A = np.array([[a_1,a_2],
                    [-a_2,a_1]])

        b_1 = np.cos(polarizer_angle_2)
        b_2 = np.sin(polarizer_angle_2)

        B = np.array([[b_1,b_2],
                    [-b_2,b_1]])

        return np.kron(np.eye(2), scipy.linalg.block_diag(A, np.eye(4*self.bell_measurement_number),B))


    def apply_polarizer(self, polarizer_angle_1, polarizer_angle_2):

        pol_matrix = self.polarizers_matrix(polarizer_angle_1, polarizer_angle_2)

        return np.dot(pol_matrix.T, np.dot(self.beam_splitter_entanglement, pol_matrix))


    def losses_matrix(self):

        n = self.source_1.optical_efficiency()*self.detector_1.efficiency*self.channel_1.transmittance()*self.receiver_1.transmittance*self.receiver_1.x_basis_transmittance()

        return n*np.eye(8*(self.bell_measurement_number+1))

    def apply_losses_matrix(self, polarizer_angle_1, polarizer_angle_2):

        K = np.sqrt(self.losses_matrix)
        A = np.eye(8*(self.bell_measurement_number+1))-self.losses_matrix

        return np.dot(K.T, np.dot(self.apply_polarizer(polarizer_angle_1, polarizer_angle_2), K))+A

    def overall_coincidence_probability(self, polarizer_angle_1, polarizer_angle_2):
        n = self.bell_measurement_number

        M = self.apply_losses_matrix(polarizer_angle_1, polarizer_angle_2)

        M_mp = mp.matrix(M.tolist())

        probability = mp.mpf(0.0)

        measured_lines = [0]

        for i in range(n):

            measured_lines.append(2+4*i)
            measured_lines.append(5+4*i)

        measured_lines.append(4*n+2)

        for k in range(0,2*(n+1)+1):

            combinations = its.combinations(measured_lines, k)
            partial_sum = mp.mpf(0.0)

            back_ground_rate = mp.mpf(self.detector_1.background_rate())

            dark_count_factor = (mp.mpf(-2.0) *(mp.mpf(1.0)-back_ground_rate))** k

            for X in combinations:

                if k == 0:
                    partial_sum = partial_sum + mp.mpf(1.0)
                    continue

                P = tuple(y + 4*(n+1) for y in X)

                sub_x = mp.matrix([[M_mp[r, c] for c in X] for r in X])
                sub_p = mp.matrix([[M_mp[r, c] for c in P] for r in P])

                for d in range(k):
                    sub_x[d, d] += mp.mpf(1.0)
                    sub_p[d, d] += mp.mpf(1.0)

                    det_val = mp.det(sub_x) * mp.det(sub_p)

                    partial_sum = partial_sum + mp.mpf(1.0)/mp.sqrt(det_val)

            probability = probability + dark_count_factor*partial_sum

        return probability


    def visibility(self):

        p_max = self.overall_coincidence_probability(0,np.pi/2)
        p_min = self.overall_coincidence_probability(0,0)

        if (p_max+p_min)<=0:
            return 0

        v = (p_max-p_min)/(p_max+p_min)
        return v


## Graphs


def key_rate_distance_km_bb84(*, min, max, values_number, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        channel.distance_km = x
        protocol = BB84(source=source, detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color='blue', label="With active decoy state")
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_km_pulsed_bbm92(*, min, max, values_number, source: Source, detector1: Detector, detector2: Detector, channel_1: FiberChannel, channel_2: FiberChannel, receiver1: Receiver, receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    x_axis = []
    y1_values = []
    for x in x_values:
        channel_1.distance_km = x
        if channel_2 is None:
            ch2 = channel_1
        else:
            ch2 = channel_2

        x_axis.append(channel_1.distance_km+ch_2.distance_km)

        protocol = Pulsed_BBM92(source=source, detector1=detector1, channel_1=channel_1, channel_2 = ch_2, receiver1=receiver1, correction_efficiency=correction_efficiency, detector2=detector2, receiver2=receiver2)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color='red')
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_loss_bb84(*, min, max, values_number, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min * channel.loss_per_km, max * channel.loss_per_km, values_number)
    y1_values = []
    for x in x_values:
        channel.distance_km = x
        protocol = BB84(source=source, detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color='blue', label="With active decoy state")
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def key_rate_loss_pulsed_bbm92(*, min, max, values_number, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, title: str, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = []
    y1_values = []

    for x in x_values:
        channel_1.distance_km = x
        if channel_2 is None:
            ch2 = channel_1
        else:
            ch2 = channel_2
        horiz_axis.append(channel_1.distance_km * channel_1.loss_per_km + ch2.distance_km * ch2.loss_per_km)

        y1_values.append(Pulsed_BBM92(source=source, detector1=detector1, detector2=detector2, channel_1=channel_1, channel_2=ch2, receiver1=receiver1, receiver2=receiver2, correction_efficiency=correction_efficiency).key_rate())

    plt.plot(horiz_axis, y1_values, color='red')
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bits/s")
    plt.title(title)
    plt.grid(True)
    plt.show()

def key_rate_loss_bbm92_continuous(*, min, max, values_number, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, title: str, coincidence_time: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = []
    y1_values = []


    for x in x_values:
        channel_1.distance_km = x
        if channel_2 is None:
            ch2 = channel_1
        else:
            ch2 = channel_2

        protocol = BBM92_continuous_wave_pumped_source(source=source, detector1=detector1, detector2=detector2, channel_1=channel_1, channel_2=ch2, receiver1=receiver1, receiver2=receiver2, correction_efficiency=correction_efficiency, coincidence_time=coincidence_time)
        y1 = protocol.key_rate()
        y1_values.append(y1)
        horiz_axis.append(-10 * np.log10((protocol.heralding_efficiency_1_x() + protocol.heralding_efficiency_1_z()) / 2 * (protocol.heralding_efficiency_2_x() + protocol.heralding_efficiency_2_z()) / 2))

    plt.plot(horiz_axis, y1_values, color='red')
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bits/s")
    plt.title(title)
    plt.grid(True)
    plt.show()

def key_rate_brightness_continuous(*, min, max, distance: float, values_number, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, title: str, coincidence_time: float, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):
    x_values = np.linspace(min, max, values_number)
    x_axis = np.linspace(min, max, values_number)
    y1_values = []

    for x in x_values:
        source.brightness = x
        protocol = BBM92_continuous_wave_pumped_source(source=source, detector1=detector1, detector2=detector2, channel_1=channel_1, channel_2=channel_2, receiver1=receiver1, receiver2=receiver2, correction_efficiency=correction_efficiency, coincidence_time=coincidence_time)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(x_axis, y1_values, color='red')
    plt.yscale('log')
    plt.xlabel("Brightness")
    plt.ylabel("Key rate in bits/s")
    plt.title(title)
    plt.grid(True)
    plt.show()

def visibility_mean_photon_number(*,min, max, values_number, bell_measurement_number: int, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver):

    x_values = np.linspace(min, max, values_number)
    y_values = []

    for x in x_values:

        source_1.mean_photon_number = x
        entanglement_swapping = Continuous_Entanglement_swapping(bell_measurement_number = bell_measurement_number, source_1 = source_1, channel_1 = channel_1, detector_1 = detector_1, receiver_1 = receiver_1)

        visibility = entanglement_swapping.visibility()

        y_values.append(visibility)

    plt.plot(x_values, y_values)
  #  plt.ylim(bottom=0)
 #   plt.xscale('log')
    plt.xlabel("Mean photon number")
    plt.ylabel("Visibility")
   # plt.title(title)
    plt.grid(True)
    plt.show()

# Test entanglement swapping

source = Sagnac_Sources(mean_photon_number = 0.48, repetition_rate=0)

detector_7 = Threshold_detector(dark_count_rate=10**(5), efficiency=0.7, time_window=10**(-10), after_pulsing=0)

channel_7 = FiberChannel(loss_per_km=0.2, distance_km=10, detection_error=0.01)

receiver_7 = Receiver(transmittance=1)

visibility_mean_photon_number(min=0, max=0.2, values_number=200, bell_measurement_number = 1, source_1=source, channel_1=channel_7, detector_1=detector_7, receiver_1=receiver_7)

# Test 1:

source_1 = Attenuated_Laser(mean_photon_number=0.48, repetition_rate=0)

detector_1 = Threshold_detector(dark_count_rate=0.17, efficiency=5/100, time_window=10**(-5), after_pulsing=0) #Y_0 = 1.7*10**(-6)

receiver_1 = Receiver(transmittance=0.9)

FiberChannel_1 = FiberChannel(loss_per_km=0.21, distance_km=0, detection_error=0.033)

f_1 = 1.22

#key_rate_distance_km_bb84(min=0, max=160, values_number=300, source=source_1, detector=detector_1, channel=FiberChannel_1, receiver=receiver_1, correction_efficiency=f_1, title="Evolution of the key rate with the distance_km for an attenuated laser")


# Test 2:

source_2 = Symmetric_Multiplexed_Heralded_Photon_Source(mean_photon_number=0.48, repetition_rate=0, sources_num=32, transmittance=0.5, efficiency=0.7)

detector_2 = Threshold_detector(dark_count_rate=20, efficiency=0.25, time_window=10**(-8), after_pulsing=0)

receiver_2 = Receiver(transmittance=1)

FiberChannel_2 = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.005)

f_2 = 1.05

#key_rate_loss_bb84(min=0, max=275, values_number=300, source=source_2, detector=detector_2, channel=FiberChannel_2, receiver=receiver_2, correction_efficiency=f_2, title="Evolution of the key rate with the loss for SMHPS")


# Test 3:

source_3 = Asymmetric_Multiplexed_Heralded_Photon_Source(mean_photon_number=0.6, repetition_rate=0, sources_num=32, transmittance=0.5, efficiency=0.7)

detector_3 = Threshold_detector(dark_count_rate=20, efficiency=0.25, time_window=10**(-8), after_pulsing=0)

receiver_3 = Receiver(transmittance=1)

FiberChannel_3 = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.005)

f_3 = 1.05

#key_rate_loss_bb84(min=0, max=275, values_number=300, source=source_3, detector=detector_3, channel=FiberChannel_3, receiver=receiver_3, correction_efficiency=f_3, title="Evolution of the key rate with the loss for AMHPS")


# Test 4:

source_4 = Entangled_PDC_Source(mean_photon_number=0.053, repetition_rate=0)

detector_4 = Threshold_detector(dark_count_rate=6.02, efficiency=14.5/100, time_window=10**(-6), after_pulsing=0)

receiver_4 = Receiver(transmittance=1)

FiberChannel_4 = FiberChannel(loss_per_km=0.21, distance_km=0, detection_error=0.015)

f_4 = 1.22

#key_rate_loss_pulsed_bbm92(min=0, max=170, values_number=300, source=source_4, detector1=detector_4, detector2=detector_4, channel_1=FiberChannel_4, receiver1=receiver_4, receiver2=receiver_4, correction_efficiency=f_4, title="Evolution of the key rate with the loss for an entangled PDC source")

## Testing BBM92 continuous, Voigt profile

def g2_source_6(x):
    return voigt_profile(x, 123.2*10**(-12), 99.3*10**(-12))

source_6 = Continuous_Wave_Pumped_Source(brightness=1646*(10**5), g2_profile= g2_source_6, optical_losses = 4.5)

detector_6 = Threshold_detector(dark_count_rate=350, efficiency=0.76, time_window=310*10**(-12), after_pulsing=0)

channel_6 = FiberChannel(loss_per_km=0.1, distance_km=0, detection_error=0.005)

receiver_6 = Receiver(transmittance=1, x_basis_loss = 6, z_basis_loss = 3)

f_6 = 1.2

#graph_proba(0,3,source_6, "Spiral resonator source statistic")

#print(source_6.coincidence_window_efficiency(310*10**(-12)))

#key_rate_loss_bbm92_continuous(min=0, max=275, values_number=300, source=source_6, detector1=detector_6, channel_1=channel_6, channel_2 = channel_6, receiver1=receiver_6, correction_efficiency=f_6, title="Key rate evolution with the loss in dB",coincidence_time =310*10**(-12))

#key_rate_brightness_continuous(min = 0.00000000001,max= 10**9, distance=100, values_number=300, detector1=detector_6, channel_1=channel_6, receiver1=receiver_6, correction_efficiency=f_6, title="Key rate evolution with the loss in dB",coincidence_time =310*10**(-12), source = source_6)


## Testing BBM92 continuous, Gaussian profile


def g2_source_7(t):
    t_delta = 10**(-10)
    return (2.0 / t_delta) * np.sqrt(np.log(2.0) / np.pi)*np.exp(-4.0 * np.log(2.0) * (t ** 2) / (t_delta**2))

source_7 = Continuous_Wave_Pumped_Source(brightness=0.05*(10**9), g2_profile = g2_source_7)

detector_7 = Threshold_detector(dark_count_rate=250, efficiency=0.76, time_window=46*10**(-12), after_pulsing=0)

channel_7 = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.01)

receiver_7 = Receiver(transmittance=1)

f_7 = 1.2

#key_rate_loss_bbm92_continuous(min=0, max=400, values_number=400, source=source_7, detector1=detector_7, channel_1=channel_7, receiver1=receiver_7, correction_efficiency=f_7, title="Key rate evolution with the loss in dB",coincidence_time =46*10**(-12))

#key_rate_brightness_continuous(min = 0.00000000001,max= 10**(10), distance=200, values_number=300, detector1=detector_7, channel_1=channel_7, receiver1=receiver_7, correction_efficiency=f_7, title="Key rate evolution with the loss in dB",coincidence_time =46*10**(-12), source = source_7)


















