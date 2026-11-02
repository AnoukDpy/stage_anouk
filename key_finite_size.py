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

## Source

class Source(ABC):
    pass

## Classes Sources filles

## Pulsed sources


class Attenuated_Laser(Source):

    def __init__(self,*, mean_photon_number: np.ndarray, repetition_rate: float, optical_losses: Optional[float] = None):

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


    def probability_sending_i_state(self,i,k) -> float:
        mu = self.mean_photon_number[k]
        return math.exp(-mu) * (mu**i) / math.factorial(i)


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
        for k in range(1, self.sources_num+1):
            if k == self.sources_num:
                k_j = self.sources_num -1
            else:
                k_j=k
            sum += np.exp(-self.efficiency*self.mean_photon_number*((self.transmittance**(1-k)-1)/(1-self.transmittance)))*(1-((1-self.efficiency)**i))*np.exp(self.efficiency*self.mean_photon_number-self.efficiency*self.mean_photon_number/self.transmittance**k_j)

        return poisson.pmf(i, self.mean_photon_number)*sum + (1-self.efficiency)*self.mean_photon_number*np.exp(-(1-self.efficiency)*self.mean_photon_number)*np.exp(-self.efficiency*self.mean_photon_number*(((2-self.transmittance)*self.transmittance**(1-self.sources_num))-1)/(1-self.transmittance))/math.factorial(i)

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

class Entangled_PDC_Source(Source):

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


class Sagnac_Sources(Source):

    def __init__(self,*, mean_photon_number: float, optical_losses: Optional[float] = None):

        self.mean_photon_number = mean_photon_number

        if optical_losses is None:
            self.optical_losses = 0
        else:
            self.optical_losses = optical_losses

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
        return (self.source.probability_sending_i_state(0)*self.detector.background_rate() + self.source.probability_sending_i_state(1)*(self.yield_i_photon_state_x(1)+self.yield_i_photon_state_z(1))/2*(1-binary_shannon_entropy(self.quantum_bit_error_rate_x(1)))-self.overall_gain_z()*self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z()))/self.source.repetition_rate

    def key_rate_no_decoy_state_inf_key(self):

        delta = (1-self.source.probability_sending_i_state(0)-self.source.probability_sending_i_state(1))/self.gain()

        return self.gain()*((1-delta)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate_x()/(1-delta)))-self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate_z()))


def chernoff(x,y):
    if x <= 0:
        return 0.0
    return np.sqrt((x/2)*np.log(21/y))

def gamma(a, b, c, d):
    if c <= 0 or d <= 0 or b <= 0 or b >= 1:
        return 1.0

    term1 = ((c + d) * (1 - b) * b) / (c * d * np.log(2))
    term2 = ((c + d) / (c * d * (1 - b) * b)) * ((21**2) / (a**2))

    if term2 <= 1:
        return 1.0

    return float(np.sqrt(term1 * np.log2(term2)))

class finite_size_BB84(Protocol):

    def __init__(self,*, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, correction: float, security: float, pulse_number: float, proba_x_basis: float, proba_k_intensity: float, intensities_proba: np.ndarray):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.correction = correction
        self.security = security
        self.pulse_number = pulse_number
        self.proba_x_basis = proba_x_basis
        self.intensities_proba = intensities_proba
        self.nb_mean = len(self.source.mean_photon_number)

## X basis

    def transmittance_i_photon_state_x(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.x_basis_transmittance()*self.channel.transmittance()*self.source.optical_efficiency())**i

    def yield_i_photon_state_x(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state_x(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state_x(self, i, k):
        return self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i, k)

    def overall_gain_x(self, k):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state_x(i, k)
        return gain

    def quantum_bit_error_rate_x(self, i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state_x(i))/self.yield_i_photon_state_x(i)

    def overall_quantum_bit_error_rate_x(self, k):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate_x(i)*self.yield_i_photon_state_x(i)*self.source.probability_sending_i_state(i, k))
        qber = qber/self.overall_gain_x(k)
        return qber

    def qber(self):
        return self.overall_error_th_block_size("x") / self.overall_th_block_size("x")

## Finite size key rate

    def theoretical_block_size(self, k, basis):

        if basis == "x":

            return self.overall_gain_x(k)*(self.proba_x_basis**2)*self.pulse_number*self.intensities_proba[k]
        else:
            return self.overall_gain_x(k)*((1-self.proba_x_basis)**2)*self.pulse_number*self.intensities_proba[k]

    def overall_th_block_size(self, basis):
        s = 0
        for k in range(self.nb_mean):
            s = s+self.theoretical_block_size(k, basis)
        return s

    def theoretical_error_block_size(self, k, basis):

        if basis == "x":

            return self.overall_gain_x(k)*(self.proba_x_basis**2)*self.pulse_number*self.intensities_proba[k]*self.overall_quantum_bit_error_rate_x(k)
        else:
            return self.overall_gain_x(k)*((1-self.proba_x_basis)**2)*self.pulse_number*self.intensities_proba[k]*self.overall_quantum_bit_error_rate_x(k)

    def overall_error_th_block_size(self, basis):
        s = 0
        for k in range(self.nb_mean):
            s = s+self.theoretical_error_block_size(k, basis)
        return s


    def extremum_block_size(self, k, extr, basis):

        corr = chernoff(self.overall_th_block_size(basis), self.security)

        if extr == "min":
            return (np.exp(self.source.mean_photon_number[k])/self.intensities_proba[k])*max(0.0, self.theoretical_block_size(k, basis)-corr)

        else:
            return (np.exp(self.source.mean_photon_number[k])/self.intensities_proba[k])*(self.theoretical_block_size(k, basis)+corr)


    def error_block_size(self, k, extr, basis):
        corr = chernoff(self.overall_error_th_block_size(basis), self.security)

        if extr == "min":
            return (np.exp(self.source.mean_photon_number[k])/self.intensities_proba[k])*max(0.0, self.theoretical_error_block_size(k, basis)-corr)

        else:
            return (np.exp(self.source.mean_photon_number[k])/self.intensities_proba[k])*(self.theoretical_error_block_size(k, basis)+corr)

## events

    def tau(self, i):
        s = 0
        for k in range(self.nb_mean):
            s += self.intensities_proba[k] * self.source.probability_sending_i_state(i, k)
        return s

    def vacuum_events_min_x(self, basis="x"):
        return max(0.0, self.tau(0)*(self.source.mean_photon_number[1]*self.extremum_block_size(2, "min", basis)-self.source.mean_photon_number[2]*self.extremum_block_size(1, "min", basis))/(self.source.mean_photon_number[1]-self.source.mean_photon_number[2]))

    def single_photon_events_min(self, basis):
        return max(0.0, self.tau(1)*self.source.mean_photon_number[0]*(self.extremum_block_size(1, "min", basis)-self.extremum_block_size(2, "max", basis)-((self.source.mean_photon_number[1]**2-self.source.mean_photon_number[2]**2)/(self.source.mean_photon_number[0]**2))*(self.extremum_block_size(0, "max", basis)-self.vacuum_events_min_x(basis)/self.tau(0)))/(self.source.mean_photon_number[0]*(self.source.mean_photon_number[1]-self.source.mean_photon_number[2])-self.source.mean_photon_number[1]**2+self.source.mean_photon_number[2]**2))

    def single_photon_error_max_z(self):
        return max(0.0, self.tau(1)*(self.error_block_size(1, "max", "z")-self.error_block_size(2, "min", "z"))/(self.source.mean_photon_number[1]-self.source.mean_photon_number[2]))

    def phase_error_max(self):
        if self.single_photon_events_min("z") <= 0 or self.single_photon_events_min("x") <= 0:
            return 0.5
        return min(0.5, self.single_photon_error_max_z()/self.single_photon_events_min("z")+gamma(self.security,self.single_photon_error_max_z()/self.single_photon_events_min("z"),self.single_photon_events_min("z"), self.single_photon_events_min("x")))

    def leak_x(self):
        return self.overall_th_block_size("x")*self.correction_efficiency*binary_shannon_entropy(self.qber())

    def secret_key_length(self):
        if self.single_photon_events_min("x") <= 0 or self.single_photon_events_min("z") <= 0 or self.phase_error_max() >= 0.5:
            return 0.0
        secure = 6*np.log2(21/self.security)+np.log2(2/self.correction)
        # partie entière ??
        return self.vacuum_events_min_x("x")+self.single_photon_events_min("x")-self.single_photon_events_min("x")*binary_shannon_entropy(self.phase_error_max())-self.leak_x()-secure


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
    plt.savefig("bbm92_gaussian_g2.pdf", bbox_inches="tight")
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

def visibility_mean_photon_number(*,min, max, values_number, bell_measurement_number: int, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver, title: str):

    x_values = np.linspace(min, max, values_number)
    y_values = []

    for x in x_values:

        source_1.mean_photon_number = x
        entanglement_swapping = Continuous_Entanglement_swapping(bell_measurement_number = bell_measurement_number, source_1 = source_1, channel_1 = channel_1, detector_1 = detector_1, receiver_1 = receiver_1)

        visibility = entanglement_swapping.visibility()

        y_values.append(visibility)

    plt.plot(x_values, y_values)
    plt.xlabel("Mean photon number")
    plt.ylabel("Visibility")
    plt.title(title)
    plt.savefig("visibility_mean_photon_number.pdf", bbox_inches="tight")
    plt.grid(True)
    plt.show()

def visibility_mean_photon_number_jwoc(*,min, max, values_number, bell_measurement_number, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver, title: str):

    x_values = np.linspace(min, max, values_number)

    for n in bell_measurement_number:
        y_values = []
        for x in x_values:

            source_1.mean_photon_number = x
            entanglement_swapping = Continuous_Entanglement_swapping(bell_measurement_number = n, source_1 = source_1, channel_1 = channel_1, detector_1 = detector_1, receiver_1 = receiver_1)

            visibility = entanglement_swapping.visibility()

            y_values.append(visibility)

        plt.plot(x_values, y_values, label = f"{n} BM")

    plt.xlabel("Mean photon number",fontsize=15)
    plt.ylabel("Visibility",fontsize=15)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc="upper right",fontsize=15)
    plt.savefig("3_bm_visibility.svg", format="svg", bbox_inches="tight")
    plt.grid(True)
    plt.show()

def visibility_attenuation_jwoc(*,min, max, values_number, bell_measurement_number, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver, title: str):

    horiz_axis = np.linspace(min * channel_1.loss_per_km, max * channel_1.loss_per_km, values_number)

    for n in bell_measurement_number:
        y_values = []
        x_values = np.linspace(min/(2*(n+1)), max/(2*(n+1)), values_number)

        for x in x_values:
            channel_1.distance_km = x
            entanglement_swapping = Continuous_Entanglement_swapping(bell_measurement_number = n, source_1 = source_1, channel_1 = channel_1, detector_1 = detector_1, receiver_1 = receiver_1)

            visibility = entanglement_swapping.visibility()

            y_values.append(visibility)

        plt.plot(horiz_axis, y_values, label = f"{n} BM")

    plt.xlabel("Attenuation in dB",fontsize=15)
    plt.ylabel("Visibility",fontsize=15)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc="upper right",fontsize=15)
    plt.savefig("3_bm_visibility.svg", format="svg", bbox_inches="tight")
    plt.grid(True)
    plt.show()

def proba_attenuation_jwoc(*,min, max, values_number, bell_measurement_number, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver, title: str):

    horiz_axis = np.linspace(min * channel_1.loss_per_km, max * channel_1.loss_per_km, values_number)

    for n in bell_measurement_number:
        y_values = []
        x_values = np.linspace(min/(2*(n+1)), max/(2*(n+1)), values_number)

        for x in x_values:
            channel_1.distance_km = x
            entanglement_swapping = Continuous_Entanglement_swapping(bell_measurement_number = n, source_1 = source_1, channel_1 = channel_1, detector_1 = detector_1, receiver_1 = receiver_1)

            p_max = entanglement_swapping.overall_coincidence_probability(0,np.pi/2)

            y_values.append(p_max)

        plt.plot(horiz_axis, y_values, label = f"{n} BM")

    plt.xlabel("Attenuation in dB",fontsize=15)
    plt.ylabel("Coincidence probability",fontsize=15)
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc="upper right",fontsize=15)
   # plt.savefig("3_bm_visibility.svg", format="svg", bbox_inches="tight")
    plt.grid(True)
    plt.show()

def proba_mean_jwoc(*,min, max, values_number, bell_measurement_number, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver, title: str):

    x_values = np.linspace(min, max, values_number)

    for n in bell_measurement_number:
        y_values = []
        for x in x_values:

            source_1.mean_photon_number = x

            entanglement_swapping = Continuous_Entanglement_swapping(bell_measurement_number = n, source_1 = source_1, channel_1 = channel_1, detector_1 = detector_1, receiver_1 = receiver_1)

          #  p_max = entanglement_swapping.overall_coincidence_probability(0,np.pi/2)

            p_min = entanglement_swapping.overall_coincidence_probability(0,0)

            y_values.append(p_min)

        plt.plot(x_values, y_values, label = f"{n} BM")

    plt.xlabel("Mean photon number",fontsize=15)
    plt.ylabel("Coincidence probability",fontsize=15)
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc="upper right",fontsize=15)
   # plt.savefig("3_bm_visibility.svg", format="svg", bbox_inches="tight")
    plt.grid(True)
    plt.show()


def loop_key_rate_loss_bbm92_continuous(*, min, max, values_number, source: Source, detector1: Detector, channel_1: FiberChannel, receiver1: Receiver, correction_efficiency: float, coincidence_time: float, label, color, detector2: Optional[Detector] = None, channel_2: Optional[FiberChannel] = None, receiver2: Optional[Receiver] = None):
    x_values = np.linspace(min/2, max/2, values_number)
    horiz_axis = np.linspace(min * channel_1.loss_per_km, max *channel_1.loss_per_km, values_number)
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

    plt.plot(horiz_axis, y1_values, color=color, label=label)

def loop_key_rate_loss_bb84(*, min, max, values_number, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, color, label):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min * channel.loss_per_km, max * channel.loss_per_km, values_number)
    y1_values = []
    for x in x_values:
        channel.distance_km = x
        protocol = BB84(source=source, detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color=color, label=label)


def loop_key_rate_loss(*, min, max, values_number, sources_bb84, sources_bbm92, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str, coincidence_time: float):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min * channel.loss_per_km, max * channel.loss_per_km, values_number)

    for key in sources_bb84:
        label = sources_bb84[key][1]
        color = sources_bb84[key][2]
        loop_key_rate_loss_bb84(min = min, max=max, values_number=values_number, source=sources_bb84[key][0], detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency, color=color, label=label)

    for key in sources_bbm92:
        label = sources_bbm92[key][1]
        color = sources_bbm92[key][2]
        loop_key_rate_loss_bbm92_continuous(min=min, max=max, values_number=values_number, source=sources_bbm92[key][0], detector1=detector, channel_1=channel, receiver1=receiver, correction_efficiency=correction_efficiency, coincidence_time=coincidence_time, color=color, label=label)

    plt.yscale('log')
    plt.xlabel("Channel loss in dB",fontsize=15)
    plt.ylabel("Secret key rate in bits/s",fontsize=15)
    #plt.title(title)
    #plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.15))
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc="lower left",fontsize=11)
    plt.savefig("all_graphs_2.pdf", bbox_inches="tight")
    plt.grid(True)
    plt.show()


def loop_key_rate_loss_bb84_2(*, min, max, values_number, sources, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min * channel.loss_per_km, max * channel.loss_per_km, values_number)
    for key in sources:
        y1_values = []
        for x in x_values:
            channel.distance_km = x
            protocol = BB84(source=sources[key][0], detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency)
            y1 = protocol.key_rate_decoy_state_inf_key()
            y1_values.append(y1)

        plt.plot(horiz_axis, y1_values, label=sources[key][1])
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.savefig("bb84_all_sources_2.pdf", bbox_inches="tight")
    plt.grid(True)
    plt.show()

def loop_statistics(*,min, max, sources):
    n_values = np.arange(min, max)
    for key in sources:
        y_values = []
        for n in n_values:
            y_values.append(sources[key][0].probability_sending_i_state(n))
        plt.bar(n_values, y_values, color = sources[key][2],label=sources[key][1])

     #   plt.yscale('log')
        plt.xlabel("n")
        plt.ylabel("Probability of sending n photons")
        plt.title("Sources statistics, linear scale")
        plt.legend()
        plt.savefig(f"bb84_sources_stat_{key}.pdf", bbox_inches="tight")
        plt.grid(True)
        plt.show()



def loop_key_rate_loss_finite_size(*, min_dist, max_dist, values_number, source, detector, channel, receiver, correction_efficiency, N_values, proba_x_basis, intensities_proba, correction, security, title):
    x_values = np.linspace(min_dist, max_dist, values_number)
    horiz_axis = np.linspace(min_dist * channel.loss_per_km, max_dist * channel.loss_per_km, values_number)
    for N in N_values:
        y_rates = []
        for x in x_values:
            channel.distance_km = x
            protocol = finite_size_BB84(source=source, detector=detector, channel=channel, receiver=receiver, correction_efficiency=correction_efficiency, correction=correction, security=security, pulse_number=N, proba_x_basis=proba_x_basis, proba_k_intensity=0.0, intensities_proba=intensities_proba)
            l = protocol.secret_key_length()
            rate = l / N if l > 0 else np.nan
            y_rates.append(rate)
        label_N = f"N = 10^{int(np.log10(N))}" if N > 0 else f"N = {N}"
        plt.plot(horiz_axis, y_rates, label=label_N)
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Secret key rate (per pulse)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

source_1 = Attenuated_Laser(mean_photon_number=np.array([0.6, 0.15, 2e-4]), repetition_rate=10**9)
detector_1 = Threshold_detector(dark_count_rate=0.06, efficiency=0.10, time_window=10**(-5), after_pulsing=0.04)
receiver_1 = Receiver(transmittance=1.0)
FiberChannel_1 = FiberChannel(loss_per_km=0.2, distance_km=0, detection_error=0.005)
f_1 = 1.16
eps_cor = 10**(-15)
eps_sec = 10**(-10)
q_x = 0.75
p_intensities = np.array([0.6, 0.25, 0.15])
N_test_values = [10**6, 10**12, 10**13]

loop_key_rate_loss_finite_size(min_dist=0, max_dist=150, values_number=300, source=source_1, detector=detector_1, channel=FiberChannel_1, receiver=receiver_1, correction_efficiency=f_1, N_values=N_test_values, proba_x_basis=q_x, intensities_proba=p_intensities, correction=eps_cor, security=eps_sec, title="Finite-size BB84 : Taux de clé vs Pertes")

#key_rate_distance_km_bb84(min=0, max=160, values_number=300, source=source_1, detector=detector_1, channel=FiberChannel_1, receiver=receiver_1, correction_efficiency=f_1, title="Evolution of the key rate with the distance_km for an attenuated laser")
