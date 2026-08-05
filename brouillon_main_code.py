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

## Source

class Source(ABC):

    @abstractmethod

    def probability_sending_i_state(self,i) -> float:
        """ Probability of sending an i-photon state """

## Classes Sources filles

## Pulsed sources

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

    def transmittance(self):
        return 10**(-self.distance_km*self.loss_per_km/10)



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

"""

class BB84(Protocol):

    def __init__(self,*, source: Source, detector: Detector, channel: FiberChannel, receiver: Receiver, correction_efficiency: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency

    def transmittance_i_photon_state(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.transmittance*self.channel.transmittance())**i

    def yield_i_photon_state(self, i):
        return  self.detector.background_rate() + self.transmittance_i_photon_state(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state(self, i):
        return self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i)

    def overall_gain(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state(i)
        return gain


    def quantum_bit_error_rate(self,i):
        return (1/2 * self.detector.background_rate() + (self.channel.detection_error+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state(i))/self.yield_i_photon_state(i)

    def overall_quantum_bit_error_rate(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate(i)*self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain()
        return qber

    def key_rate_decoy_state_inf_key(self):
        return self.source.probability_sending_i_state(0)*self.detector.background_rate() + self.source.probability_sending_i_state(1)*self.yield_i_photon_state(1)*(1-binary_shannon_entropy(self.quantum_bit_error_rate(1)))-self.overall_gain()*self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate())

    def key_rate_no_decoy_state_inf_key(self):

        delta = (1-self.source.probability_sending_i_state(0)-self.source.probability_sending_i_state(1))/self.overall_gain()

        return self.overall_gain()*((1-delta)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate()/(1-delta)))-self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate()))

"""
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
## Functions

def no_x_event_i_time(x,i):

    return (1-x)**i
## Protocol
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



## Graphs
"""
def key_rate_distance_km_bb84(min, max, values_number, source: Source, detector: Detector, FiberChannel: FiberChannel, correction_efficiency: float):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    y2_values = []
    for x in x_values:
        protocol = Protocol(source, detector, FiberChannel, correction_efficiency, x)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

        y2 = protocol.key_rate_no_decoy_state_inf_key()
        y2_values.append(y2)

    plt.plot(x_values, y1_values, color = 'blue', label = "With active decoy state")
    plt.plot(x_values, y2_values, color = 'red', label = "Without decoy state")
   # plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the distance_km")
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_km_bb84(min, max, values_number, source: Source, detector: Detector, FiberChannel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BB84(source, detector, FiberChannel, receiver, correction_efficiency, x)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color = 'blue', label = "With active decoy state")
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_km_BBM92(min, max, values_number, source: Source, detector1: Detector,detector2: Detector, channel_1: FiberChannel,channel_2: FiberChannel, receiver1: Receiver,receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BBM92(source, detector1,detector2, channel_1,channel_2, receiver1,receiver2, correction_efficiency, x,0)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color = 'red')
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def key_rate_distance_km_bb84_decoy_state(min, max, values_number, intensities, detector: Detector, FiberChannel: FiberChannel, correction_efficiency: float):
    for i in intensities:
        source = Attenuated_Laser(i)
        x_values = np.linspace(min, max, values_number)
        y1_values = []
        for x in x_values:
            protocol = Protocol(source, detector, FiberChannel, correction_efficiency, x)
            y1 = protocol.key_rate_decoy_state_inf_key()
            y1_values.append(y1)
        plt.plot(x_values, y1_values, label = f"Source's intensity = {i}")
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the distance_km")
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_km_bb84_mhps(min, max, values_number, intensity, hs_units, detector: Detector, FiberChannel: FiberChannel, correction_efficiency: float):
    for u in hs_units:
        source = Multiplexed_Heralded_Photon_Source(intensity, u )
        x_values = np.linspace(min, max, values_number)
        y1_values = []
        for x in x_values:
            protocol = Protocol(source, detector, FiberChannel, correction_efficiency, x)
            y1 = protocol.key_rate_decoy_state_inf_key()
            y1_values.append(y1)
        plt.plot(x_values, y1_values, label = f"HS units = {u}")
    plt.yscale('log')
    plt.xlabel("distance_km in km")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the distance_km")
    plt.legend()
    plt.grid(True)
    plt.show()


def key_rate_intensity_attenuated_laser(min, max, values_number, detector: Detector, FiberChannel: FiberChannel, correction_efficiency: float, distance_km: float):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    y2_values = []
    for x in x_values:
        source = Attenuated_Laser(x)
        protocol = Protocol(source, detector, FiberChannel, correction_efficiency, distance_km)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

        y2 = protocol.key_rate_no_decoy_state_inf_key()
        y2_values.append(y2)

    plt.plot(x_values, y1_values, color = 'blue', label = "With active decoy state")
    plt.plot(x_values, y2_values, color = 'red', label = "Without decoy state")
    plt.yscale('log')
    plt.xlabel("Source intensity")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the source's intensity")
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_intensity_attenuated_laser_test(min, max, values_number, detector: Detector, FiberChannel: FiberChannel, correction_efficiency: float, distance_km: float):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        source = Attenuated_Laser(x)
        protocol = Protocol(source, detector, FiberChannel, correction_efficiency, distance_km)
        y1 = protocol.key_rate_decoy_state_no_correction()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color = 'blue', label = "With active decoy state")
    plt.yscale('log')
    plt.xlabel("Source intensity")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the source's intensity")
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_intensity_mhps(min, max, values_number, sources_num: float, detector: Detector, FiberChannel: FiberChannel, correction_efficiency: float, distance_km: float):
    x_values = np.linspace(min, max, values_number)
    y_values = []
    for x in x_values:
        source = Multiplexed_Heralded_Photon_Source(x, sources_num)
        protocol = Protocol(source, detector, FiberChannel, correction_efficiency, distance_km)
        y = protocol.key_rate_decoy_state_inf_key()
        y_values.append(y)

    plt.plot(x_values, y_values)
    #plt.yscale('log')
    plt.xlabel("Source intensity")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the source's intensity")
    plt.grid(True)
    plt.show()

def key_rate_hs_units_mhps(min, max, intensity: float, detector: Detector, FiberChannel: FiberChannel, correction_efficiency: float, distance_km: float):
    x_values = np.arange(min,max+1)
    y_values = []
    for x in x_values:
        source = Multiplexed_Heralded_Photon_Source(intensity, x)
        protocol = Protocol(source, detector, FiberChannel, correction_efficiency, distance_km)
        y = protocol.key_rate_decoy_state_inf_key()
        y_values.append(y)

    plt.plot(x_values, y_values, 'x')
    #plt.yscale('log')
    plt.xlabel("HS units")
    plt.ylabel("Key rate")
    plt.title("Evolution of the key rate with the HS units")
    plt.grid(True)
    plt.show()

def graph_proba(min, max, source: Source, title: str):
    x_values = np.arange(min, max+1)
    y_values =[]
    for x in x_values:
        y = source.probability_sending_i_state(x)
        y_values.append(y)
    plt.plot(x_values, y_values, 'x')
    plt.yscale('log')
    plt.xlabel("Photons number")
    plt.ylabel("Probability of sending")
    plt.title(title)
    plt.grid(True)
    plt.show()


"""

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

key_rate_loss_bbm92_continuous(min=0, max=275, values_number=300, source=source_6, detector1=detector_6, channel_1=channel_6, channel_2 = channel_6, receiver1=receiver_6, correction_efficiency=f_6, title="Key rate evolution with the loss in dB",coincidence_time =310*10**(-12))

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

key_rate_loss_bbm92_continuous(min=0, max=400, values_number=400, source=source_7, detector1=detector_7, channel_1=channel_7, receiver1=receiver_7, correction_efficiency=f_7, title="Key rate evolution with the loss in dB",coincidence_time =46*10**(-12))

#key_rate_brightness_continuous(min = 0.00000000001,max= 10**(10), distance=200, values_number=300, detector1=detector_7, channel_1=channel_7, receiver1=receiver_7, correction_efficiency=f_7, title="Key rate evolution with the loss in dB",coincidence_time =46*10**(-12), source = source_7)













