from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson


## Source

class Source(ABC):
    def __init__(self, repetition_rate: float):

        self.repetition_rate = repetition_rate


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

    def probability_sending_i_state(self,i) -> float:
        """ Probability of sending an i-photon state """


## Classes Sources filles

class Attenuated_Laser(Source):

    def __init__(self, mean_photon_number: float, repetition_rate: float):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

        super().__init__(repetition_rate)

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


class Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, mean_photon_number: float, repetition_rate: float, sources_num: int):

        self.mean_photon_number = mean_photon_number
        self.sources_num = sources_num
        self.repetition_rate = repetition_rate

        super().__init__(repetition_rate)

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

class Symmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, mean_photon_number: float,repetition_rate: float, sources_num: int, transmittance: float, efficiency: float):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

        super().__init__(repetition_rate)

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

class Asymmetric_Multiplexed_Heralded_Photon_Source(Source):

    def __init__(self, mean_photon_number: float, repetition_rate: float, sources_num: int, transmittance: float, efficiency: float):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate
        self.sources_num = sources_num
        self.transmittance = transmittance
        self.efficiency = efficiency

        super().__init__(repetition_rate)

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


class Single_Photon_Source(Source):

    def __init__(self, repetition_rate: float, brightness: float, g2: float):

        self.repetition_rate = repetition_rate
        self.brightness = brightness
        self.g2 = g2

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


class Entangled_PDC_Source(Source):

    def __init__(self, mean_photon_number: float, repetition_rate: float):

        self.mean_photon_number = mean_photon_number
        self.repetition_rate = repetition_rate

        super().__init__(repetition_rate)

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

    def __init__(self, dark_count_rate: float, efficiency: float, time_window: float, after_pulsing: float):

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

class FiberFiberChannel:
    """A fiber optic communication channel."""

    def __init__(self, loss_per_km: float, visibility: float):
        """Initialize the fiber channel with the given parameters.

        Parameters
        ----------
        loss_per_km : float
            Attenuation coefficient in dB/km.
        Visibility : float
        """
        self.loss_per_km = loss_per_km
        self.visibility = visibility

    @property
    def loss_per_km(self) -> float:
        """Return the attenuation coefficient of the fiber channel in dB/km. Must be positive."""
        return self._loss_per_km

    @loss_per_km.setter
    def loss_per_km(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"loss_per_km must be positive, got {value}")
        self._loss_per_km = value

    def compute_channel_losses(self, distance_km) -> float:
        """Return the total losses of the fiber channel in dB."""
        if distance_km< 0:
            raise ValueError(f"distance_km must be positive, got {value}")
        return distance_km * self.loss_per_km

    def transmittance(self, distance_km):
        return 10**(-self.compute_channel_losses(distance_km)/10)

    def probability_hitting_wrong_detector(self):
        return (1-self.visibility)/2

## Receiver

class Receiver:

    def __init__(self, transmittance: float):

        self.transmittance = transmittance

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


## Protocol

class Protocol(ABC):

    def __init__(self, source: Source, detector: Detector, FiberChannel: FiberChannel, receiver: Receiver, correction_efficiency: float, distance_km: float):

        self.source = source
        self.detector = detector
        self.FiberChannel = FiberChannel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.distance_km = distance_km

class BB84(Protocol):

    def __init__(self, source: Source, detector: Detector, FiberChannel: FiberChannel, receiver: Receiver, correction_efficiency: float, distance_km: float):

        self.source = source
        self.detector = detector
        self.FiberChannel = FiberChannel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.distance_km = distance_km

    def transmittance_i_photon_state(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.transmittance*self.FiberChannel.transmittance(self.distance_km))**i

    def yield_i_photon_state(self, i):
        #probability for Bob to have a detection assuming that Alice sent an i-photon state
        return  self.detector.background_rate() + self.transmittance_i_photon_state(i)*(1+self.detector.after_pulsing)

    def gain_i_photon_state(self, i):
        #probability for Alice to send an i-photon state and for Bob to have a detection
        return self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i)

    def overall_gain(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state(i)
        return gain


    def quantum_bit_error_rate(self,i):
        return (1/2 * self.detector.background_rate() + (self.FiberChannel.probability_hitting_wrong_detector()+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state(i))/self.yield_i_photon_state(i)

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


class BBM92(Protocol):

    def __init__(self, source: Source, detector1: Detector, detector2: Detector, FiberChannel1: FiberChannel, FiberChannel2: FiberChannel, receiver1: Receiver, receiver2: Receiver, correction_efficiency: float, distance_km1: float, distance_km2: float):

        self.source = source
        self.detector1 = detector1
        self.detector2 = detector2
        self.FiberChannel1 = FiberChannel1
        self.FiberChannel2 = FiberChannel2
        self.receiver1 = receiver1
        self.receiver2 = receiver2
        self.correction_efficiency = correction_efficiency
        self.distance_km1 = distance_km1
        self.distance_km2 = distance_km2

    def transmittance_i_photon_state1(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.transmittance*self.FiberChannel1.transmittance(self.distance_km1)*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.transmittance*self.FiberChannel2.transmittance(self.distance_km2)*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_state(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2(i)))

    def gain_i_photon_state(self, i):

        return self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i)

    def overall_gain(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state(i)
        return gain


    def entanglement_error(self,n,m):

        return 1/2-((1/2-((self.FiberChannel1.probability_hitting_wrong_detector()+self.FiberChannel2.probability_hitting_wrong_detector()+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_state(n))*(-self.transmittance_i_photon_state1(n-m)+self.transmittance_i_photon_state1(m))*(-self.transmittance_i_photon_state2(n-m)+self.transmittance_i_photon_state2(m))


    def quantum_bit_error_rate(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_error(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rate(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate(i)*self.yield_i_photon_state(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain()
        return max(0,qber)

    def key_rate(self):

        return (self.overall_gain()/2)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate())*(1+self.correction_efficiency))

class New_BBM92(Protocol):

    def __init__(self, source: Source, detector1: Detector, detector2: Detector, FiberChannel1: FiberChannel, FiberChannel2: FiberChannel, receiver1: Receiver, receiver2: Receiver, correction_efficiency: float, distance_km1: float, distance_km2: float):

        self.source = source
        self.detector1 = detector1
        self.detector2 = detector2
        self.FiberChannel1 = FiberChannel1
        self.FiberChannel2 = FiberChannel2
        self.receiver1 = receiver1
        self.receiver2 = receiver2
        self.correction_efficiency = correction_efficiency
        self.distance_km1 = distance_km1
        self.distance_km2 = distance_km2

## Basis Z
    def transmittance_i_photon_state1_Z(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.transmittance*self.FiberChannel1.transmittance(self.distance_km1)*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_Z(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.transmittance*self.FiberChannel2.transmittance(self.distance_km2)*(1+self.detector2.after_pulsing))**i

    def yield_i_photon_stateZ(self, i):

        return  (1-(1-self.detector1.background_rate())*(1-self.transmittance_i_photon_state1_Z(i)))*(1-(1-self.detector2.background_rate())*(1-self.transmittance_i_photon_state2_Z(i)))

    def gain_i_photon_stateZ(self, i):

        return self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i)

    def overall_gainZ(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_stateZ(i)
        return gain


    def entanglement_error Z(self,n,m):

        return 1/2-((1/2-((self.FiberChannel1.probability_hitting_wrong_detector()+self.FiberChannel2.probability_hitting_wrong_detector()+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateZ(n))*(-self.transmittance_i_photon_state1_Z(n-m)+self.transmittance_i_photon_state1_Z(m))*(-self.transmittance_i_photon_state2_Z(n-m)+self.transmittance_i_photon_state2_Z(m))


    def quantum_bit_error_rateZ(self,i):
        qber = 0
        for n in range(0,i+1):
            qber = qber + self.entanglement_errorZ(i,n)
        return qber/(1+i)


    def overall_quantum_bit_error_rateZ(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rateZ(i)*self.yield_i_photon_stateZ(i)*self.source.probability_sending_i_state(i))
        qber = qber/self.overall_gain1()
        return max(0,qber)

## Basis X
    def transmittance_i_photon_state1_X(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.transmittance*self.FiberChannel1.transmittance(self.distance_km1)*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2_X(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.transmittance*self.FiberChannel2.transmittance(self.distance_km2)*(1+self.detector2.after_pulsing))**i

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

        return 1/2-((1/2-((self.FiberChannel1.probability_hitting_wrong_detector()+self.FiberChannel2.probability_hitting_wrong_detector()+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_stateX(n))*(-self.transmittance_i_photon_state1_X(n-m)+self.transmittance_i_photon_state1_X(m))*(-self.transmittance_i_photon_state2_X(n-m)+self.transmittance_i_photon_state2_X(m))


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

        return (self.final_gain()/2)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rateX())+binary_shannon_entropy(self.overall_quantum_bit_error_rateZ())*self.correction_efficiency)



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
"""
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

def key_rate_distance_km_BBM92(min, max, values_number, source: Source, detector1: Detector,detector2: Detector, FiberChannel1: FiberChannel,FiberChannel2: FiberChannel, receiver1: Receiver,receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BBM92(source, detector1,detector2, FiberChannel1,FiberChannel2, receiver1,receiver2, correction_efficiency, x,0)
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




"""Key rate evolution with the distance_km:"""

## Sources

source1 = Attenuated_Laser(0.48, 0)

source2 = Multiplexed_Heralded_Photon_Source(0.48,0,32)

source3 = Symmetric_Multiplexed_Heralded_Photon_Source(0.48,0,8, 0.2, 0.1)

source4 = Asymmetric_Multiplexed_Heralded_Photon_Source(0.4,0,8, 0.2, 0.1)

source5 = Single_Photon_Source(0, 80/100, 1/100)

source6 = Entangled_PDC_Source(0.053, 0)

## FiberChannels, error corrections and receivers

FiberChannel1 = FiberFiberChannel(0.21,0.934) #Ma's values

FiberChannel2 = FiberFiberChannel(0.21,0.97)

f =1.22

receiver1 = Receiver(5/10)
receiver2 = Receiver(14/100)
receiver3 = Receiver(1)
## Detectors

detector1 = Threshold_detector(0.17,1/10,10**(-5),0.8/100) #Y_0 = 1.7*10**(-6)

detector2 = Threshold_detector(0.17,1/10,10**(-5),0) #without after pulsing

detector3 = Threshold_detector(6.02,14.5/100,10**(-6),0) #without after pulsing

## Calls

#key_rate_distance_km_bb84(0,160,300,source1,detector1,FiberChannel1,receiver1,f,"Evolution of the key rate with the distance_km for an attenuated laser")

#key_rate_distance_km_bb84(0,160,300,source2,detector1,FiberChannel1,receiver1,f,"Evolution of the key rate with the distance_km for a MHPS")

#key_rate_distance_km_bb84(0,160,300,source3,detector2,FiberChannel1,receiver1,f,"Evolution of the key rate with the distance_km for a SMHPS")

#key_rate_distance_km_bb84(0,160,300,source4,detector1,FiberChannel1,receiver1,f,"Evolution of the key rate with the distance_km for an AMHPS")

#key_rate_distance_km_bb84(0,160,300,source5,detector1,FiberChannel1,receiver1,f,"Evolution of the key rate with the distance_km for a single photon source")

#key_rate_distance_km_BBM92(0,160,300, source6, detector3, detector3, FiberChannel2, FiberChannel2, receiver3, receiver3, 1.22, "Evolution of the key rate with the distance_km for an entangled PDC source")

#graph_proba(0,25,source1, "Attenuated laser statistic")

#graph_proba(0,25,source3, "SMHPS statistic for 8 HS units")

#graph_proba(0,25,source4, "AMHPS statistic for 8 HS units")

#graph_proba(0,25,source6, "Entangled PDC source statistic")

##


#key_rate_distance_km_bb84(0,160,300,source1,detector1,FiberChannel1,f)

#key_rate_distance_km_bb84(0,160,300,source6,detector1,FiberChannel1,f)

intensities = [0.1, 0.2, 0.5, 0.7, 1]

hs_units = [2,4,8,32]

#key_rate_distance_km_bb84_mhps(0,160,300,0.48,hs_units,detector1,FiberChannel1,f)

#key_rate_distance_km_bb84_decoy_state(0,160,300,intensities,detector1,FiberChannel1,f)



#source2 = Multiplexed_Heralded_Photon_Source(0.1,32)

#key_rate_distance_km_bb84(0,160,300,source2,detector1,FiberChannel1,f)

distance_km1 = 40

distance_km2 = 10

#key_rate_intensity_attenuated_laser(0,1.2,200,detector1,FiberChannel1,f,distance_km1)

#key_rate_intensity_attenuated_laser_test(0,1.2,200,detector1,FiberChannel1,f,distance_km1)

#key_rate_intensity_attenuated_laser(0,0.1,200,detector1,FiberChannel1,f,distance_km2)

#key_rate_intensity_mhps(0,1.2,200,32,detector1,FiberChannel1,f,distance_km1)

#key_rate_hs_units_mhps(1,30,0.62,detector1,FiberChannel1,f,distance_km1)

## Values for tests
""" I put where the values are coming from and the results in the section 'Example' of the latex doc"""

# Plot

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

def key_rate_distance_km_bbm92(min, max, values_number, source: Source, detector1: Detector,detector2: Detector, FiberChannel1: FiberChannel,FiberChannel2: FiberChannel, receiver1: Receiver,receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BBM92(source, detector1,detector2, FiberChannel1,FiberChannel2, receiver1,receiver2, correction_efficiency, x,0)
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

def key_rate_loss_bb84(min, max, values_number, source: Source, detector: Detector, FiberChannel: FiberChannel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min*FiberChannel.loss_per_km, max*FiberChannel.loss_per_km, values_number)
    y1_values = []
    for x in x_values:
        protocol = BB84(source, detector, FiberChannel, receiver, correction_efficiency, x)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color = 'blue', label = "With active decoy state")
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_loss_bbm92(min, max, values_number, source: Source, detector1: Detector,detector2: Detector, FiberChannel1: FiberChannel,FiberChannel2: FiberChannel, receiver1: Receiver,receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    horiz_axis = np.linspace(min*FiberChannel1.loss_per_km, max*FiberChannel1.loss_per_km, values_number)
    y1_values = []
    for x in x_values:
        protocol = BBM92(source, detector1,detector2, FiberChannel1,FiberChannel2, receiver1,receiver2, correction_efficiency, x,0)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(horiz_axis, y1_values, color = 'red')
    plt.yscale('log')
    plt.xlabel("Loss in dB")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Test 1:

source_1 = Attenuated_Laser(0.48, 0)

detector_1 = Threshold_detector(0.17,5/100,10**(-5),0) #Y_0 = 1.7*10**(-6)

receiver_1 = Receiver(0.9)

FiberChannel_1 = FiberFiberChannel(0.21,0.934)

f_1 = 1.22

#key_rate_distance_km_bb84(0,160,300,source_1,detector_1,FiberChannel_1,receiver_1,f_1,"Evolution of the key rate with the distance_km for an attenuated laser")

#graph_proba(0,25,source_1, "Attenuated laser statistic")


# Test 2:

source_2 = Symmetric_Multiplexed_Heralded_Photon_Source(0.48,0,32, 0.5, 0.7)

detector_2 = Threshold_detector(20,0.25,10**(-8),0)

receiver_2 = Receiver(1)

FiberChannel_2 = FiberFiberChannel(0.2,0.99)

f_2 = 1.05

#key_rate_loss_bb84(0,275,300,source_2,detector_2,FiberChannel_2,receiver_2,f_2,"Evolution of the key rate with the loss for SMHPS")

#graph_proba(0,25,source_2, "SMHPS statistic")


# Test 3:

source_3 = Asymmetric_Multiplexed_Heralded_Photon_Source(0.6,0,32, 0.5, 0.7)

detector_3 = Threshold_detector(20,0.25,10**(-8),0)

receiver_3 = Receiver(1)

FiberChannel_3 = FiberFiberChannel(0.2,0.99)

f_3 = 1.05

#key_rate_loss_bb84(0,275,300,source_3,detector_3,FiberChannel_3,receiver_3,f_3,"Evolution of the key rate with the loss for AMHPS")

#graph_proba(0,25,source_3, "AMHPS statistic")

# Test 4:

source_4 = Entangled_PDC_Source(0.053, 0)

detector_4 = Threshold_detector(6.02,14.5/100,10**(-6),0)

receiver_4 = Receiver(1)

FiberChannel_4 = FiberFiberChannel(0.21,0.97)

f_4 = 1.22

key_rate_loss_bbm92(0,170,300, source_4, detector_4, detector_4, FiberChannel_4, FiberChannel_4, receiver_4, receiver_4, f_4, "Evolution of the key rate with the loss for an entangled PDC source")

#graph_proba(0,25,source_4, "Entangles PDC source statistic")