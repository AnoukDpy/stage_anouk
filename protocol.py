from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson


## Functions

def binary_shannon_entropy(x):
    if x<=0 or x>=1:
        return 0
    return -x*math.log2(x)-(1-x)*math.log2(1-x)


## Protocol

class Protocol(ABC):

    def __init__(self, source: Source, detector: Detector, channel: Channel, receiver: Receiver, correction_efficiency: float, distance: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.distance = distance

class BB84(Protocol):

    def __init__(self, source: Source, detector: Detector, channel: Channel, receiver: Receiver, correction_efficiency: float, distance: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.receiver = receiver
        self.correction_efficiency = correction_efficiency
        self.distance = distance

    def transmittance_i_photon_state(self, i):

        return 1-(1-self.detector.efficiency*self.receiver.transmittance*self.channel.transmittance(self.distance))**i

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
        return (1/2 * self.detector.background_rate() + (self.channel.probability_hitting_wrong_detector()+1/2 *self.detector.after_pulsing ) * self.transmittance_i_photon_state(i))/self.yield_i_photon_state(i)

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


class BB92(Protocol):

    def __init__(self, source: Source, detector1: Detector, detector2: Detector, channel1: Channel, channel2: Channel, receiver1: Receiver, receiver2: Receiver, correction_efficiency: float, distance1: float, distance2: float):

        self.source = source
        self.detector1 = detector1
        self.detector2 = detector2
        self.channel1 = channel1
        self.channel2 = channel2
        self.receiver1 = receiver1
        self.receiver2 = receiver2
        self.correction_efficiency = correction_efficiency
        self.distance1 = distance1
        self.distance2 = distance2

    def transmittance_i_photon_state1(self, i):

        return 1-(1-self.detector1.efficiency*self.receiver1.transmittance*self.channel1.transmittance(self.distance1)*(1+self.detector1.after_pulsing))**i

    def transmittance_i_photon_state2(self, i):

        return 1-(1-self.detector2.efficiency*self.receiver2.transmittance*self.channel2.transmittance(self.distance2)*(1+self.detector2.after_pulsing))**i

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

        return 1/2-((1/2-((self.channel1.probability_hitting_wrong_detector()+self.channel2.probability_hitting_wrong_detector()+(self.detector1.after_pulsing+self.detector2.after_pulsing)/4)/(1+(self.detector1.after_pulsing+self.detector2.after_pulsing)/2)))/self.yield_i_photon_state(n))*(-self.transmittance_i_photon_state1(n-m)+self.transmittance_i_photon_state1(m))*(-self.transmittance_i_photon_state2(n-m)+self.transmittance_i_photon_state2(m))


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


## Graphs
"""
def key_rate_distance_bb84(min, max, values_number, source: Source, detector: Detector, channel: Channel, correction_efficiency: float):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    y2_values = []
    for x in x_values:
        protocol = Protocol(source, detector, channel, correction_efficiency, x)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

        y2 = protocol.key_rate_no_decoy_state_inf_key()
        y2_values.append(y2)

    plt.plot(x_values, y1_values, color = 'blue', label = "With active decoy state")
    plt.plot(x_values, y2_values, color = 'red', label = "Without decoy state")
   # plt.yscale('log')
    plt.xlabel("Distance in km")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the distance")
    plt.legend()
    plt.grid(True)
    plt.show()
"""
def key_rate_distance_bb84(min, max, values_number, source: Source, detector: Detector, channel: Channel, receiver: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BB84(source, detector, channel, receiver, correction_efficiency, x)
        y1 = protocol.key_rate_decoy_state_inf_key()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color = 'blue', label = "With active decoy state")
    plt.yscale('log')
    plt.xlabel("Distance in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_bb92(min, max, values_number, source: Source, detector1: Detector,detector2: Detector, channel1: Channel,channel2: Channel, receiver1: Receiver,receiver2: Receiver, correction_efficiency: float, title: str):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        protocol = BB92(source, detector1,detector2, channel1,channel2, receiver1,receiver2, correction_efficiency, x,0)
        y1 = protocol.key_rate()
        y1_values.append(y1)

    plt.plot(x_values, y1_values, color = 'red')
    plt.yscale('log')
    plt.xlabel("Distance in km")
    plt.ylabel("Key rate in bpp")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def key_rate_distance_bb84_decoy_state(min, max, values_number, intensities, detector: Detector, channel: Channel, correction_efficiency: float):
    for i in intensities:
        source = Attenuated_Laser(i)
        x_values = np.linspace(min, max, values_number)
        y1_values = []
        for x in x_values:
            protocol = Protocol(source, detector, channel, correction_efficiency, x)
            y1 = protocol.key_rate_decoy_state_inf_key()
            y1_values.append(y1)
        plt.plot(x_values, y1_values, label = f"Source's intensity = {i}")
    plt.yscale('log')
    plt.xlabel("Distance in km")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the distance")
    plt.legend()
    plt.grid(True)
    plt.show()

def key_rate_distance_bb84_mhps(min, max, values_number, intensity, hs_units, detector: Detector, channel: Channel, correction_efficiency: float):
    for u in hs_units:
        source = Multiplexed_Heralded_Photon_Source(intensity, u )
        x_values = np.linspace(min, max, values_number)
        y1_values = []
        for x in x_values:
            protocol = Protocol(source, detector, channel, correction_efficiency, x)
            y1 = protocol.key_rate_decoy_state_inf_key()
            y1_values.append(y1)
        plt.plot(x_values, y1_values, label = f"HS units = {u}")
    plt.yscale('log')
    plt.xlabel("Distance in km")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the distance")
    plt.legend()
    plt.grid(True)
    plt.show()


def key_rate_intensity_attenuated_laser(min, max, values_number, detector: Detector, channel: Channel, correction_efficiency: float, distance: float):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    y2_values = []
    for x in x_values:
        source = Attenuated_Laser(x)
        protocol = Protocol(source, detector, channel, correction_efficiency, distance)
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

def key_rate_intensity_attenuated_laser_test(min, max, values_number, detector: Detector, channel: Channel, correction_efficiency: float, distance: float):
    x_values = np.linspace(min, max, values_number)
    y1_values = []
    for x in x_values:
        source = Attenuated_Laser(x)
        protocol = Protocol(source, detector, channel, correction_efficiency, distance)
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

def key_rate_intensity_mhps(min, max, values_number, sources_num: float, detector: Detector, channel: Channel, correction_efficiency: float, distance: float):
    x_values = np.linspace(min, max, values_number)
    y_values = []
    for x in x_values:
        source = Multiplexed_Heralded_Photon_Source(x, sources_num)
        protocol = Protocol(source, detector, channel, correction_efficiency, distance)
        y = protocol.key_rate_decoy_state_inf_key()
        y_values.append(y)

    plt.plot(x_values, y_values)
    #plt.yscale('log')
    plt.xlabel("Source intensity")
    plt.ylabel("Key rate in bpp")
    plt.title("Evolution of the key rate with the source's intensity")
    plt.grid(True)
    plt.show()

def key_rate_hs_units_mhps(min, max, intensity: float, detector: Detector, channel: Channel, correction_efficiency: float, distance: float):
    x_values = np.arange(min,max+1)
    y_values = []
    for x in x_values:
        source = Multiplexed_Heralded_Photon_Source(intensity, x)
        protocol = Protocol(source, detector, channel, correction_efficiency, distance)
        y = protocol.key_rate_decoy_state_inf_key()
        y_values.append(y)

    plt.plot(x_values, y_values, 'x')
    #plt.yscale('log')
    plt.xlabel("HS units")
    plt.ylabel("Key rate")
    plt.title("Evolution of the key rate with the HS units")
    plt.grid(True)
    plt.show()



"""Key rate evolution with the distance:"""

## Sources

source1 = Attenuated_Laser(0.48, 0)

source2 = Multiplexed_Heralded_Photon_Source(0.48,0,32)

source3 = Symmetric_Multiplexed_Heralded_Photon_Source(0.48,0,8, 0.2, 0.1)

source4 = Asymmetric_Multiplexed_Heralded_Photon_Source(0.4,0,8, 0.2, 0.1)

source5 = Single_Photon_Source(0, 80/100, 1/100)

source6 = Entangled_PDC_Source(0.053, 0)

## Channels, error corrections and receivers

channel1 = Channel(0.21,0.934) #Ma's values

channel2 = Channel(0.21,0.97)

f =1.22

receiver1 = Receiver(5/10)
receiver2 = Receiver(14/100)
receiver3 = Receiver(1)
## Detectors

detector1 = Threshold_detector(0.17,1/10,10**(-5),0.8/100) #Y_0 = 1.7*10**(-6)

detector2 = Threshold_detector(0.17,1/10,10**(-5),0) #without after pulsing

detector3 = Threshold_detector(6.02,14.5/100,10**(-6),0) #without after pulsing

## Calls

#key_rate_distance_bb84(0,160,300,source1,detector1,channel1,receiver1,f,"Evolution of the key rate with the distance for an attenuated laser")

#key_rate_distance_bb84(0,160,300,source2,detector1,channel1,receiver1,f,"Evolution of the key rate with the distance for a MHPS")

#key_rate_distance_bb84(0,160,300,source3,detector2,channel1,receiver1,f,"Evolution of the key rate with the distance for a SMHPS")

#key_rate_distance_bb84(0,160,300,source4,detector1,channel1,receiver1,f,"Evolution of the key rate with the distance for an AMHPS")

#key_rate_distance_bb84(0,160,300,source5,detector1,channel1,receiver1,f,"Evolution of the key rate with the distance for a single photon source")

key_rate_distance_bb92(0,160,300, source6, detector3, detector3, channel2, channel2, receiver3, receiver3, 1.22, "Evolution of the key rate with the distance for an entangled PDC source")

##


#key_rate_distance_bb84(0,160,300,source1,detector1,channel1,f)

#key_rate_distance_bb84(0,160,300,source6,detector1,channel1,f)

intensities = [0.1, 0.2, 0.5, 0.7, 1]

hs_units = [2,4,8,32]

#key_rate_distance_bb84_mhps(0,160,300,0.48,hs_units,detector1,channel1,f)

#key_rate_distance_bb84_decoy_state(0,160,300,intensities,detector1,channel1,f)



#source2 = Multiplexed_Heralded_Photon_Source(0.1,32)

#key_rate_distance_bb84(0,160,300,source2,detector1,channel1,f)

distance1 = 40

distance2 = 10

#key_rate_intensity_attenuated_laser(0,1.2,200,detector1,channel1,f,distance1)

#key_rate_intensity_attenuated_laser_test(0,1.2,200,detector1,channel1,f,distance1)

#key_rate_intensity_attenuated_laser(0,0.1,200,detector1,channel1,f,distance2)

#key_rate_intensity_mhps(0,1.2,200,32,detector1,channel1,f,distance1)

#key_rate_hs_units_mhps(1,30,0.62,detector1,channel1,f,distance1)

