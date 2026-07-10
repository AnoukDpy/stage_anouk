from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import poisson

from source import Source
from source import Attenuated_Laser
from source import Multiplexed_Heralded_Photon_Source

from detector import Detector
from detector import Threshold_detector

## Functions

def binary_shannon_entropy(x):
    if x<=0 or x>=1:
        return 0
    return -x*math.log2(x)-(1-x)*math.log2(1-x)



## Channel

class Channel:

    def __init__(self, loss_coef: float, visibility: float):

        self.loss_coef = loss_coef
        self.visibility = visibility

    def transmittance(self, distance):

        return 10**(-self.loss_coef*distance/10)

    def probability_hitting_wrong_detector(self):

        return (1-self.visibility)/2



## Protocol

class Protocol:

    def __init__(self, source: Source, detector: Detector, channel: Channel, correction_efficiency: float, distance: float):

        self.source = source
        self.detector = detector
        self.channel = channel
        self.correction_efficiency = correction_efficiency
        self.distance = distance

    def transmittance_i_photon_state(self, i):

        return 1-(1-self.detector.efficiency*self.detector.transmittance*self.channel.transmittance(self.distance))**i

    def yield_i_photon_state(self, i):
        #probability for Bob to have a detection assuming that Alice sent an i-photon state
        return  self.detector.dark_count_probability() + self.transmittance_i_photon_state(i)

    def gain_i_photon_state(self, i):
        #probability for Alice to send an i-photon state and for Bob to have a detection
        return self.yield_i_photon_state(i)*self.source.probability_sending_i_photons(i)

    def overall_gain(self):
        gain = 0
        for i in range(0,50):
            gain += self.gain_i_photon_state(i)
        return gain


    def quantum_bit_error_rate(self,i):
        return (1/2 * self.detector.dark_count_probability() + self.channel.probability_hitting_wrong_detector() * self.transmittance_i_photon_state(i))/self.yield_i_photon_state(i)

    def overall_quantum_bit_error_rate(self):
        qber = 0
        for i in range(0,50):
            qber += (self.quantum_bit_error_rate(i)*self.yield_i_photon_state(i)*self.source.probability_sending_i_photons(i))
        qber = qber/self.overall_gain()
        return qber

    def key_rate_decoy_state_inf_key(self):
        return self.source.probability_sending_i_photons(0)*self.detector.dark_count_probability() + self.source.probability_sending_i_photons(1)*self.yield_i_photon_state(1)*(1-binary_shannon_entropy(self.quantum_bit_error_rate(1)))-self.overall_gain()*self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate())

    def key_rate_no_decoy_state_inf_key(self):

        delta = (1-self.source.probability_sending_i_photons(0)-self.source.probability_sending_i_photons(1))/self.overall_gain()

        return self.overall_gain()*((1-delta)*(1-binary_shannon_entropy(self.overall_quantum_bit_error_rate()/(1-delta)))-self.correction_efficiency*binary_shannon_entropy(self.overall_quantum_bit_error_rate()))

## Graphs

def key_rate_distance(min, max, values_number, source: Source, detector: Detector, channel: Channel, correction_efficiency: float):
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


def key_rate_distance_decoy_state(min, max, values_number, intensities, detector: Detector, channel: Channel, correction_efficiency: float):
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

def key_rate_distance_mhps(min, max, values_number, intensity, hs_units, detector: Detector, channel: Channel, correction_efficiency: float):
    for u in hs_units:
        source = Multiplexed_Heralded_Photon_Source(intensity, u )
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


## Calls

#source1 = Attenuated_Laser(0.48)

source1 = Attenuated_Laser(0.01)

detector1 = Threshold_detector(0.17,1/10,10**(-5),5/10) #Y_0 = 1.7*10**(-6)

channel1 = Channel(0.21,0.934) #Ma's values

f =1.22

#key_rate_distance(0,160,300,source1,detector1,channel1,f)

intensities = [0.1, 0.2, 0.5, 0.7, 1]

hs_units = [2,4,8,32]

key_rate_distance_mhps(0,160,300,0.48,hs_units,detector1,channel1,f)

#key_rate_distance_decoy_state(0,160,300,intensities,detector1,channel1,f)

source2 = Multiplexed_Heralded_Photon_Source(0.48,32)

#source2 = Multiplexed_Heralded_Photon_Source(0.1,32)

#key_rate_distance(0,160,300,source2,detector1,channel1,f)

distance1 = 40

distance2 = 10

#key_rate_intensity_attenuated_laser(0,1.2,200,detector1,channel1,f,distance1)

#key_rate_intensity_attenuated_laser(0,0.1,200,detector1,channel1,f,distance2)

#key_rate_intensity_mhps(0,1.2,200,32,detector1,channel1,f,distance1)

#key_rate_hs_units_mhps(1,30,0.62,detector1,channel1,f,distance1)
