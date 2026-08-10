import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum
import math
from scipy.stats import poisson
from typing import Optional
from scipy.integrate import quad
from typing import Callable
from scipy.special import voigt_profile
from itertools import product


## Functions

def conditionnal_detection_probability(no_dark_count_proba, channel_efficiency, x_value, n_photons_sent):
    if x_value == 0:
        return no_dark_count_proba*(1-channel_efficiency*no_dark_count_proba)**n_photons_sent

    else:
        return 1-(no_dark_count_proba*(1-channel_efficiency*no_dark_count_proba)**n_photons_sent)

##

class Entanglement_swapping:

    def __init__(self,*, dimension: int, source_1: Source, channel_1: Channel, detector_1: Detector, receiver_1: Receiver, , polarizer_angle_1: float, receiver_2: Optional[Receiver] = None, receiver_3: Optional[Receiver] = None, receiver_4: Optional[Receiver] = None, detector_2: Optional[Detector] = None, detector_3: Optional[Detector] = None, detector_4: Optional[Detector] = None, source_2: Optionnal[Source] = None, channel_2: Optional[Channel] = None, channel_3: Optional[Channel] = None, channel_4: Optional[Channel] = None, beam_splitter_angle: Optionnal[float] = None, polarizer_angle_2: Optionnal[float] = None):

        self.dimension = dimension
        self.source_1 = source_1
        self.channel_1 = channel_1
        self.detector_1 = detector_1
        self.receiver_1 = receiver_1
        self.polarizer_angle_1 = polarizer_angle_1

        if receiver_2 is None:
            self.receiver_2 = receiver_1

        else:
            self.receiver_2 = receiver_2

        if receiver_3 is None:
            self.receiver_3 = receiver_1

        else:
            self.receiver_3 = receiver_3

        if receiver_4 is None:
            self.receiver_4 = receiver_1

        else:
            self.receiver_4 = receiver_4

        if detector_2 is None:
            self.detector_2 = detector_1

        else:
            self.detector_2 = detector_2

        if detector_3 is None:
            self.detector_3 = detector_1

        else:
            self.detector_3 = detector_3

        if detector_4 is None:
            self.detector_4 = detector_1

        else:
            self.detector_4 = detector_4

        if source_2 is None:
            self.source_2 = source_1

        else:
            self.source_2 = source_2

        if channel_2 is None:
            self.channel_2 = channel_1

        else:
            self.channel_2 = channel_2

        if channel_3 is None:
            self.channel_3 = channel_1

        else:
            self.channel_3 = channel_3

        if channel_4 is None:
            self.channel_4 = channel_1

        else:
            self.channel_4 = channel_4

        if beam_splitter_angle is None:
            self.beam_splitter_angle = np.pi/4

        else:
            self.beam_splitter_angle = beam_splitter_angle

        if polarizer_angle_2 is None:
            self.polarizer_angle_2 = polarizer_angle_1

        else:
            self.polarizer_angle_2 = polarizer_angle_2


        ## Quantum definitions

    def entangled_photons_state(self, i):

        if i == 1:
            source = self.source_1

        elif i == 2:
            source = self.source_2
        else:
            raise ValueError(f"i must be 1 or 2, got {value}")

        N = self.dimension

        creation_a_h = qt.tensor(qt.create(N), qt.qeye(N), qt.qeye(N), qt.qeye(N)
        creation_a_v = qt.tensor(qt.qeye(N), qt.create(N), qt.qeye(N), qt.qeye(N))
        creation_b_h = qt.tensor(qt.qeye(N), qt.qeye(N), qt.create(N), qt.qeye(N))
        creation_b_v = qt.tensor(qt.qeye(N), qt.qeye(N), qt.qeye(N), qt.create(N))

        identity = qt.qeye(N**4)

        vac = qt.tensor(qt.fock(N, 0), qt.fock(N, 0), qt.fock(N, 0), qt.fock(N, 0))

        exponent = 1j*np.tanh(source.multi_pair_production_rate)*(creation_a_h*creation_b_h+creation_a_v*creation_b_v)

        return ((1/np.cosh(source.multi_pair_production_rate)**2)*exponent.expm())*vac

    def beam_splitter_operator(self):

        N = self.dimension

        anihilation_b_h = qt.tensor(qt.destroy(N), qt.qeye(N), qt.qeye(N), qt.qeye(N)
        anihilation_b_v = qt.tensor(qt.qeye(N), qt.destroy(N), qt.qeye(N), qt.qeye(N))
        anihilation_c_h = qt.tensor(qt.qeye(N), qt.qeye(N), qt.destroy(N), qt.qeye(N))
        anihilation_c_v = qt.tensor(qt.qeye(N), qt.qeye(N), qt.qeye(N), qt.destroy(N))

        generator = self.beam_splitter_angle*((anihilation_b_h.dag()*anihilation_c_h - anihilation_c_h.dag()*anihilation_b_h)+(anihilation_b_v.dag()*anihilation_c_v - anihilation_c_v.dag()*anihilation_b_v))

        return qt.tensor(qt.qeye(N**2),generator.expm(),qt.qeye(N**2))

    def beam_splitter_outing(self):

        return self.beam_splitter_operator()*qt.tensor(entangled_photons_state(1), entangled_photons_state(2))

    def heralding_probability(self, i, j, k, l):
        N = self.dimension

        ket_ijkl = qt.tensor(qt.fock(N,i), qt.fock(N,j), qt.fock(N,k), qt.fock(N,l))
        bra_ijkl = ket_ijkl.dag()

        ijkl_projector = qt.tensor(qt.qeye(N**2), ket_ijkl*bra_ijkl, qt.qeye(N**2))

        return beam_splitter_outing().dag()*ijkl_projector*beam_splitter_outing()

    def b_c_bell_measurement(self, i, j, k, l):

        N = self.dimension

        ket_ijkl = qt.tensor(qt.fock(N,i), qt.fock(N,j), qt.fock(N,k), qt.fock(N,l))
        bra_ijkl = ket_ijkl.dag()

        ijkl_projector = qt.tensor(qt.qeye(N**2), ket_ijkl*bra_ijkl, qt.qeye(N**2))

        partial_dot_product_matrix = qt.tensor(qt.qeye(N**2), bra_ijkl, qt.qeye(N**2))

        return partial_dot_product_matrix*ijkl_projector*self.beam_splitter_outing()/np.sqrt(heralding_probability(i, j, k, l))

    def entrance_knowing_detection_probability(self, q, r, s, t, i, j, k, l):

        N = self.dimension

        no_back_ground_rate_2 = 1-detector_2.back_ground_rate()

        no_back_ground_rate_3 = 1-detector_3.back_ground_rate()

        efficiency_2_x = self.detector_2.efficiency*self.receiver_2.x_basis_transmittance()*self.channel_2.transmittance()*self.source_1.optical_efficiency()

        efficiency_2_z = self.detector_2.efficiency*self.receiver_2.z_basis_transmittance()*self.channel_2.transmittance()*self.source_1.optical_efficiency()

        efficiency_3_x = self.detector_3.efficiency*self.receiver_3.x_basis_transmittance()*self.channel_3.transmittance()*self.source_2.optical_efficiency()

        efficiency_3_z = self.detector_3.efficiency*self.receiver_3.z_basis_transmittance()*self.channel_3.transmittance()*self.source_2.optical_efficiency()

        probability_qrst_detection = sum(conditionnal_detection_probability(no_back_ground_rate_2, efficiency_2_x, q, n) * conditionnal_detection_probability(no_back_ground_rate_2, efficiency_2_z, r, m) * conditionnal_detection_probability(no_back_ground_rate_3, efficiency_3_x, s, o) * conditionnal_detection_probability(no_back_ground_rate_3, efficiency_3_z, t, p)*self.heralding_probability(n, m, o, p) for n in range(N) for m in range(N) for o in range(N) for p in range(N))

        return (conditionnal_detection_probability(no_back_ground_rate_2, efficiency_2_x, q, i) * conditionnal_detection_probability(no_back_ground_rate_2, efficiency_2_z, r, j) * conditionnal_detection_probability(no_back_ground_rate_3, efficiency_3_x, s, k) * conditionnal_detection_probability(no_back_ground_rate_3, efficiency_3_z, t, l)*)/probability_qrst_detection

    def mixed_state_density_matrix(self, q, r, s, t):

        N = self.dimension

        return sum(entrance_knowing_detection_probability(q, r, s, t, n, m, o, p)*b_c_bell_measurement(n, m, o, p)*b_c_bell_measurement(n, m, o, p).dag() for n in range(N) for m in range(N) for o in range(N) for p in range(N))

    def polarizers_action(self):

        anihilation_a_h = qt.tensor(qt.destroy(N), qt.qeye(N), qt.qeye(N), qt.qeye(N)
        anihilation_a_v = qt.tensor(qt.qeye(N), qt.destroy(N), qt.qeye(N), qt.qeye(N))
        anihilation_d_h = qt.tensor(qt.qeye(N), qt.qeye(N), qt.destroy(N), qt.qeye(N))
        anihilation_d_v = qt.tensor(qt.qeye(N), qt.qeye(N), qt.qeye(N), qt.destroy(N))

        generator_a = self.polarizer_angle_1*((anihilation_a_h.dag()*anihilation_a_v - anihilation_a_v.dag()*anihilation_a_h)

        generator_d = self.polarizer_angle_2*((anihilation_d_h.dag()*anihilation_d_v - anihilation_d_v.dag()*anihilation_d_h)

        return generator_a.expm()*generator_d.expm()

    def last_detectors_entrance_probability(self, n, m, o, p, q, r, s, t):

        ket_nmop = qt.tensor(qt.fock(N,n), qt.fock(N,m), qt.fock(N,o), qt.fock(N,p))
        bra_nmop = ket_nmop.dag()

        return bra_nmop*polarizers_action()*mixed_state_density_matrix(self, q, r, s, t)*polarizers_action().dag()*ket_nmop

    def outer_detection_probability_given_inner_results(self, u, v, w, x, q, r, s, t):

        N = self.dimension

        no_back_ground_rate_1 = 1-detector_1.back_ground_rate()

        no_back_ground_rate_4 = 1-detector_4.back_ground_rate()

        efficiency_1_x = self.detector_1.efficiency*self.receiver_1.x_basis_transmittance()*self.channel_1.transmittance()*self.source_1.optical_efficiency()

        efficiency_1_z = self.detector_1.efficiency*self.receiver_1.z_basis_transmittance()*self.channel_1.transmittance()*self.source_1.optical_efficiency()

        efficiency_4_x = self.detector_4.efficiency*self.receiver_4.x_basis_transmittance()*self.channel_4.transmittance()*self.source_2.optical_efficiency()

        efficiency_4_z = self.detector_4.efficiency*self.receiver_4.z_basis_transmittance()*self.channel_4.transmittance()*self.source_2.optical_efficiency()

        probability_uvwx_qrst_detection = sum(conditionnal_detection_probability(no_back_ground_rate_1, efficiency_1_x, u, n) * conditionnal_detection_probability(no_back_ground_rate_1, efficiency_1_z, v, m) * conditionnal_detection_probability(no_back_ground_rate_4, efficiency_4_x, w, o) * conditionnal_detection_probability(no_back_ground_rate_4, efficiency_4_z, x, p) * last_detectors_entrance_probability(n, m, o, p, q, r, s, t) for n in range(N) for m in range(N) for o in range(N) for p in range(N))

    def max_outer_detection_probability_given_inner_results(self, q, r, s, t):

        combinations = list(product([0, 1], repeat=4))

        possible_probabilities = [outer_detection_probability_given_inner_results(self, u, v, w, x, q, r, s, t) for u, v, w, x in combinaisons]

        return max(possible_probabilities)

    def min_outer_detection_probability_given_inner_results(self, q, r, s, t):

        combinations = list(product([0, 1], repeat=4))

        possible_probabilities = [outer_detection_probability_given_inner_results(self, u, v, w, x, q, r, s, t) for u, v, w, x in combinaisons]

        return min(possible_probabilities)

    def visibility(self, q, r, s, t):

        return (max_outer_detection_probability_given_inner_results(q, r, s, t)-min_outer_detection_probability_given_inner_results(q, r, s, t))/(max_outer_detection_probability_given_inner_results(q, r, s, t)+min_outer_detection_probability_given_inner_results(q, r, s, t))































































