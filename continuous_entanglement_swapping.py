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
import itertools as its

## functions

def f(x):
    return 2*x+1

def h(x):
    return 2*math.sqrt(x*(x+1))

def reduced_matrix(x,y):

    return np.array([[x,0,0,y],
                     [0,x,y,0],
                     [0,y,x,0],
                     [y,0,0,x]])

def beam_splitter_matrix(t):
    a = np.sqrt(t)
    b = np.sqrt(1-t)
    partial_matrix = np.array([[a,0,b,0],
                               [0,a,0,b],
                               [-b,0,a,0],
                               [0,-b,0,a]])

    one_quadrature = scipy.linalg.block_diag(np.eye(2), partial_matrix, np.eye(2))
    return np.kron(np.eye(2), one_quadrature)

def polarizer_partial_matrix(t):
    a = np.sqrt(t)
    b = np.sqrt(1-t)
    partial_matrix = np.array([[a,b],
                               [-b,a]])
    return partial_matrix

def polarizers_matrix(t_a,t_b):
    a = polarizer_partial_matrix(t_a)
    b = polarizer_partial_matrix(t_b)
    one_quadrature = scipy.linalg.block_diag(a, np.eye(2), np.eye(2), b)
    return np.kron(np.eye(2), one_quadrature)




## Protocol

class Continuous_Entanglement_swapping:

    def __init__(self,*, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver, polarizer_angle_1: float):


    def __init__(self,*, dimension: int, source_1: Source, channel_1: FiberChannel, detector_1: Detector, receiver_1: Receiver, polarizer_angle_1: float, receiver_2: Optional[Receiver] = None, receiver_3: Optional[Receiver] = None, receiver_4: Optional[Receiver] = None, detector_2: Optional[Detector] = None, detector_3: Optional[Detector] = None, detector_4: Optional[Detector] = None, source_2: Optional[Source] = None, channel_2: Optional[FiberChannel] = None, channel_3: Optional[FiberChannel] = None, channel_4: Optional[FiberChannel] = None, beam_splitter_angle: Optional[float] = None, polarizer_angle_2: Optional[float] = None):

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

        self.beam_splitter_matrix = beam_splitter_matrix(1/2)

        self.polarizers_matrix = polarizers_matrix(self.polarizer_angle_1,self.polarizer_angle_2)

        self.detectors = {0: self.detector_1, 2: self.detector_2, 5: self.detector_3, 6: self.detector_4}

    def photons_emitted_state(self):

        mu_1 = self.source_1.mean_photon_number
        mu_2 = self.source_2.mean_photon_number

        c_1 = f(mu_1)
        d_1 = h(mu_1)

        c_2 = f(mu_2)
        d_2 = h(mu_2)

        x_quadrature_1 = reduced_matrix(c_1,d_1)
        x_quadrature_2 = reduced_matrix(c_2,d_2)

        p_quadrature_1 = reduced_matrix(c_1,-d_1)
        p_quadrature_2 = reduced_matrix(c_2,-d_2)

        overall_matrix = scipy.linalg.block_diag(x_quadrature_1, x_quadrature_2, p_quadrature_1, p_quadrature_2)

        return overall_matrix

    def beam_splitter_entanglement(self):

        return np.dot(self.beam_splitter_matrix.T, np.dot(self.photons_emitted_state(), self.beam_splitter_matrix))


    def polarizer_rotators(self):

        return np.dot(self.polarizers_matrix.T, np.dot(self.beam_splitter_entanglement(), self.polarizers_matrix))


    def losses(self):

        n_a_1 = self.source_1.optical_efficiency()*self.detector_1.efficiency*self.channel_1.transmittance()*self.receiver_1.transmittance*self.receiver_1.x_basis_transmittance()

        n_a_2 = self.source_1.optical_efficiency()*self.detector_1.efficiency*self.channel_1.transmittance()*self.receiver_1.transmittance*self.receiver_1.z_basis_transmittance()

        n_b_1 = self.source_1.optical_efficiency()*self.detector_2.efficiency*self.channel_2.transmittance()*self.receiver_2.transmittance*self.receiver_2.x_basis_transmittance()

        n_b_2 = self.source_1.optical_efficiency()*self.detector_2.efficiency*self.channel_2.transmittance()*self.receiver_2.transmittance*self.receiver_2.z_basis_transmittance()

        n_c_1 = self.source_2.optical_efficiency()*self.detector_3.efficiency*self.channel_3.transmittance()*self.receiver_3.transmittance*self.receiver_3.x_basis_transmittance()

        n_c_2 = self.source_2.optical_efficiency()*self.detector_3.efficiency*self.channel_3.transmittance()*self.receiver_3.transmittance*self.receiver_3.z_basis_transmittance()

        n_d_1 = self.source_2.optical_efficiency()*self.detector_4.efficiency*self.channel_4.transmittance()*self.receiver_4.transmittance*self.receiver_4.x_basis_transmittance()

        n_d_2 = self.source_2.optical_efficiency()*self.detector_4.efficiency*self.channel_4.transmittance()*self.receiver_4.transmittance*self.receiver_4.z_basis_transmittance()

        k_loss_matrix = np.sqrt(np.kron(np.eye(2), np.diag(n_a_1, n_a_2, n_b_1, n_b_2, n_c_1, n_c_2, n_d_1, n_d_2)))

        alpha_loss_matrix = np.kron(np.eye(2), np.eye(8)-np.diag(n_a_1, n_a_2, n_b_1, n_b_2, n_c_1, n_c_2, n_d_1, n_d_2))
        return np.dot(k_loss_matrix.T, np.dot(self.polarizer_rotators(), k_loss_matrix))+alpha_loss_matrix


    def overall_coincidence_probability(self):

        M = self.losses()

        probability = 0

        measured_lines = [0,2,5,6]

        for k in range(0,5):

            combinations = its.combinations(measured_lines, k)
            dark_count_factor = (-2*(1-dark_count))**k
            partial_sum = 0

            for X in combinations:

                dark_count_factor = 1

                for y in X:
                    dark_count_factor = dark_count_factor*(-2*(1-self.detectors[y].back_ground_rate()))

                M_sub_x = M[np.ix_(X, X)]

                P = tuple(y + 10 for y in X)

                M_sub_p = M[np.ix_(P, P)]

                partial_sum =  partial_sum + dark_count_factor/np.sqrt(np.linalg.det(M_sub+np.eye(k)))

            probability = probability + partial_sum

        return probability




































