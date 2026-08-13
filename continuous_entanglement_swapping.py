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

def polarizer_matrix(t_a,t_b):
    a = polarizer_partial_matrix(t_a)
    b = polarizer_partial_matrix(t_b)




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

        self.polarizer_1_matrix =

        self.polarizer_2_matrix =

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

        overall_matrix = self.photons_emitted_state

        a_x_1 = overall_matrix[2,1]
        b_x_1 = overall_matrix[2,2]
        c_x_1 = overall_matrix[4,4]
        d_x_1 = overall_matrix[4,7]

        a_p_1 = overall_matrix[10,9]
        b_p_1 = overall_matrix[10,10]
        c_p_1 = overall_matrix[12,12]
        d_p_1 = overall_matrix[12,15]

        sub_matrix_x_1 = np.array([[b_x_1,a_x_1],
                                   [d_x_1,c_x_1]])

        sub_matrix_p_1 = np.array([[b_p_1,a_p_1],
                                   [d_p_1,c_p_1]])

        a_x_2 = overall_matrix[3,0]
        b_x_2 = overall_matrix[3,3]
        c_x_2 = overall_matrix[5,5]
        d_x_2 = overall_matrix[5,6]

        a_p_2 = overall_matrix[11,8]
        b_p_2 = overall_matrix[11,11]
        c_p_2 = overall_matrix[13,13]
        d_p_2 = overall_matrix[13,14]

        sub_matrix_x_2 = np.array([[b_x_2,a_x_2],
                                   [d_x_2,c_x_2]])

        sub_matrix_p_2 = np.array([[b_p_2,a_p_2],
                                   [d_p_2,c_p_2]])

        bs_sub_matrix_x_1 = np.dot(self.beam_splitter_matrix.T,np.dot(sub_matrix_x_1,self.beam_splitter_matrix))

        bs_sub_matrix_x_2 = np.dot(self.beam_splitter_matrix.T,np.dot(sub_matrix_x_2,self.beam_splitter_matrix))

        bs_sub_matrix_p_1 = np.dot(self.beam_splitter_matrix.T,np.dot(sub_matrix_p_1,self.beam_splitter_matrix))

        bs_sub_matrix_p_2 = np.dot(self.beam_splitter_matrix.T,np.dot(sub_matrix_p_2,self.beam_splitter_matrix))

        overall_matrix[2,1] = bs_sub_matrix_x_1[0,1]
        overall_matrix[2,2] = bs_sub_matrix_x_1[0,0]
        overall_matrix[4,4] = bs_sub_matrix_x_1[1,1]
        overall_matrix[4,7] = bs_sub_matrix_x_1[1,0]

        overall_matrix[10,9] = bs_sub_matrix_p_1[0,1]
        overall_matrix[10,10] = bs_sub_matrix_p_1[0,0]
        overall_matrix[12,12] = bs_sub_matrix_p_1[1,1]
        overall_matrix[12,14] = bs_sub_matrix_p_1[1,0]

        overall_matrix[3,0] = bs_sub_matrix_x_2[0,1]
        overall_matrix[3,3] = bs_sub_matrix_x_2[0,0]
        overall_matrix[5,5] = bs_sub_matrix_x_2[1,1]
        overall_matrix[5,6] = bs_sub_matrix_x_2[1,0]

        overall_matrix[11,8] = bs_sub_matrix_p_2[0,1]
        overall_matrix[11,11] = bs_sub_matrix_p_2[0,0]
        overall_matrix[13,13] = bs_sub_matrix_p_2[1,1]
        overall_matrix[13,14] = bs_sub_matrix_p_2[1,0]

        return overall_matrix

    def polarizer_rotators(self):

        overall_matrix = self.beam_splitter_entanglement

        a_x_a = overall_matrix[0,0]
        b_x_a = overall_matrix[0,3]
        c_x_a = overall_matrix[1,1]
        d_x_a = overall_matrix[1,2]

        a_p_a = overall_matrix[8,8]
        b_p_a = overall_matrix[8,1]
        c_p_a = overall_matrix[9,9]
        d_p_a = overall_matrix[9,10]

        sub_matrix_x_a = np.array([[a_x_a,b_x_a],
                                   [c_x_a,d_x_a]])

        sub_matrix_p_1 = np.array([[a_p_a,b_p_a],
                                   [c_p_a,d_p_a]])

        a_x_d = overall_matrix[6,4]
        b_x_d = overall_matrix[6,7]
        c_x_d = overall_matrix[7,5]
        d_x_d = overall_matrix[7,6]

        a_p_d = overall_matrix[14,13]
        b_p_d = overall_matrix[14,14]
        c_p_d = overall_matrix[15,12]
        d_p_d = overall_matrix[15,15]

        sub_matrix_x_d = np.array([[a_x_d,b_x_d],
                                   [c_x_d,d_x_d]])

        sub_matrix_p_d = np.array([[a_p_d,b_p_d],
                                   [c_p_d,d_p_d]])

        bs_sub_matrix_x_a = np.dot(self.polarizer_1_matrix.T,np.dot(sub_matrix_x_a,self.polarizer_1_matrix))

        bs_sub_matrix_x_d = np.dot(self.polarizer_2_matrix.T,np.dot(sub_matrix_x_d,self.polarizer_2_matrix))

        bs_sub_matrix_p_a = np.dot(self.polarizer_1_matrix.T,np.dot(sub_matrix_p_a,self.polarizer_1_matrix))

        bs_sub_matrix_p_d = np.dot(self.polarizer_2_matrix.T,np.dot(sub_matrix_p_d,self.polarizer_2_matrix))

        overall_matrix[0,0] = bs_sub_matrix_x_a[0,0]
        overall_matrix[0,3] = bs_sub_matrix_x_a[0,1]
        overall_matrix[1,1] = bs_sub_matrix_x_a[1,0]
        overall_matrix[1,2] = bs_sub_matrix_x_a[1,1]

        overall_matrix[8,8] = bs_sub_matrix_p_a[0,0]
        overall_matrix[8,1] = bs_sub_matrix_p_a[0,1]
        overall_matrix[9,9] = bs_sub_matrix_p_a[1,0]
        overall_matrix[9,10] = bs_sub_matrix_p_a[1,1]

        overall_matrix[6,4] = bs_sub_matrix_x_d[0,0]
        overall_matrix[6,7] = bs_sub_matrix_x_d[0,1]
        overall_matrix[7,5] = bs_sub_matrix_x_d[1,0]
        overall_matrix[7,6] = bs_sub_matrix_x_d[1,1]

        overall_matrix[14,13] = bs_sub_matrix_p_d[0,0]
        overall_matrix[14,14] = bs_sub_matrix_p_d[0,1]
        overall_matrix[15,12] = bs_sub_matrix_p_d[1,0]
        overall_matrix[15,15] = bs_sub_matrix_p_d[1,1]

        return overall_matrix



































