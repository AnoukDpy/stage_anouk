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


class Entanglement_swapping:

    def __init__(self,*, dimension: int, source_1: Source, channel_1: Channel, detector_1: Detector, detector_2: Optionnal[Detector] = None, source_2: Optionnal[Source] = None, channel_2: Optional[Channel] = None, channel_3: Optional[Channel] = None, channel_4: Optional[Channel] = None):

        self.dimension = dimension
        self.source_1 = source_1
        self:channel_1 = channel_1
        self.detector_1 = detector_1

        if detector_2 is None:
            self.detector_2 = detector_1

        else:
            self.detector_2 = detector_2

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

        vac = qt.tensor(qt.fock(N, 0), qt.fock(N, 0), qt.fock(N, 0), qt.fock(N, 0))

        exponent = 1j*np.tanh(source.multi_pair_production_rate)*(creation_a_h*creation_b_h+creation_a_v*creation_b_v)

        return (1/np.cosh(source.multi_pair_production_rate)**2)*exponent.expm()



























