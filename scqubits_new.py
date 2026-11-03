import os
from simulation_tools import*
import numpy as np
import qutip as qt
import scqubits as scq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

scq.settings.PROGRESSBAR_DISABLED = False
scq.settings.T1_DEFAULT_WARNING = False
scq.set_units('GHz')  # energies/frequencies in GHz, coherence times in ns

TWO_PI = 2 * np.pi
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')



def make_transmon(ej=20.0, ec=0.25, ng=0.0, ncut=30, n_levels=4):
    """
    scqubits Transmon (cosine potential, charge basis).
    ej : float
        Josephson energy in GHz.
    ec : float
        Charging energy in GHz.
    ng : float
        Offset charge (0 = the charge-insensitive sweet spot).
    ncut : int
        Truncation level for the charge basis.
    n_levels : int
        Number of energy levels to consider.
    """
    scq_transmon = scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=ncut, truncated_dim=n_levels)
    return scq_transmon

def make_resonator(f_res=7.0, n_res=12):
    """scqubits Oscillator (the readout resonator).
    f_res : float
        Resonator frequency in GHz.
    n_res : int
        Truncation level for the resonator.
    """
    scq_resonator = scq.Oscillator(E_osc=f_res, truncated_dim=n_res)
    return scq_resonator

def Transmon_Resonator_Hamiltonian(tmon, resonator, g=0.1):
    """
    Transmon-resonator Hamiltonian in the dispersive regime (scqubits).
    tmon : scqubits.Transmon
        The transmon qubit.
    resonator : scqubits.Oscillator
        The readout resonator.
    g : float
        Coupling strength in GHz.
    """
    H = tmon.get_hamiltonian() + resonator.get_hamiltonian() + g * (qt.sigmap*resonator.a + qt.sigmam * resonator.a.dag())
    return H

