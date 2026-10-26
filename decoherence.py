import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.optimize import curve_fit

from simulation_tools import save_and_show


class Decoherence:
    """
    T1/T2 decoherence of a transmon, as Lindblad collapse operators on its
    n_levels Fock space (the same c_ops qutip-qip's DispersiveCavityQED
    builds for its qubits, but for the full Duffing ladder):

        relaxation:  sqrt(1/T1) * a          (level n decays at n/T1)
        dephasing:   sqrt(2/T_phi) * a_dag a (the 0-1 coherence decays at 1/T_phi,
                                              the n-m coherence at (n-m)^2/T_phi)

    with 1/T2 = 1/(2*T1) + 1/T_phi. Give T1 and either T2 or t_phi
    (t_phi=None with only T2 given is derived; T2 must be <= 2*T1).
    None for T1 / no dephasing means that channel is off. Times are in the
    same arbitrary units as the rest of the simulation.
    """

    def __init__(self, t1=None, t2=None, t_phi=None):
        if t2 is not None and t_phi is not None:
            raise ValueError("give either t2 or t_phi, not both")
        if t2 is not None:
            gamma_phi = 1.0 / t2 - (0.5 / t1 if t1 else 0.0)
            if gamma_phi < 0:
                raise ValueError("T2 cannot exceed 2*T1")
            t_phi = 1.0 / gamma_phi if gamma_phi > 0 else None
        self.t1 = t1
        self.t_phi = t_phi

    @property
    def t2(self):
        """Resulting coherence time, 1/T2 = 1/(2 T1) + 1/T_phi (None if neither channel is on)."""
        rate = (0.5 / self.t1 if self.t1 else 0.0) + (1.0 / self.t_phi if self.t_phi else 0.0)
        return 1.0 / rate if rate else None

    def c_ops(self, n_levels):
        ops = []
        if self.t1:
            ops.append(np.sqrt(1.0 / self.t1) * qt.destroy(n_levels))
        if self.t_phi:
            ops.append(np.sqrt(2.0 / self.t_phi) * qt.num(n_levels))
        return ops


def driven_hamiltonian(qubit, pulse, t0=0.0, rwa=False):
    """Same time-dependent Hamiltonian as TransmonQubit.apply_gate builds, and the pulse's time grid."""
    pulse.generate_drive_signal(qubit.omega_q, t0, qubit.alpha)
    t_vals = pulse.time_array(t0)
    if rwa:
        c, theta = pulse.complex_envelope, pulse.drive_phase
        H = [qubit.H0,
             [qubit.a_dag, 0.5 * np.conj(c) * np.exp(-1j * theta)],
             [qubit.a,     0.5 * c * np.exp(1j * theta)]]
    else:
        H = [qubit.H0, [qubit.a + qubit.a_dag, pulse.total_modulated_signal]]
    return H, t_vals


def apply_gate_open(qubit, pulse, decoherence, rho0=None, t0=0.0, rwa=False):
    """
    Open-system version of TransmonQubit.apply_gate: propagate the density
    matrix through `pulse` with qutip.mesolve under the same Hamiltonian plus
    the T1/T2 collapse operators of `decoherence`.

    rho0 may be a ket, a density matrix or None (ground state).
    Returns the time grid and the density matrix at every step, as an array of
    shape (len(t_vals), n_levels, n_levels); populations are the diagonals.
    """
    rho0 = qubit.ground_state if rho0 is None else rho0
    if not isinstance(rho0, qt.Qobj):
        rho0 = qt.Qobj(np.asarray(rho0, dtype=complex).reshape(-1, 1))
    H, t_vals = driven_hamiltonian(qubit, pulse, t0, rwa)
    result = qt.mesolve(H, rho0, t_vals, c_ops=decoherence.c_ops(qubit.n_levels),
                        options={'store_states': True, 'nsteps': 100000})
    return t_vals, np.array([state.full() for state in result.states])


def _fit_exponential(t, y):
    """Fit y = A*exp(-t/tau) + B and return (tau, fit curve)."""
    def model(t, A, tau, B):
        return A * np.exp(-t / tau) + B
    p0 = [y[0] - y[-1], t[-1] / 3, y[-1]]
    (A, tau, B), _ = curve_fit(model, t, y, p0=p0, maxfev=10000)
    return tau, model(t, A, tau, B)


def measure_t1(qubit, decoherence, t_max=None, n_points=200, save_as=None):
    """
    Simulated T1 experiment: start in |1>, evolve freely (H = 0: the frame rotating with H0, which leaves
    populations unchanged and keeps long times cheap) and fit the |1>
    population to exp(-t/T1). Returns (fitted T1, t, population).
    """
    t_max = t_max or 5 * decoherence.t1
    t = np.linspace(0, t_max, n_points)
    result = qt.mesolve(qt.qzero(qubit.n_levels), qt.basis(qubit.n_levels, 1), t,
                        c_ops=decoherence.c_ops(qubit.n_levels),
                        e_ops=[qt.projection(qubit.n_levels, 1, 1)])
    pop = np.real(result.expect[0])
    t1_fit, fit = _fit_exponential(t, pop)

    plt.plot(t, pop, label='simulation')
    plt.plot(t, fit, 'k--', label=f'fit: T1 = {t1_fit:.4g}')
    plt.xlabel('Time (arb. units)')
    plt.ylabel(r'$|1\rangle$ population')
    plt.title('T1 relaxation')
    plt.legend()
    plt.grid()
    save_and_show(save_as)
    return t1_fit, t, pop


def measure_t2(qubit, decoherence, t_max=None, n_points=200, save_as=None):
    """
    Simulated Ramsey-style T2 experiment: start in |+> = (|0>+|1>)/sqrt(2),
    evolve freely and fit the envelope |rho_01(t)| to exp(-t/T2). The envelope
    (rather than Re rho_01) removes the free precession at omega_q, as the
    rotating-frame demodulation does in an experiment, so H = 0 here too.
    Returns (fitted T2, t, |rho_01|).
    """
    t_max = t_max or 5 * decoherence.t2
    n = qubit.n_levels
    t = np.linspace(0, t_max, n_points)
    plus = (qt.basis(n, 0) + qt.basis(n, 1)).unit()
    result = qt.mesolve(qt.qzero(n), plus, t, c_ops=decoherence.c_ops(n),
                        e_ops=[qt.projection(n, 0, 1)])  # <|0><1|> = rho_01 (conj irrelevant)
    coherence = np.abs(result.expect[0])
    t2_fit, fit = _fit_exponential(t, coherence)

    plt.plot(t, coherence, label='simulation')
    plt.plot(t, fit, 'k--', label=f'fit: T2 = {t2_fit:.4g}')
    plt.xlabel('Time (arb. units)')
    plt.ylabel(r'$|\rho_{01}|$')
    plt.title('T2 dephasing (coherence envelope)')
    plt.legend()
    plt.grid()
    save_and_show(save_as)
    return t2_fit, t, coherence
