import copy
import os

import numpy as np
import qutip as qt
import scqubits as scq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from simulation_tools import TransmonQubit, XGate

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
    return scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=ncut, truncated_dim=n_levels)


def make_resonator(f_res=7.0, n_res=12):
    """scqubits Oscillator (the readout resonator).
    f_res : float
        Resonator frequency in GHz.
    n_res : int
        Truncation level for the resonator.
    """
    return scq.Oscillator(E_osc=f_res, truncated_dim=n_res)


def transmon_resonator(tmon, resonator, g):
    """
    Hamiltonian (qutip, GHz) of the transmon coupled to the resonator,
    H = H_transmon + H_resonator + g * n_transmon * (a + a_dag)
    H = H_q ⊗ 𝟙 + 𝟙 ⊗ H_r + g · n ⊗ (a + a†),
    written in the product of the two eigenbases (exact transmon energies and
    charge-operator matrix elements, so the coupling is not restricted to the
    qubit's two levels). 
    n_op is an operator counting the Cooper pairs that have tunnelled 
    across the Josephson junction.
    tmon : scqubits.Transmon
        The transmon object.
    resonator : scqubits.Oscillator
        The resonator object.
    g : float
        Coupling strength in GHz.
    """
    n_q, n_r = tmon.truncated_dim, resonator.truncated_dim
    n_op = qt.Qobj(tmon.matrixelement_table('n_operator', evals_count=n_q))
    a = qt.Qobj(resonator.annihilation_operator())  # Fock basis = eigenbasis
    h_q = qt.Qobj(np.diag(tmon.eigenvals(n_q)))
    h_r = qt.Qobj(np.diag(resonator.eigenvals(n_r)))
    return (qt.tensor(h_q, qt.qeye(n_r)) + qt.tensor(qt.qeye(n_q), h_r)
            + g * qt.tensor(n_op, a + a.dag()))


def dispersive_shift(H, n_q, n_r):
    """
    Dressed frequencies (GHz) from the eigenstates of `H` (as built by
    transmon_resonator), labelled by their largest overlap with the bare
    |qubit level, photon number>: 'f01' (resonator in |0>), 'f_res_0' /
    'f_res_1' (resonator frequency with the qubit in |0> / |1>) and the
    dispersive shift 'chi' = (f_res_1 - f_res_0) / 2.
    """
    evals, evecs = H.eigenstates() # Diagonalizes the entire coupled system Hamiltonian

    def dressed(j, k):
        bare = qt.tensor(qt.basis(n_q, j), qt.basis(n_r, k))
        return evals[int(np.argmax([abs(bare.overlap(v)) for v in evecs]))]

    e00, e10, e01, e11 = dressed(0, 0), dressed(1, 0), dressed(0, 1), dressed(1, 1)
    f_res_0, f_res_1 = e01 - e00, e11 - e10
    return {'f01': e10 - e00, 'f_res_0': f_res_0, 'f_res_1': f_res_1,
            'chi': 0.5 * (f_res_1 - f_res_0)}


def purcell_t1(g, delta, kappa):
    """
    Purcell-limited T1 (ns) of a qubit detuned by `delta` (GHz) from a
    resonator of linewidth `kappa` (GHz, angular rate/2pi) coupled with g (GHz):
    gamma = kappa * (g/delta)^2, in angular units. That is in the dispersive shift regime, 
    where the qubit-resonator detuning is much larger than the coupling strength. 
    """
    return 1.0 / (TWO_PI * kappa * (g / delta) ** 2)

# Read up to here

def coherence_times(tmon, noise_channels=None):
    """
    T1, T_phi and T2 of the 0<->1 transition from scqubits' noise models.
    All returned values are in ns (1 us = 1e3 ns), because of set_units('GHz').
    `noise_channels` picks channels by scqubits name (see
    tmon.supported_noise_channels()); by default every supported channel but
    't1_charge_impedance', which gives an unphysical T1 of ~0.2 ns for a
    transmon. 1/T2 = 1/(2 T1) + 1/T_phi.
    """
    supported = tmon.supported_noise_channels()
    if noise_channels is None:
        noise_channels = [c for c in supported if c != 't1_charge_impedance']
    wanted = [c for c in noise_channels if c in supported]
    t1_channels = [c for c in wanted if c.startswith('t1')]
    tphi_channels = [c for c in wanted if c.startswith('tphi')]

    t1 = tmon.t1_effective(noise_channels=t1_channels) if t1_channels else np.inf
    rate_phi = sum(1.0 / getattr(tmon, c)() for c in tphi_channels)
    t_phi = 1.0 / rate_phi if rate_phi else np.inf
    t2 = 1.0 / (0.5 / t1 + 1.0 / t_phi)
    return {'t1': t1, 't_phi': t_phi, 't2': t2}


T1_CHANNEL_INFO = {
    't1_capacitive':
        'dielectric loss: the qubit\'s electric field reaches lossy materials (surface oxides,\n'
        '      substrate, two-level-system defects in the interfaces), which absorb its energy.\n'
        '      Set by the capacitor\'s quality factor Q_cap (scqubits default ~1e6).',
    't1_charge_impedance':
        'loss into a 50 Ohm charge line: the qubit\'s charge couples to the impedance of a\n'
        '      drive/charge line and emits photons into it. scqubits assumes direct coupling\n'
        '      to the line, giving an unphysical ~0.1 ns for a transmon, so it is excluded\n'
        '      by default (real lines are coupled through a tiny capacitance).',
}


def t1_channel_report(tmon, noise_channels=None):
    """
    Lines for the report: T1 of each scqubits T1 channel of `tmon`, with a
    short explanation of its physical cause (T1_CHANNEL_INFO) and whether it
    is included in the effective T1 (same default selection as
    coherence_times).
    """
    supported = [c for c in tmon.supported_noise_channels() if c.startswith('t1')]
    if noise_channels is None:
        noise_channels = [c for c in supported if c != 't1_charge_impedance']
    lines = ['T1 noise channels (1/T1 = sum of the included rates):']
    for c in supported:
        status = 'included' if c in noise_channels else 'excluded'
        lines.append(f'  {c}: T1 = {getattr(tmon, c)() / 1e3:.4g} us ({status})')
        lines.append(f'      {T1_CHANNEL_INFO.get(c, "no description")}')
    return lines


class TransmonDynamics(TransmonQubit):
    """
    A scqubits Transmon as a simulation_tools TransmonQubit, so the pulses
    there (DrivePulse, XGate, YGate, RabiDrive) and apply_gate,
    plot_bloch_trajectory, fit_rabi, ... work on it unchanged. Only the
    Hamiltonian and the collapse operators change; the Duffing model is
    replaced by the exact scqubits spectrum, in the lab frame and in angular
    units (rad/ns, times in ns):

        H0 = sum_j 2 pi (E_j - E_0) |j><j|,    drive_op = n / |<0|n|1>|

    n is the charge operator in the eigenbasis, normalized so a pulse's area
    is still its rotation angle on 0<->1. T1 and pure dephasing are Lindblad
    operators (c_ops), so apply_gate runs mesolve and returns density matrices.
    """

    def __init__(self, tmon, t1=np.inf, t_phi=np.inf):
        """tmon: scqubits.Transmon; t1, t_phi: ns (np.inf = channel off)."""
        n = tmon.truncated_dim
        e = tmon.eigenvals(n)
        self.f01 = e[1] - e[0]  # GHz
        self.anharmonicity = (e[2] - e[1]) - self.f01  # GHz
        self.t1, self.t_phi = t1, t_phi
        self.t2 = 1.0 / (0.5 / t1 + 1.0 / t_phi)
        c_ops = []
        if np.isfinite(t1):
            c_ops += [np.sqrt(1.0 / t1) * qt.basis(n, j - 1) * qt.basis(n, j).dag()
                      * np.sqrt(j) for j in range(1, n)]
        if np.isfinite(t_phi):
            # dephasing of the 0-1 coherence at rate 1/t_phi
            c_ops.append(np.sqrt(0.5 / t_phi) * qt.Qobj(np.diag([-1, 1] + [0] * (n - 2))))
        super().__init__(omega_q=TWO_PI * self.f01, alpha=TWO_PI * self.anharmonicity,
                         n_levels=n, c_ops=c_ops)
        self._H0 = qt.Qobj(np.diag(TWO_PI * (e - e[0])))
        n_op = tmon.matrixelement_table('n_operator', evals_count=n)
        self._drive_op = qt.Qobj(n_op / abs(n_op[0, 1]))

    @property
    def H0(self):
        return self._H0

    @property
    def drive_op(self):
        return self._drive_op

    def free_evolution(self, rho0, times, f_frame=None):
        """
        Undriven mesolve from rho0 over `times`, in the frame rotating at
        f_frame (GHz, default f01): H = H0 - 2 pi f_frame * N. H0 is
        diagonal, so the frame only changes the phases of the coherences, not
        the populations, and it removes the fast lab-frame precession that
        would make microsecond-long evolutions slow. Returns the states.
        """
        f_frame = self.f01 if f_frame is None else f_frame
        H = self.H0 - TWO_PI * f_frame * qt.num(self.n_levels)
        return qt.mesolve(H, rho0, times, c_ops=self.c_ops,
                          options={'nsteps': 100000}).states

    def measure_t1(self, pulse=None, t_max=None, n_points=60):
        """
        Excite |0> with `pulse` (default XGate(pi)), decay freely and fit
        P1 = a exp(-t/T1). `t_max` in ns. Returns (t1_fit [ns], t [ns], p1).
        """
        pulse = XGate(np.pi) if pulse is None else pulse
        _, traj = self.apply_gate(pulse)
        t = np.linspace(0, t_max or 3 * self.t1, n_points)
        states = self.free_evolution(qt.Qobj(traj[-1]), t)
        p1 = np.array([np.real(s.full()[1, 1]) for s in states])
        popt, _ = curve_fit(lambda x, tau, a: a * np.exp(-x / tau), t, p1, p0=[self.t1, p1[0]])
        return popt[0], t, p1

    def measure_t2(self, pulse=None, t_max=None, detuning=None, n_points=500):
        """
        Ramsey: pulse - wait t - pulse, with `pulse` (default XGate(pi/2))
        driven `detuning` GHz above f01 (default: 20 fringes over the window);
        `t_max` in ns. The pulse is simulated once, in the lab frame with
        decoherence, as a superoperator S, and the wait is free evolution in
        the frame rotating at the drive frequency. In that frame the second
        pulse acts like the first (up to counter-rotating terms), so
            P1(t) = <1| S( U_wait(t)[ R S(|0><0|) R^dag ] ) |1>,
        with R moving the state after the first pulse into that frame.
        Fits P1 = b + a exp(-t/T2) cos(2 pi detuning t + phi) and returns
        (t2_fit [ns], t [ns], p1).
        """
        t_max = t_max or 3 * self.t2
        detuning = detuning or 20.0 / t_max
        pulse = XGate(np.pi / 2) if pulse is None else copy.copy(pulse)
        f_drive = self.f01 + detuning
        pulse.drive_frequency = TWO_PI * f_drive

        # pulse on [0, duration], so the drive carrier and the frame agree at t = 0
        H, t_pulse = self.driven_hamiltonian(pulse, t0=pulse.n_sigma * pulse.sigma)
        S = qt.propagator(qt.QobjEvo(H, tlist=t_pulse), t_pulse[-1], c_ops=self.c_ops,
                          options={'nsteps': 100000})
        S = S if S.issuper else qt.to_super(S)

        def apply_s(rho):
            return qt.vector_to_operator(S * qt.operator_to_vector(rho))

        R = (1j * TWO_PI * f_drive * t_pulse[-1] * qt.num(self.n_levels)).expm()
        rho1 = R * apply_s(qt.ket2dm(self.ground_state)) * R.dag()
        t = np.linspace(0, t_max, n_points)
        states = self.free_evolution(rho1, t, f_frame=f_drive)
        p1 = np.array([np.real(apply_s(s).full()[1, 1]) for s in states])
        fit = lambda x, tau, a, b, phi: b + a * np.exp(-x / tau) * np.cos(TWO_PI * detuning * x + phi)
        popt, _ = curve_fit(fit, t, p1, p0=[t_max / 3, 0.5, 0.5, 0.0])
        return popt[0], t, p1


def save_fig(name, dpi=150):
    """Save the current figure to DATA_DIR/name (.png) and close it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name if '.' in name else name + '.png')
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return path


def save_text(text, name):
    """Save text to DATA_DIR/name (.txt)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name if '.' in name else name + '.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


if __name__ == '__main__':
    # GHz
    ej = 30
    ec = 0.25
    f_res = 5.0
    kappa = 0.001
    g = 0.1

    tmon = make_transmon(ej, ec)
    resonator = make_resonator(f_res)
    disp = dispersive_shift(transmon_resonator(tmon, resonator, g),
                            tmon.truncated_dim, resonator.truncated_dim)
    times = coherence_times(tmon)
    dyn = TransmonDynamics(tmon, times['t1'], times['t_phi'])
    delta = f_res - dyn.f01

    t1_fit, t_t1, p_t1 = dyn.measure_t1()
    t2_fit, t_t2, p_t2 = dyn.measure_t2()

    x_gate = XGate(np.pi, sigma=5.0)
    t_pulse, traj = dyn.apply_gate(x_gate)
    pops = dyn.populations(traj)
    x_gate.frequency_spectrum(omega_q=dyn.f01, save_as='x_pi_frequency_spectrum')
    # x_gate.plot_bloch_trajectory()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t_pulse, pops[:, :3])
    axes[0].set(xlabel='t (ns)', ylabel='population', title='pi pulse (Gaussian, sigma = 5 ns)')
    axes[0].legend(['|0>', '|1>', '|2>'])
    axes[1].plot(t_t1 / 1e3, p_t1)
    axes[1].set(xlabel='t (us)', ylabel='P(|1>)', title='T1 decay')
    axes[2].plot(t_t2 / 1e3, p_t2)
    axes[2].set(xlabel='t (us)', ylabel='P(|1>)', title='Ramsey')
    plt.tight_layout()
    fig_path = save_fig('scq_dynamics')

    report = '\n'.join([
        f'Transmon: EJ = {ej} GHz, EC = {ec} GHz (EJ/EC = {ej / ec:.0f})',
        f'  f01 = {dyn.f01:.4f} GHz, anharmonicity = {dyn.anharmonicity * 1e3:.1f} MHz',
        '',
        'Coherence (scqubits noise models):',
        f'  T1 = {times["t1"] / 1e3:.2f} us, T_phi = {times["t_phi"] / 1e3:.2f} us, '
        f'T2 = {times["t2"] / 1e3:.2f} us',
        f'  simulated fits: T1 = {t1_fit / 1e3:.2f} us, T2 = {t2_fit / 1e3:.2f} us',
        '',
        *t1_channel_report(tmon),
        '',
        'pi pulse final populations |0>,|1>,|2>,...:',
        f'  {np.round(pops[-1], 5)}',
        '',
        f'Resonator: f_res = {f_res} GHz, g = {g * 1e3:.0f} MHz, kappa = {kappa * 1e3:.1f} MHz',
        f'  detuning = {delta * 1e3:.0f} MHz, dressed f01 = {disp["f01"]:.4f} GHz',
        f'  f_res(|0>) = {disp["f_res_0"]:.5f} GHz, f_res(|1>) = {disp["f_res_1"]:.5f} GHz',
        f'  dispersive shift chi = {disp["chi"] * 1e3:.2f} MHz',
        f'  Purcell-limited T1 = {purcell_t1(g, delta, kappa) / 1e3:.1f} us',
        '      Purcell decay: the qubit is slightly hybridized with the resonator, so it',
        '      leaks energy out through the resonator\'s output line at kappa (g/delta)^2.',
        '      Not included in the T1 above (not a scqubits noise channel).',
        ''])
    print(report)
    print('report saved to', save_text(report, 'scq_report'))
    print('figure saved to', fig_path)
