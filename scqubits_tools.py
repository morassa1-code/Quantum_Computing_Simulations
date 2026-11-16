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

T1_CHANNEL_INFO = {
    't1_capacitive':
        'dielectric loss: the qubit\'s electric field reaches lossy materials (surface oxides,\n'
        '      substrate, two-level-system defects in the interfaces), which absorb its energy.\n'
        '      Set by the capacitor\'s quality factor Q_cap (scqubits default ~1e6).',
    't1_charge_impedance':
        'loss into a 50 Ohm charge line: the qubit\'s charge couples to the impedance of a\n'
        '      drive/charge line and emits photons into it. scqubits assumes direct coupling\n'
        '      to the line, which gives an unphysically short T1 for a transmon\n'
        '      (real lines are coupled through a tiny capacitance).',
}

T_PHI_CHANNEL_INFO = {
    'tphi_1_over_f_cc':
        '1/f dephasing time (or rate) due to critical current noise.\n',
    'tphi_1_over_f_ng':
        '1/f dephasing time (or rate) due to charge noise\n'
}


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
    is still its rotation angle on 0<->1. T1 and pure dephasing (from the
    scqubits noise models, see `coherence_times`) are Lindblad operators
    (c_ops), so apply_gate runs mesolve and returns density matrices.

    Optionally the qubit (its 0-1 transition, f01) is coupled to a readout
    resonator; the `readout_*`, `chi`, `n_crit` and `purcell_t1` members then
    describe the Jaynes-Cummings model and its dispersive approximation.
    """

    # ------------------------------------------------------------------ setup

    def __init__(self, tmon, f_res=None, g=None, kappa=None, n_res=12, noise_channels=None):
        """
        tmon : scqubits.Transmon
        f_res, g, kappa : readout resonator frequency, qubit-resonator coupling
            and linewidth (all GHz). Only needed for the readout methods.
        n_res : resonator Fock-space truncation.
        noise_channels : scqubits noise-channel names to include (default: all
            the transmon supports, see tmon.supported_noise_channels()).
        """
        n = tmon.truncated_dim
        e = tmon.eigenvals(n)
        self.tmon = tmon
        self.f_res = f_res
        self.g = g
        self.kappa = kappa
        self.n_res = n_res
        self.f01 = e[1] - e[0]  # GHz
        self.anharmonicity = (e[2] - e[1]) - self.f01  # GHz
        self.noise_channels = self._select_channels(noise_channels)
        self.t1, self.t_phi, self.t2 = self.coherence_times()

        c_ops = []
        if np.isfinite(self.t1):
            c_ops += [np.sqrt(j / self.t1) * qt.basis(n, j - 1) * qt.basis(n, j).dag()
                      for j in range(1, n)]
        if np.isfinite(self.t_phi):
            # dephasing of the 0-1 coherence at rate 1/t_phi
            c_ops.append(np.sqrt(0.5 / self.t_phi) * qt.Qobj(np.diag([-1, 1] + [0] * (n - 2))))
        super().__init__(omega_q=TWO_PI * self.f01, alpha=TWO_PI * self.anharmonicity,
                         n_levels=n, c_ops=c_ops)
        self._H0 = qt.Qobj(np.diag(TWO_PI * (e - e[0]))) # ground state is the reference energy, so H0|0> = 0
        n_op = tmon.matrixelement_table('n_operator', evals_count=n)
        # self._drive_op = qt.Qobj(n_op / abs(n_op[0, 1]))

    @property
    def H0(self):
        return self._H0

    # @property
    # def drive_op(self):
    #     return self._drive_op

    # -------------------------------------------------------------- coherence

    def _select_channels(self, noise_channels):
        supported = self.tmon.supported_noise_channels()
        if noise_channels is None:  # scqubits' default: every channel but t1_charge_impedance
            return [c for c in supported if c != 't1_charge_impedance']
        return [c for c in noise_channels if c in supported]

    def coherence_times(self):
        """
        (T1, T_phi, T2) in ns of the 0<->1 transition from the scqubits noise
        models selected at construction (np.inf if no channel contributes);
        1/T2 = 1/(2 T1) + 1/T_phi.
        """
        t1_channels = [c for c in self.noise_channels if c.startswith('t1')]
        tphi_channels = [c for c in self.noise_channels if c.startswith('tphi')]
        t1 = self.tmon.t1_effective(noise_channels=t1_channels) if t1_channels else np.inf
        rate_phi = sum(1.0 / getattr(self.tmon, c)() for c in tphi_channels)
        t_phi = 1.0 / rate_phi if rate_phi else np.inf
        return t1, t_phi, 1.0 / (0.5 / t1 + 1.0 / t_phi)

    def noise_channel_report(self):
        """Report lines: T1 of every scqubits T1 channel, its physical cause and whether it is included."""
        lines = ['T1 noise channels (1/T1 = sum of the included rates):']
        for c in (c for c in self.tmon.supported_noise_channels() if c.startswith('t1')):
            status = 'included' if c in self.noise_channels else 'excluded'
            lines.append(f'  {c}: T1 = {getattr(self.tmon, c)() / 1e3:.4g} us ({status})')
            lines.append(f'      {T1_CHANNEL_INFO.get(c, "no description")}')
        for c in (c for c in self.tmon.supported_noise_channels() if c.startswith('tphi')):
            status = 'included' if c in self.noise_channels else 'excluded'
            lines.append(f'  {c}: T_phi = {getattr(self.tmon, c)() / 1e3:.4g} us ({status})')
            lines.append(f'      {T_PHI_CHANNEL_INFO.get(c, "no description")}')
        return lines

    # ---------------------------------------------------------------- readout

    @property
    def delta(self):
        """Qubit-resonator detuning Delta_qr = f01 - f_res (GHz)."""
        if None in (self.f_res, self.g):
            raise ValueError('f_res and g must be given to TransmonDynamics for readout quantities')
        return self.f01 - self.f_res

    @property
    def chi(self):
        """Dispersive shift chi = g^2 / Delta_qr (GHz)."""
        return self.g ** 2 / self.delta

    @property
    def n_crit(self):
        """n_crit = Delta_qr^2 / (4 g^2): the dispersive approximation needs n << n_crit."""
        return self.delta ** 2 / (4 * self.g ** 2)

    def readout_hamiltonian(self, dispersive=False):
        """
        Qubit ⊗ resonator Hamiltonian (qutip, GHz, hbar = 1) with the qubit
        reduced to its two levels, in the convention of the readout notes:
        basis(2, 0) = |g>, basis(2, 1) = |e>, sigma_z|g> = +|g>.

        Exact Jaynes-Cummings (dispersive=False):
            H = f_r a†a - (f_q / 2) sigma_z + g (a† sigma_- + a sigma_+)
        Its exchange between |e, n> and |g, n+1> is what the dispersive
        (second order in g/Delta_qr) effective Hamiltonian removes:
            H = (f_r - chi sigma_z) a†a - (f_q + chi) / 2 sigma_z,   chi = g^2 / Delta_qr
        which commutes with sigma_z: the qubit state only selects the resonator
        frequency f_r -/+ chi (g/e). Valid for g sqrt(n+1) << |Delta_qr|.
        """
        self.delta  # check that f_res and g are given
        sz, sm = qt.sigmaz(), qt.destroy(2)  # sigma_minus = |g><e|
        a = qt.destroy(self.n_res)
        n_op, id_q, id_r = a.dag() * a, qt.qeye(2), qt.qeye(self.n_res)
        if dispersive:
            return (qt.tensor(id_q, self.f_res * n_op) - self.chi * qt.tensor(sz, n_op)
                    - 0.5 * (self.f01 + self.chi) * qt.tensor(sz, id_r))   # dispersive regime Hamiltonian, valid for g sqrt(n+1) << |Delta_qr|
        return (qt.tensor(id_q, self.f_res * n_op) - 0.5 * self.f01 * qt.tensor(sz, id_r)
                + self.g * (qt.tensor(sm, a.dag()) + qt.tensor(sm.dag(), a)))  # full Jaynes-Cummings Hamiltonian

    @property
    def chi_transmon(self):
        """
        Analytic dispersive shift of a multilevel transmon (Koch et al. 2007),
        chi = g^2 alpha / (Delta (Delta + alpha)) in GHz. The |1>-|2> transition
        also pushes the resonator, so this differs from g^2/Delta when |Delta| is
        not much larger than |alpha|; it tends to g^2/Delta for |alpha| -> infinity.
        """
        return self.g ** 2 * self.anharmonicity / (self.delta * (self.delta + self.anharmonicity))

    def transmon_readout_hamiltonian(self):
        """
        Transmon ⊗ resonator Hamiltonian (qutip, GHz, hbar = 1) keeping all
        `n_levels` transmon levels, to compare with the two-level models:
            H = sum_j (E_j - E_0) |j><j| + f_r a†a + g (n / |<0|n|1>|) (a + a†)
        n is the charge operator in the transmon eigenbasis. Dividing by
        |<0|n|1>| makes g the coupling of the 0<->1 transition, the same g as in
        the Jaynes-Cummings model; the higher transitions then couple with
        their true relative matrix elements (<1|n|2> ~ sqrt(2) <0|n|1>) and the
        counter-rotating terms are kept.
        """
        self.delta  # check that f_res and g are given
        n_q = self.n_levels
        n_mat = self.tmon.matrixelement_table('n_operator', evals_count=n_q)
        n_op = qt.Qobj(n_mat / abs(n_mat[0, 1]))
        a = qt.destroy(self.n_res)
        return (qt.tensor(self.H0 / TWO_PI, qt.qeye(self.n_res))
                + qt.tensor(qt.qeye(n_q), self.f_res * a.dag() * a)
                + self.g * qt.tensor(n_op, a + a.dag()))

    @staticmethod
    def dressed_frequencies(H):
        """
        Dressed frequencies (GHz) from the eigenstates of a qubit ⊗ resonator
        Hamiltonian `H`, labelled by their largest overlap with the bare
        |qubit level j, photon number k>: 'f01' (resonator in |0>), 'f_res_0' /
        'f_res_1' (resonator frequency with the qubit in level 0 / 1) and the
        dispersive shift 'chi' = (f_res_1 - f_res_0) / 2.
        """
        n_q, n_r = H.dims[0]
        evals, evecs = H.eigenstates()

        def dressed(j, k):
            bare = qt.tensor(qt.basis(n_q, j), qt.basis(n_r, k))
            return evals[int(np.argmax([abs(bare.overlap(v)) for v in evecs]))] # largest overlap with the bare state |j, k>

        e00, e10, e01, e11 = dressed(0, 0), dressed(1, 0), dressed(0, 1), dressed(1, 1)
        f_res_0, f_res_1 = e01 - e00, e11 - e10
        return {'f01': e10 - e00, 'f_res_0': f_res_0, 'f_res_1': f_res_1,
                'chi': 0.5 * (f_res_1 - f_res_0)}

    def readout_summary(self):
        """
        Dressed frequencies of the exact Jaynes-Cummings model ('jc'), of the
        dispersive Hamiltonian ('disp') and of the full multilevel transmon
        ('transmon', see transmon_readout_hamiltonian), next to the analytic
        prediction ('analytic': f_r -/+ chi for g/e, f_q + chi for the qubit),
        the multilevel analytic shift 'chi_transmon', and n_crit.
        """
        return {'jc': self.dressed_frequencies(self.readout_hamiltonian()),
                'transmon': self.dressed_frequencies(self.transmon_readout_hamiltonian()),
                'chi_transmon': self.chi_transmon,
                'disp': self.dressed_frequencies(self.readout_hamiltonian(dispersive=True)),
                'analytic': {'f01': self.f01 + self.chi, 'f_res_0': self.f_res - self.chi,
                             'f_res_1': self.f_res + self.chi, 'chi': self.chi},
                'n_crit': self.n_crit}

    def purcell_t1(self):
        """
        Purcell-limited T1 (ns): gamma = kappa (g/Delta)^2 (angular units, kappa
        in GHz as a rate/2pi), valid in the dispersive regime |Delta| >> g.
        """
        return 1.0 / (TWO_PI * self.kappa * (self.g / self.delta) ** 2)

    # --------------------------------------------------------------- dynamics

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
        P1 = a exp(-t/T1). `t_max` in ns. Returns (t1_fit [ns], t [ns], p1,
        fit curve on t).
        """
        pulse = XGate(np.pi) if pulse is None else pulse
        _, traj = self.apply_gate(pulse, psi0=qt.ket2dm(self.ground_state))
        t = np.linspace(0, t_max or 3 * self.t1, n_points)
        states = self.free_evolution(qt.Qobj(traj[-1]), t)
        p1 = np.array([np.real(s.full()[1, 1]) for s in states])
        model = lambda x, tau, a: a * np.exp(-x / tau)
        popt, _ = curve_fit(model, t, p1, p0=[self.t1, p1[0]])
        return popt[0], t, p1, model(t, *popt)

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
        (t2_fit [ns], t [ns], p1, fit curve on t, (lower, upper) decay
        envelope b -/+ |a| exp(-t/T2)).
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
        tau, a, b, _ = popt
        decay = abs(a) * np.exp(-t / tau)
        return tau, t, p1, fit(t, *popt), (b - decay, b + decay)


def save_output(name, text=None, dpi=150):
    """
    Save to DATA_DIR/name: `text` as .txt if given, otherwise the current
    matplotlib figure as .png (then closed). Returns the path.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name if '.' in name else name + ('.txt' if text is not None else '.png'))
    if text is None:
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
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
    dyn = TransmonDynamics(tmon, f_res=f_res, g=g, kappa=kappa)
    readout = dyn.readout_summary()
    disp = readout['jc']

    t1_fit, t_t1, p_t1, fit_t1 = dyn.measure_t1()
    t2_fit, t_t2, p_t2, fit_t2, env_t2 = dyn.measure_t2()

    x_gate = XGate(np.pi, sigma=5.0)
    t_pulse, traj = dyn.apply_gate(x_gate)
    pops = dyn.populations(traj)
    # x_gate.frequency_spectrum(omega_q=dyn.f01, save_as='x_pi_frequency_spectrum')
    # x_gate.plot_bloch_trajectory()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t_pulse, pops[:, :3])
    axes[0].set(xlabel='t (ns)', ylabel='population', title='pi pulse (Gaussian, sigma = 5 ns)')
    axes[0].legend(['|0>', '|1>', '|2>'])
    axes[1].plot(t_t1 / 1e3, p_t1, label='simulation')
    axes[1].plot(t_t1 / 1e3, fit_t1, 'k--', label=f'fit: T1 = {t1_fit / 1e3:.2f} us')
    axes[1].set(xlabel='t (us)', ylabel='P(|1>)', title='T1 decay')
    axes[1].legend()
    axes[2].plot(t_t2 / 1e3, p_t2, label='simulation')
    axes[2].plot(t_t2 / 1e3, fit_t2, 'k--', lw=0.8, label='fit')
    axes[2].plot(t_t2 / 1e3, env_t2[0], 'r-', label=f'envelope: T2 = {t2_fit / 1e3:.2f} us')
    axes[2].plot(t_t2 / 1e3, env_t2[1], 'r-')
    axes[2].set(xlabel='t (us)', ylabel='P(|1>)', title='Ramsey')
    axes[2].legend()
    plt.tight_layout()
    fig_path = save_output('scq_dynamics')

    report = '\n'.join([
        f'Transmon: EJ = {ej} GHz, EC = {ec} GHz (EJ/EC = {ej / ec:.0f})',
        f'  f01 = {dyn.f01:.4f} GHz, anharmonicity = {dyn.anharmonicity * 1e3:.1f} MHz',
        '',
        'Coherence (scqubits noise models):',
        f'  T1 = {dyn.t1 / 1e3:.2f} us, T_phi = {dyn.t_phi / 1e3:.2f} us, '
        f'T2 = {dyn.t2 / 1e3:.2f} us',
        f'  simulated fits: T1 = {t1_fit / 1e3:.2f} us, T2 = {t2_fit / 1e3:.2f} us',
        '',
        *dyn.noise_channel_report(),
        '',
        'pi pulse final populations |0>,|1>,|2>,...:',
        f'  {np.round(pops[-1], 5)}',
        '',
        f'Resonator: f_res = {f_res} GHz, g = {g * 1e3:.0f} MHz, kappa = {kappa * 1e3:.1f} MHz',
        f'  detuning = {dyn.delta * 1e3:.0f} MHz, dressed f01 = {disp["f01"]:.4f} GHz',
        f'  f_res(|0>) = {disp["f_res_0"]:.5f} GHz, f_res(|1>) = {disp["f_res_1"]:.5f} GHz',
        f'  dispersive shift chi: exact JC = {disp["chi"] * 1e3:.3f} MHz, '
        f'H_disp = {readout["disp"]["chi"] * 1e3:.3f} MHz, g^2/Delta = {readout["analytic"]["chi"] * 1e3:.3f} MHz',
        f'  multilevel transmon (all {tmon.truncated_dim} levels, coupling g n/|n01|): '
        f'chi = {readout["transmon"]["chi"] * 1e3:.3f} MHz, '
        f'analytic g^2 alpha/(Delta (Delta + alpha)) = {readout["chi_transmon"] * 1e3:.3f} MHz',
        f'  critical photon number n_crit = Delta^2/(4 g^2) = {readout["n_crit"]:.0f}',
        f'  Purcell-limited T1 = {dyn.purcell_t1() / 1e3:.1f} us',
        '      Purcell decay: the qubit is slightly hybridized with the resonator, so it',
        '      leaks energy out through the resonator\'s output line at kappa (g/delta)^2.',
        '      Not included in the T1 above (not a scqubits noise channel).',
        ''])
    print(report)
    print('report saved to', save_output('scq_report', text=report))
    print('figure saved to', fig_path)
