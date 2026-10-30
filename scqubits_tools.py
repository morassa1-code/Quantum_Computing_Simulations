import os

import numpy as np
import qutip as qt
import scqubits as scq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

scq.settings.PROGRESSBAR_DISABLED = True
scq.settings.T1_DEFAULT_WARNING = False
scq.set_units('GHz')  # energies/frequencies in GHz, coherence times in ns

TWO_PI = 2 * np.pi
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def make_transmon(ej=20.0, ec=0.25, ng=0.0, ncut=30, n_levels=4):
    """
    scqubits Transmon (cosine potential, charge basis). ej, ec in GHz; ng is
    the offset charge (0 = the charge-insensitive sweet spot).
    """
    return scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=ncut, truncated_dim=n_levels)


def make_resonator(f_res=7.0, n_res=12):
    """scqubits Oscillator (the readout resonator), f_res in GHz."""
    return scq.Oscillator(E_osc=f_res, truncated_dim=n_res)


def transmon_resonator(tmon, resonator, g):
    """
    Hamiltonian (qutip, GHz) of the scqubits transmon coupled to the scqubits
    resonator, H = H_transmon + H_resonator + g * n_transmon * (a + a_dag),
    written in the product of the two eigenbases (exact transmon energies and
    charge-operator matrix elements, so the coupling is not restricted to the
    qubit's two levels). scqubits' HilbertSpace is not used: its 4.3.1 release
    does not accept qutip 5 operators.
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
    evals, evecs = H.eigenstates()

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
    gamma = kappa * (g/delta)^2, in angular units.
    """
    return 1.0 / (TWO_PI * kappa * (g / delta) ** 2)


def coherence_times(tmon, noise_channels=None):
    """
    T1, T_phi and T2 (ns) of the 0<->1 transition from scqubits' noise models.
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


class TransmonDynamics:
    """
    Time evolution of a scqubits Transmon with qutip, in the eigenbasis and in
    the frame rotating at the drive frequency (RWA), with T1 and pure
    dephasing as Lindblad operators. Rates in rad/ns, times in ns.
    """

    def __init__(self, tmon, t1=np.inf, t_phi=np.inf, f_drive=None):
        n = tmon.truncated_dim
        e = tmon.eigenvals(n)
        self.n = n
        self.t1, self.t_phi = t1, t_phi
        self.f01 = e[1] - e[0]
        self.alpha = (e[2] - e[1]) - self.f01
        self.t2 = 1.0 / (0.5 / t1 + 1.0 / t_phi)
        self.f_drive = self.f01 if f_drive is None else f_drive
        # rotating frame: level j shifts by -j * f_drive
        self.H0 = qt.Qobj(np.diag(TWO_PI * (e - e[0] - np.arange(n) * self.f_drive)))
        n_op = tmon.matrixelement_table('n_operator', evals_count=n)
        self.n_up = qt.Qobj(np.triu(n_op, 1))  # raising part, kept by the RWA
        self.n01 = abs(n_op[0, 1])
        self.c_ops = []
        if np.isfinite(t1):
            self.c_ops += [np.sqrt(1.0 / t1) * qt.basis(n, j - 1) * qt.basis(n, j).dag()
                           * np.sqrt(j) for j in range(1, n)]
        if np.isfinite(t_phi):
            # dephasing of the 0-1 coherence at rate 1/t_phi
            self.c_ops.append(np.sqrt(0.5 / t_phi) * qt.Qobj(np.diag([-1, 1] + [0] * (n - 2))))

    def _drive(self, amplitude, phase=0.0):
        """Rotating-frame drive, (A/2)(e^{i phase} n_up + h.c.), A in rad/ns."""
        return 0.5 * amplitude * (np.exp(1j * phase) * self.n_up
                                  + np.exp(-1j * phase) * self.n_up.dag())

    def gaussian_pulse(self, angle=np.pi, sigma=5.0, phase=0.0, sample_rate=50.0):
        """
        Gaussian pulse (truncated at +-4 sigma) rotating the 0<->1 transition by
        `angle`. Returns the time grid and the envelope A(t) in rad/ns, with
        the area scaled so that  int A(t) dt * |n01| = angle.
        """
        t = np.arange(0, 8 * sigma, 1.0 / sample_rate)
        env = np.exp(-0.5 * ((t - 4 * sigma) / sigma) ** 2)
        env *= angle / (self.n01 * (env.sum() / sample_rate))
        return t, env

    def evolve(self, state, times, envelope=None, phase=0.0, e_ops=None):
        """
        mesolve from `state` over `times`; `envelope` is A(t) sampled on
        `times` (None = free evolution).
        """
        if envelope is None:
            H = self.H0
        else:
            H = [self.H0, [self._drive(1.0, phase), envelope]]
        return qt.mesolve(H, state, times, c_ops=self.c_ops, e_ops=e_ops or [],
                          options={'nsteps': 100000})

    def populations(self, states):
        """Level populations of each state (ket or density matrix), shape (len(states), n)."""
        return np.array([np.real(np.diag(qt.ket2dm(s).full() if s.isket else s.full()))
                         for s in states])

    def apply_pulse(self, angle=np.pi, sigma=5.0, sample_rate=50.0):
        """pi-type pulse on |0>: returns times, level populations vs. time."""
        t, env = self.gaussian_pulse(angle, sigma, sample_rate=sample_rate)
        res = self.evolve(qt.basis(self.n, 0), t, env)
        return t, self.populations(res.states)

    def measure_t1(self, t_max=None, n_points=60):
        """Prepare |1>, decay freely; fit P1 = exp(-t/T1). Returns (t1_fit, t, p1)."""
        t1_guess = self.t1
        t = np.linspace(0, t_max or 3 * t1_guess, n_points)
        res = self.evolve(qt.basis(self.n, 1), t)
        p1 = self.populations(res.states)[:, 1]
        popt, _ = curve_fit(lambda x, tau: np.exp(-x / tau), t, p1, p0=[t1_guess])
        return popt[0], t, p1

    def measure_t2(self, t_max=None, detuning=None, n_points=200):
        """
        Ramsey: pi/2 - wait - pi/2 with the drive detuned by `detuning` (GHz)
        from the qubit (default: 20 fringes over the window, GHz). Fits P1 = 0.5 + 0.5 exp(-t/T2) cos(2 pi detuning t)
        and returns (t2_fit, t, p1).
        """
        t_max = t_max or 3 * self.t2
        detuning = detuning or 20.0 / t_max
        probe = TransmonDynamics.__new__(TransmonDynamics)
        probe.__dict__.update(self.__dict__)
        probe.H0 = self.H0 + qt.Qobj(np.diag(TWO_PI * detuning * np.arange(self.n)))
        t = np.linspace(0, t_max, n_points)
        # ideal pi/2 rotations about x on the 0-1 subspace (no dissipation during them)
        half = (-1j * (np.pi / 2) / (2 * self.n01) * (self.n_up + self.n_up.dag())).expm()
        res = probe.evolve(half * qt.basis(self.n, 0), t)
        p1 = self.populations([half * s * half.dag() for s in
                               (qt.ket2dm(x) if x.isket else x for x in res.states)])[:, 1]
        fit = lambda x, tau, a, b: b + a * np.exp(-x / tau) * np.cos(TWO_PI * detuning * x)
        popt, _ = curve_fit(fit, t, p1, p0=[t_max / 3, -0.5, 0.5])
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
    ej, ec, f_res, g, kappa = 20.0, 0.25, 7.0, 0.1, 0.001  # GHz

    tmon = make_transmon(ej, ec)
    resonator = make_resonator(f_res)
    disp = dispersive_shift(transmon_resonator(tmon, resonator, g),
                            tmon.truncated_dim, resonator.truncated_dim)
    times = coherence_times(tmon)
    dyn = TransmonDynamics(tmon, times['t1'], times['t_phi'])
    delta = f_res - dyn.f01

    t1_fit, t_t1, p_t1 = dyn.measure_t1()
    t2_fit, t_t2, p_t2 = dyn.measure_t2()
    t_pulse, pops = dyn.apply_pulse(np.pi, sigma=5.0)

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
        f'  f01 = {dyn.f01:.4f} GHz, anharmonicity = {dyn.alpha * 1e3:.1f} MHz',
        '',
        'Coherence (scqubits noise models):',
        f'  T1 = {times["t1"] / 1e3:.2f} us, T_phi = {times["t_phi"] / 1e3:.2f} us, '
        f'T2 = {times["t2"] / 1e3:.2f} us',
        f'  simulated fits: T1 = {t1_fit / 1e3:.2f} us, T2 = {t2_fit / 1e3:.2f} us',
        '',
        'pi pulse final populations |0>,|1>,|2>,...:',
        f'  {np.round(pops[-1], 5)}',
        '',
        f'Resonator: f_res = {f_res} GHz, g = {g * 1e3:.0f} MHz, kappa = {kappa * 1e3:.1f} MHz',
        f'  detuning = {delta * 1e3:.0f} MHz, dressed f01 = {disp["f01"]:.4f} GHz',
        f'  f_res(|0>) = {disp["f_res_0"]:.5f} GHz, f_res(|1>) = {disp["f_res_1"]:.5f} GHz',
        f'  dispersive shift chi = {disp["chi"] * 1e3:.2f} MHz',
        f'  Purcell-limited T1 = {purcell_t1(g, delta, kappa) / 1e3:.1f} us',
        ''])
    print(report)
    print('report saved to', save_text(report, 'scq_report'))
    print('figure saved to', fig_path)
