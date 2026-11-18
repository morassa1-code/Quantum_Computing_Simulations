import copy
import os

import numpy as np
import qutip as qt
import scqubits as scq
import matplotlib.pyplot as plt
from scipy.linalg import expm
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
    't1_flux_bias_line':
        'loss into the flux bias line: the qubit\'s SQUID loop is inductively coupled to the\n'
        '      on-chip line that sets the flux, so the qubit emits photons into the line\'s\n'
        '      impedance. Grows with the flux sensitivity of f01 (zero at the flux sweet spot\n'
        '      for a symmetric SQUID); set by the mutual inductance M and line impedance\n'
        '      (scqubits default Z = 50 Ohm, M = 400).',
}

T_PHI_CHANNEL_INFO = {
    'tphi_1_over_f_flux':
        '1/f dephasing time (or rate) due to flux noise: fluctuating magnetic flux in the\n'
        '      SQUID loop (surface spins) shifts EJ and so f01 in proportion to df01/dflux.\n'
        '      Dominant dephasing away from the flux sweet spot (flux = 0 or 0.5).\n',
    'tphi_1_over_f_cc':
        '1/f dephasing time (or rate) due to critical current noise.\n',
    'tphi_1_over_f_ng':
        '1/f dephasing time (or rate) due to charge noise\n'
}


def make_transmon(ej=20.0, ec=0.25, ng=0.0, ncut=30, n_levels=4, flux=0.0, d=0.0):
    """
    scqubits TunableTransmon (flux-tunable, asymmetric SQUID, charge basis):
        EJ(flux) = EJmax sqrt(cos^2(pi flux) + d^2 sin^2(pi flux))
    ej : float
        Maximum Josephson energy EJmax in GHz (the value at flux = 0).
    ec : float
        Charging energy in GHz.
    ng : float
        Offset charge (0 = the charge-insensitive sweet spot).
    ncut : int
        Truncation level for the charge basis.
    n_levels : int
        Number of energy levels to consider.
    flux : float
        External flux through the SQUID loop in units of the flux quantum
        (0 = the top sweet spot, EJ = EJmax; 0.5 = the bottom, EJ = d EJmax).
    d : float
        SQUID junction asymmetry, d = (EJ2 - EJ1) / (EJ1 + EJ2) (0 = symmetric).
    """
    return scq.TunableTransmon(EJmax=ej, EC=ec, d=d, flux=flux, ng=ng, ncut=ncut,
                               truncated_dim=n_levels)


def make_resonator(f_res=5.0, n_res=12):
    """
    scqubits Oscillator: a harmonic readout resonator, H = f_res a†a.
    f_res : float
        Resonator frequency in GHz.
    n_res : int
        Fock-space truncation (number of photon levels kept).
    """
    return scq.Oscillator(E_osc=f_res, truncated_dim=n_res)


class Transmon_Resonator_Dynamics(TransmonQubit):
    """
    A scqubits TunableTransmon coupled to a readout resonator (scqubits Oscillator),
    as a simulation_tools TransmonQubit, so the pulses there (DrivePulse,
    XGate, YGate, RabiDrive) and apply_gate, fit_rabi, ... work on it
    unchanged. Every operator lives in the joint transmon ⊗ resonator space
    (dims [[n_levels, n_res], [n_levels, n_res]]), in the lab frame and in
    angular units (rad/ns, times in ns):

        H0       = 2 pi * readout_hamiltonian()      multilevel Jaynes-Cummings
        drive_op = (n / |<0|n|1>|) ⊗ 1               the drive acts on the transmon

    n is the transmon charge operator in its eigenbasis, normalized so a
    pulse's area is still its rotation angle on 0<->1. The pulses are tuned
    to the dressed qubit frequency f01_dressed (bare f01 + Lamb shift).

    Lindblad operators (c_ops): the transmon T1 and pure dephasing (from the
    scqubits noise models, see `coherence_times`) ⊗ 1, and the resonator
    decay sqrt(2 pi kappa) 1 ⊗ a. The qubit is slightly hybridized with the
    resonator, so the resonator decay also relaxes the qubit: Purcell decay
    appears in the simulation by itself (compare with `purcell_t1`).

    `populations` returns the transmon level populations with the resonator
    traced out, `resonator_populations` the photon-number distribution.
    """

    # ------------------------------------------------------------------ setup

    def __init__(self, tmon, resonator, g, kappa=None, noise_channels=None):
        """
        tmon : scqubits.TunableTransmon (truncated_dim = number of transmon levels).
        resonator : scqubits.Oscillator (E_osc = f_res in GHz, truncated_dim =
            number of photon states).
        g, kappa : qubit-resonator coupling and resonator linewidth (GHz).
            kappa=None: no resonator decay.
        noise_channels : scqubits noise-channel names to include (default: all
            the transmon supports, see tmon.supported_noise_channels()).
        """
        n = tmon.truncated_dim
        e = tmon.eigenvals(n)
        self.tmon = tmon
        self.resonator = resonator
        self.f_res = self.resonator.E_osc
        self.g = g
        self.kappa = kappa
        self.n_res = resonator.truncated_dim
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
        self.transmon_H0 = self.tmon.hamiltonian()
        # self.resonator_H0 = self.resonator.hamiltonian() no such command - do something else
        # charge operator <i|n|j> in the transmon eigenbasis (n x n numpy array)
        self.n_mat = tmon.matrixelement_table('n_operator', evals_count=n)
        self.tr_detuning = self.f01 - self.f_res
        _, _, chi_j = self.ladder_couplings()
        self.f_res_0 = self.f_res - chi_j[0]
        self.f_res_1 = self.f_res + chi_j[0] - (chi_j[1] if len(chi_j) > 1 else 0.0)

        # ---- joint transmon ⊗ resonator system, angular units (rad/ns)
        id_q, id_r = qt.qeye(n), qt.qeye(self.n_res)
        a = qt.Qobj(resonator.annihilation_operator())
        self._H0 = TWO_PI * self.readout_hamiltonian()
        self._drive_op = qt.tensor(qt.Qobj(self.n_mat / abs(self.n_mat[0, 1])), id_r)
        self.c_ops = [qt.tensor(c, id_r) for c in c_ops]
        if kappa:
            self.c_ops.append(np.sqrt(TWO_PI * kappa) * qt.tensor(id_q, a))
        # total excitation number; the JC coupling conserves it (see free_evolution)
        self.n_total = qt.tensor(qt.num(n), id_r) + qt.tensor(id_q, a.dag() * a)
        # drive the qubit at its dressed frequency (bare f01 + Lamb shift)
        self.f01_dressed = self.dressed_frequencies(self.readout_hamiltonian())['f01']
        self.omega_q = TWO_PI * self.f01_dressed

    @property
    def H0(self):
        return self._H0

    @property
    def drive_op(self):
        return self._drive_op

    @property
    def ground_state(self):
        """|0>_transmon ⊗ |0>_resonator (an exact eigenstate of the RWA JC model)."""
        return qt.tensor(qt.basis(self.n_levels, 0), qt.basis(self.n_res, 0))

    def _diagonals(self, psi_traj):
        """Joint-basis populations at each step, shape (n_steps, n_levels, n_res)."""
        psi_traj = np.asarray(psi_traj)
        if psi_traj.ndim == 3:  # density matrices
            d = np.real(np.diagonal(psi_traj, axis1=1, axis2=2))
        else:  # kets
            d = np.abs(psi_traj.reshape(len(psi_traj), -1)) ** 2
        return d.reshape(len(d), self.n_levels, self.n_res)

    def populations(self, psi_traj):
        """Transmon level populations, resonator traced out (shape (n_steps, n_q))."""
        return self._diagonals(psi_traj).sum(axis=2)

    def resonator_populations(self, psi_traj):
        """Photon-number populations, transmon traced out (shape (n_steps, n_res))."""
        return self._diagonals(psi_traj).sum(axis=1)

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
    def chi(self):
        """
        Dispersive shift of the multilevel transmon (GHz):
            chi = (f_res_1 - f_res_0) / 2 = chi_0 - chi_1 / 2,   chi_j = g_j^2 / detuning_j
        (see ladder_couplings). The chi_1 term is the push of the |1>-|2>
        transition on the resonator; with g_1 = sqrt(2) g this is Koch et al.'s
        g^2 alpha / (Delta (Delta + alpha)). For n_levels = 2, or
        |Delta| >> |alpha|, it reduces to the two-level g^2 / Delta.
        """
        return 0.5 * (self.f_res_1 - self.f_res_0)

    @property
    def n_crit(self):
        """n_crit = detuning_qr^2 / (4 g^2): the dispersive approximation needs n << n_crit."""
        return self.tr_detuning ** 2 / (4 * self.g ** 2)

    def ladder_couplings(self):
        """
        Nearest-neighbour data of the transmon ladder, for j = 0 .. n_levels-2:
            g_j     = g <j|n|j+1> / |<0|n|1>|        coupling of j <-> j+1 (g_0 = g)
            detuning_j = (E_{j+1} - E_j) - f_r          detuning of j <-> j+1 from the resonator
            chi_j   = g_j^2 / detuning_j                partial dispersive shift
        All in GHz. For a transmon g_j ~ sqrt(j+1) g, like a harmonic oscillator.
        """
        n_mat = self.n_mat
        e = self.tmon.eigenvals(self.n_levels)
        j = np.arange(self.n_levels - 1)
        g_j = self.g * np.abs(n_mat[j, j + 1]) / abs(n_mat[0, 1])
        detuning_j = np.diff(e) - self.f_res
        chi_j = g_j ** 2 / detuning_j
        return g_j, detuning_j, chi_j

    def readout_hamiltonian(self, dispersive=False):
        """
        Multilevel transmon ⊗ resonator Hamiltonian (qutip, GHz, hbar = 1),
        keeping all `n_levels` transmon levels; basis(n_levels, j) = |j>.

        Generalized Jaynes-Cummings (dispersive=False, rotating-wave approx.):
            H = sum_j E_j |j><j| + f_r a†a
                + sum_j g_j (a† |j><j+1| + a |j+1><j|)
        with E_j = E_j - E_0 and g_j, detuning_j, chi_j from `ladder_couplings`.
        Each photon can only move the transmon one rung (j <-> j+1). For
        n_levels = 2 this is the usual two-level JC model.

        Dispersive (second order in g_j / detuning_j): |j, n> is pushed by its two
        neighbours |j+1, n-1> (shift -n chi_j) and |j-1, n+1> (shift
        +(n+1) chi_{j-1}), which gives the diagonal Hamiltonian
            H = sum_j [E_j + chi_{j-1}] |j><j| + sum_j (f_r + chi_{j-1} - chi_j) |j><j| a†a
        (chi_{-1} = 0). Level j only sets the resonator frequency
        f_r + chi_{j-1} - chi_j; the 0/1 readout shift is
        (f_res_1 - f_res_0) / 2 = chi_0 - chi_1 / 2, Koch et al.'s chi.
        Valid for g_j sqrt(n+1) << |detuning_j|. The top level has no j+1 in the
        truncated space, so its shift is a truncation artefact; keep it unused.
        """
        n_q = self.tmon.truncated_dim
        n_r = self.resonator.truncated_dim

        g_j, _, chi_j = self.ladder_couplings()
        e = self.tmon.eigenvals(n_q)
        e = e - e[0]
        # scqubits returns operators as numpy arrays; wrap as Qobj for qutip
        a = qt.Qobj(self.resonator.annihilation_operator())
        n_op = a.dag() * a
        id_r = qt.qeye(n_r)
        proj = lambda j, k: qt.basis(n_q, j) * qt.basis(n_q, k).dag()  # |j><k|

        if dispersive:
            chi_below = np.concatenate([[0.0], chi_j])        # chi_{j-1}
            chi_above = np.concatenate([chi_j, [0.0]])        # chi_j (0 for the top level)
            H = 0
            for j in range(n_q):
                H += qt.tensor(proj(j, j), (e[j] + chi_below[j]) * id_r
                               + (self.f_res + chi_below[j] - chi_above[j]) * n_op)
            return H

        H = qt.tensor(qt.Qobj(np.diag(e)), id_r) + qt.tensor(qt.qeye(n_q), self.f_res * n_op)
        for j, gj in enumerate(g_j):
            H += gj * (qt.tensor(proj(j, j + 1), a.dag()) + qt.tensor(proj(j + 1, j), a))
        return H

    # def transmon_readout_hamiltonian(self):
    #     """
    #     Transmon ⊗ resonator Hamiltonian (qutip, GHz, hbar = 1) keeping all
    #     `n_levels` transmon levels, to compare with the two-level models:
    #         H = sum_j (E_j - E_0) |j><j| + f_r a†a + g (n / |<0|n|1>|) (a + a†)
    #     n is the charge operator in the transmon eigenbasis. Dividing by
    #     |<0|n|1>| makes g the coupling of the 0<->1 transition, the same g as in
    #     the Jaynes-Cummings model; the higher transitions then couple with
    #     their true relative matrix elements (<1|n|2> ~ sqrt(2) <0|n|1>) and the
    #     counter-rotating terms are kept.
    #     """
    #     n_q = self.n_levels
    #     n_op = qt.Qobj(self.n_mat / abs(self.n_mat[0, 1]))
    #     a = qt.destroy(self.n_res)
    #     return (qt.tensor(self.H0 / TWO_PI, qt.qeye(self.n_res))
    #             + qt.tensor(qt.qeye(n_q), self.f_res * a.dag() * a)
    #             + self.g * qt.tensor(n_op, a + a.dag()))

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
        _, _, chi_j = self.ladder_couplings()
        return {'jc': self.dressed_frequencies(self.readout_hamiltonian()),
                'transmon': self.dressed_frequencies(self.readout_hamiltonian()),
                'chi_transmon': self.chi,
                'disp': self.dressed_frequencies(self.readout_hamiltonian(dispersive=True)),
                'analytic': {'f01': self.f01 + self.chi, 'f_res_0': self.f_res - self.chi,
                             'f_res_1': self.f_res + self.chi, 'chi': self.chi},
                'n_crit': self.n_crit}

    def purcell_t1(self):
        """
        Purcell-limited T1 (ns): gamma = kappa (g/detuning)^2 (angular units, kappa
        in GHz as a rate/2pi), valid in the dispersive regime |detuning| >> g.
        """
        return 1.0 / (TWO_PI * self.kappa * (self.g / self.tr_detuning) ** 2)

    @property
    def t1_expected(self):
        """T1 (ns) expected in the simulation: scqubits channels + Purcell, 1/T1 = 1/T1_scq + 1/T1_purcell."""
        purcell_rate = 1.0 / self.purcell_t1() if self.kappa else 0.0
        return 1.0 / (1.0 / self.t1 + purcell_rate)

    @property
    def t2_expected(self):
        """T2 (ns) expected in the simulation: 1/T2 = 1/(2 T1_expected) + 1/T_phi."""
        return 1.0 / (0.5 / self.t1_expected + 1.0 / self.t_phi)

    # --------------------------------------------------------------- dynamics

    def free_evolution(self, rho0, times, f_frame=None):
        """
        Undriven Lindblad evolution of transmon ⊗ resonator from rho0, returned
        at `times` (uniformly spaced; rho0 is the state at times[0]), in the
        frame rotating at f_frame (GHz, default f01_dressed):
            H = H0 - 2 pi f_frame N_tot,    N_tot = transmon level + photon number.
        The JC coupling conserves N_tot, so this frame is exact and H stays
        time-independent. The qubit-resonator detuning still makes GHz
        oscillations, which would force an ODE solver into ~1e7 steps over
        microseconds; instead the Liouvillian propagator exp(L dt) is
        computed once and applied step by step (exact, any length of time).
        """
        f_frame = self.f01_dressed if f_frame is None else f_frame
        times = np.asarray(times, dtype=float)
        dt = times[1] - times[0]
        if not np.allclose(np.diff(times), dt):
            raise ValueError('free_evolution needs uniformly spaced times')
        H = self.H0 - TWO_PI * f_frame * self.n_total
        # H = self.H0
        step = expm(qt.liouvillian(H, self.c_ops).full() * dt)
        dim = rho0.shape[0]
        vec = rho0.full().ravel(order='F')  # column stacking, as qutip's superoperators
        states = []
        for _ in times:
            states.append(qt.Qobj(vec.reshape(dim, dim, order='F'), dims=rho0.dims))
            vec = step @ vec
        return states

    def measure_t1(self, pulse=None, t_max=None, n_points=60, **kwargs):
        """
        Excite |0, 0 photons> with `pulse` (default XGate(pi)), decay freely
        and fit the transmon P1 = a exp(-t/T1). Includes the Purcell decay
        through the resonator, so compare with t1_expected. `t_max` in ns.
        Returns (t1_fit [ns], t [ns], p1, fit curve on t).
        """
        pulse = XGate(np.pi, sigma=2.0, **kwargs) if pulse is None else pulse
        _, traj = self.apply_gate(pulse, psi0=qt.ket2dm(self.ground_state))
        t = np.linspace(0, t_max or 3 * self.t1_expected, n_points)
        states = self.free_evolution(qt.Qobj(traj[-1], dims=self.H0.dims), t)
        p1 = self.populations([s.full() for s in states])[:, 1]
        model = lambda x, tau, a: a * np.exp(-x / tau)
        popt, _ = curve_fit(model, t, p1, p0=[self.t1_expected, p1[0]])
        return popt[0], t, p1, model(t, *popt)

    def measure_t2(self, pulse=None, t_max=None, detuning=None, n_points=500, **kwargs):
        """
        Ramsey: pulse - wait t - pulse, with `pulse` (default XGate(pi/2))
        driven `detuning` GHz above f01_dressed (default: 20 fringes over the
        window); `t_max` in ns. The pulse is simulated once, in the lab frame,
        as a unitary U on transmon ⊗ resonator (decoherence during the ~tens of
        ns pulse is neglected: a superoperator of the joint space would be
        (n_levels n_res)^2 wide and very slow to compute). The wait is free
        evolution with decoherence in the frame rotating at the drive
        frequency, where the second pulse acts like the first (up to
        counter-rotating terms), so
            P1(t) = Tr_res <1| U ( W(t)[ R U |0,0><0,0| U^dag R^dag ] ) U^dag |1>,
        with R moving the state after the first pulse into that frame.
        Fits P1 = b + a exp(-t/T2) cos(2 pi detuning t + phi) and returns
        (t2_fit [ns], t [ns], p1, fit curve on t, (lower, upper) decay
        envelope b -/+ |a| exp(-t/T2)).
        """
        t_max = t_max or 3 * self.t2_expected
        detuning = detuning or 20.0 / t_max
        pulse = XGate(np.pi / 2) if pulse is None else copy.copy(pulse)
        f_drive = self.f01_dressed + detuning
        pulse.drive_frequency = TWO_PI * f_drive

        # pulse on [0, duration], so the drive carrier and the frame agree at t = 0
        H, t_pulse = self.driven_hamiltonian(pulse, t0=pulse.n_sigma * pulse.sigma)
        # propagator evolves every joint basis state, including high ones like
        # |3, n_res-1> whose lab-frame phase spins at ~2 pi (E_3 + (n_res-1) f_res),
        # so it needs far more ODE steps than evolving a single state
        U = qt.propagator(qt.QobjEvo(H, tlist=t_pulse), t_pulse[-1],
                          options={'nsteps': 10 ** 7})

        def apply_s(rho):
            return U * rho * U.dag()

        R = (1j * TWO_PI * f_drive * t_pulse[-1] * self.n_total).expm()
        rho1 = R * apply_s(qt.ket2dm(self.ground_state)) * R.dag()
        t = np.linspace(0, t_max, n_points)
        states = self.free_evolution(rho1, t, f_frame=f_drive)
        p1 = self.populations([apply_s(s).full() for s in states])[:, 1]
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
