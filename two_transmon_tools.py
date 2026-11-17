"""
Two capacitively coupled transmons, built from scqubits Transmon objects.

Each transmon is kept in its own eigenbasis (truncated_dim levels), and they
interact through their charges (a coupling capacitor between the islands):

    H / h = sum_j E1_j |j><j| ⊗ 1 + 1 ⊗ sum_j E2_j |j><j|
            + g (n1 / |<0|n1|1>|) ⊗ (n2 / |<0|n2|1>|)
            + delta2(t) N2                                  (flux tuning of qubit 2)

Frequencies in GHz; the dynamics run in angular units (rad/ns, times in ns),
in the lab frame. Dividing each n by its 0-1 matrix element makes g the
coupling of the |10> <-> |01> exchange; the higher levels couple with their
true relative matrix elements (<1|n|2> ~ sqrt(2) <0|n|1>), and the
counter-rotating terms (|00> <-> |11>) are kept.

Qubit 2 is flux tunable: delta2 shifts level j of qubit 2 by j * delta2
(N2 = 1 ⊗ sum_j j |j><j|). That is the first-order effect of changing E_J
with flux: the frequency moves while the anharmonicity (~ -E_C) stays put,
a good approximation for excursions of a few hundred MHz.

Decoherence: each transmon's T1 and pure dephasing from the scqubits noise
models, as Lindblad operators (same model as scqubits_tools).
"""
import numpy as np
import qutip as qt
import scqubits as scq
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar

from scqubits_tools import TWO_PI

COMP_LABELS = ((0, 0), (0, 1), (1, 0), (1, 1))  # computational states |q1 q2>
CZ = np.diag([1, 1, 1, -1]).astype(complex)
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
TARGET_GATES = {'cz': CZ, 'iswap': ISWAP}


def transmon_coherence(tmon, noise_channels=None):
    """
    (T1, T_phi) in ns of a scqubits transmon's 0-1 transition (np.inf if no
    channel contributes). Default channels: all supported but
    t1_charge_impedance (scqubits' default).
    """
    supported = tmon.supported_noise_channels()
    if noise_channels is None:
        channels = [c for c in supported if c != 't1_charge_impedance']
    else:
        channels = [c for c in noise_channels if c in supported]
    t1_channels = [c for c in channels if c.startswith('t1')]
    tphi_channels = [c for c in channels if c.startswith('tphi')]
    t1 = tmon.t1_effective(noise_channels=t1_channels) if t1_channels else np.inf
    rate_phi = sum(1.0 / getattr(tmon, c)() for c in tphi_channels)
    return t1, (1.0 / rate_phi if rate_phi else np.inf)


def local_z(a, b):
    """Z rotations exp(i a) on qubit 1 and exp(i b) on qubit 2, in the basis |00>,|01>,|10>,|11>."""
    return np.diag([1, np.exp(1j * b), np.exp(1j * a), np.exp(1j * (a + b))])


class TwoTransmonDynamics:
    """Two coupled scqubits transmons; see the module docstring for the model."""

    # ------------------------------------------------------------------ setup

    def __init__(self, tmon1, tmon2, g, noise=True, noise_channels=None):
        """
        tmon1, tmon2 : scqubits.Transmon (truncated_dim >= 3 each). Qubit 2 is
            the flux-tunable one.
        g : 0-1 exchange coupling (GHz).
        noise : include T1 / T_phi of both transmons as Lindblad operators.
        noise_channels : scqubits noise channels to use (default: scqubits').
        """
        self.tmons = (tmon1, tmon2)
        self.g = g
        self.dims = [t.truncated_dim for t in self.tmons]
        if min(self.dims) < 3:
            raise ValueError('each transmon needs truncated_dim >= 3 (|2> matters for ZZ and CZ)')

        self.energies, self.n_ops = [], []
        for t, d in zip(self.tmons, self.dims):
            e = t.eigenvals(d)
            self.energies.append(e - e[0])
            n_mat = t.matrixelement_table('n_operator', evals_count=d)
            self.n_ops.append(qt.Qobj(n_mat / abs(n_mat[0, 1])))
        self.f01 = [e[1] for e in self.energies]  # bare, GHz
        self.anharmonicity = [e[2] - 2 * e[1] for e in self.energies]

        self.level_op = [self.embed(qt.Qobj(np.diag(np.arange(d, dtype=float))), k)
                         for k, d in enumerate(self.dims)]  # N_k = sum_j j |j><j|
        self.drive_ops = [self.embed(self.n_ops[k], k) for k in (0, 1)]
        self.H_static = self.hamiltonian()  # GHz
        self.H0 = TWO_PI * self.H_static    # rad/ns

        self.t1, self.t_phi = [np.inf, np.inf], [np.inf, np.inf]
        self.c_ops = []
        if noise:
            for k, (t, d) in enumerate(zip(self.tmons, self.dims)):
                self.t1[k], self.t_phi[k] = transmon_coherence(t, noise_channels)
                if np.isfinite(self.t1[k]):
                    self.c_ops += [np.sqrt(j / self.t1[k])
                                   * self.embed(qt.basis(d, j - 1) * qt.basis(d, j).dag(), k)
                                   for j in range(1, d)]
                if np.isfinite(self.t_phi[k]):
                    self.c_ops.append(np.sqrt(0.5 / self.t_phi[k])
                                      * self.embed(qt.Qobj(np.diag([-1.0, 1.0] + [0.0] * (d - 2))), k))

    def embed(self, op, k):
        """Operator of transmon k (0 or 1) on the joint space."""
        ids = [qt.qeye(d) for d in self.dims]
        ids[k] = op
        return qt.tensor(*ids)

    def bare(self, j, k):
        """Bare product state |j>_1 ⊗ |k>_2."""
        return qt.tensor(qt.basis(self.dims[0], j), qt.basis(self.dims[1], k))

    def hamiltonian(self, delta2=0.0):
        """Joint Hamiltonian in GHz, with qubit 2 shifted by delta2 (see module docstring)."""
        return (self.embed(qt.Qobj(np.diag(self.energies[0])), 0)
                + self.embed(qt.Qobj(np.diag(self.energies[1])), 1)
                + self.g * qt.tensor(self.n_ops[0], self.n_ops[1])
                + delta2 * self.level_op[1])

    # --------------------------------------------------------------- spectrum

    def dressed_states(self, delta2=0.0, labels=COMP_LABELS):
        """
        {(j, k): (energy [GHz], eigenket)}: the eigenstate of hamiltonian(delta2)
        with the largest overlap with bare |j, k>, its phase fixed so that
        overlap is real and positive. Near a resonance two labels can pick the
        same eigenstate (the states are fully hybridized there).
        """
        evals, evecs = self.hamiltonian(delta2).eigenstates()
        out = {}
        for j, k in labels:
            bare = self.bare(j, k)
            overlaps = [bare.overlap(v) for v in evecs]
            i = int(np.argmax(np.abs(overlaps)))
            out[(j, k)] = (evals[i], evecs[i] * np.exp(-1j * np.angle(overlaps[i])))
        return out

    def zz_analytic(self, delta2=0.0):
        """
        Second-order ZZ (GHz): |11> is pushed by |20> and |02> (couplings
        sqrt(2) g), while the |10>, |01> shifts cancel:
            ZZ = 2 g^2 (a1 + a2) / ((D + a1)(D - a2)),   D = f1 - f2.
        Diverges at D = -a1 (|11> ~ |20>) and D = a2 (|11> ~ |02>).
        """
        D = self.f01[0] - (self.f01[1] + delta2)
        a1, a2 = self.anharmonicity
        return 2 * self.g ** 2 * (a1 + a2) / ((D + a1) * (D - a2))

    def spectrum_summary(self, delta2=0.0):
        """
        Dressed transition frequencies (GHz): 'f1' / 'f2' (other qubit in |0>),
        'f1_q2_excited' / 'f2_q1_excited' (other qubit in |1>), and
        ZZ = E11 - E10 - E01 + E00 = f1_q2_excited - f1, the frequency shift
        of one qubit conditioned on the other, next to 'zz_analytic'.
        """
        E = {l: v[0] for l, v in self.dressed_states(delta2).items()}
        return {'f1': E[(1, 0)] - E[(0, 0)], 'f2': E[(0, 1)] - E[(0, 0)],
                'f1_q2_excited': E[(1, 1)] - E[(0, 1)], 'f2_q1_excited': E[(1, 1)] - E[(1, 0)],
                'zz': E[(1, 1)] - E[(1, 0)] - E[(0, 1)] + E[(0, 0)],
                'zz_analytic': self.zz_analytic(delta2)}

    def sweep_delta2(self, deltas):
        """
        Spectrum vs qubit-2 frequency: returns bare f2 (GHz), dressed E10 and
        E01 above E00 (they anticross at f2 ~ f1, gap 2g), ZZ and ZZ analytic.
        """
        rows = []
        for d in deltas:
            dressed = self.dressed_states(d)
            E = {l: v[0] for l, v in dressed.items()}
            rows.append([self.f01[1] + d, E[(1, 0)] - E[(0, 0)], E[(0, 1)] - E[(0, 0)],
                         E[(1, 1)] - E[(1, 0)] - E[(0, 1)] + E[(0, 0)], self.zz_analytic(d)])
        f2, e10, e01, zz, zz_an = np.array(rows).T
        return {'f2': f2, 'E10': e10, 'E01': e01, 'zz': zz, 'zz_analytic': zz_an}

    def scqubits_check(self, evals_count=6):
        """
        The same system built with scqubits' own HilbertSpace (interaction
        g_raw n1 n2 with g_raw = g / (|n1_01| |n2_01|)): its lowest
        eigenvalues above the ground state (GHz), to compare with ours.
        """
        n01 = [abs(t.matrixelement_table('n_operator', evals_count=2)[0, 1]) for t in self.tmons]
        hs = scq.HilbertSpace(list(self.tmons))
        hs.add_interaction(g_strength=self.g / (n01[0] * n01[1]),
                           op1=self.tmons[0].n_operator, op2=self.tmons[1].n_operator)
        ev_scq = hs.eigenvals(evals_count=evals_count)
        ev_ours = self.H_static.eigenenergies()[:evals_count]
        return ev_scq - ev_scq[0], ev_ours - ev_ours[0]

    # ------------------------------------------------------------ populations

    def joint_populations(self, states):
        """Populations of the bare |j, k>, shape (n_steps, d1, d2), from kets or density matrices."""
        arr = np.asarray([s.full() if isinstance(s, qt.Qobj) else s for s in states])
        if arr.ndim == 3 and arr.shape[2] > 1:  # density matrices
            d = np.real(np.diagonal(arr, axis1=1, axis2=2))
        else:  # kets
            d = np.abs(arr.reshape(len(arr), -1)) ** 2
        return d.reshape(len(d), *self.dims)

    def qubit_populations(self, states, k):
        """Level populations of transmon k with the other traced out, shape (n_steps, d_k)."""
        return self.joint_populations(states).sum(axis=2 - k)

    # --------------------------------------------------------------- dynamics

    def free_evolution(self, rho0, times, delta2=0.0):
        """
        Undriven Lindblad evolution (lab frame) at fixed delta2, returned at the
        uniformly spaced `times` (rho0 at times[0]). The step propagator
        exp(L dt) is computed once and applied repeatedly, so it is exact for
        any length of time, GHz oscillations included.
        """
        rho0 = rho0 if rho0.isoper else qt.ket2dm(rho0)
        times = np.asarray(times, dtype=float)
        dt = times[1] - times[0]
        if not np.allclose(np.diff(times), dt):
            raise ValueError('free_evolution needs uniformly spaced times')
        step = expm(qt.liouvillian(TWO_PI * self.hamiltonian(delta2), self.c_ops).full() * dt)
        dim = rho0.shape[0]
        vec = rho0.full().ravel(order='F')  # column stacking, as qutip's superoperators
        states = []
        for _ in times:
            states.append(qt.Qobj(vec.reshape(dim, dim, order='F'), dims=rho0.dims))
            vec = step @ vec
        return states

    def swap_oscillations(self, t_max=None, n_points=400, delta2=None):
        """
        Start in bare |1,0>, with qubit 2 tuned to qubit 1 (default delta2 =
        f1 - f2), and let the excitation swap: P10 = cos^2(2 pi g t), so a
        full swap takes 1/(4g) and the period is 1/(2g) (default window: 3
        periods). Returns (t [ns], joint populations (n_steps, d1, d2)).
        """
        delta2 = self.f01[0] - self.f01[1] if delta2 is None else delta2
        t = np.linspace(0, t_max or 3 / (2 * self.g), n_points)
        states = self.free_evolution(self.bare(1, 0), t, delta2)
        return t, self.joint_populations(states)

    def apply_gate(self, pulse, qubit=0, psi0=None):
        """
        Microwave pulse (simulation_tools DrivePulse, e.g. XGate) on transmon
        `qubit` through its charge, H = H0 + Omega(t) cos(w_d t + phase) n_k,
        with decoherence. Drive frequency: the pulse's own, else the dressed
        frequency of `qubit` with the other qubit in |0>. psi0: default |00>.
        The pulse starts at t = 0. Returns (t [ns], states).
        """
        s = self.spectrum_summary()
        f_q = s['f1'] if qubit == 0 else s['f2']
        t0 = pulse.n_sigma * pulse.sigma
        pulse.generate_drive_signal(TWO_PI * f_q, t0, TWO_PI * self.anharmonicity[qubit])
        t = pulse.time_array(t0)
        H = [self.H0, [self.drive_ops[qubit], pulse.total_modulated_signal]]
        psi0 = self.bare(0, 0) if psi0 is None else psi0
        result = qt.mesolve(H, psi0, t, c_ops=self.c_ops,
                            options={'nsteps': 10 ** 6, 'store_states': True})
        return t, result.states

    # ------------------------------------------------------- two-qubit gates

    @staticmethod
    def flux_pulse(delta, t_flat, t_ramp=2.0, dt=0.01):
        """
        delta2(t): cosine ramp 0 -> delta over t_ramp, flat for t_flat, ramp back.
        Returns (t [ns], delta2(t) [GHz]).
        """
        t_total = t_flat + 2 * t_ramp
        t = np.arange(0.0, t_total + dt / 2, dt)
        shape = np.ones_like(t)
        up, down = t < t_ramp, t > t_ramp + t_flat
        shape[up] = 0.5 * (1 - np.cos(np.pi * t[up] / t_ramp))
        shape[down] = 0.5 * (1 - np.cos(np.pi * (t_total - t[down]) / t_ramp))
        return t, delta * shape

    def calibrate(self, gate='cz', window=0.05):
        """
        Operating point of a flux-pulse gate: the delta2 where the two
        relevant states anticross, found by minimizing their dressed splitting
        (the bare-energy estimate is off by the level shifts from the
        coupling), and the time on resonance:
          'cz'    : |11> <-> |02>, coupling ~ sqrt(2) g. A full cycle,
                    t = 1/gap, returns |11> to itself with a -1 sign.
          'iswap' : |10> <-> |01>, coupling g. Half a cycle, t = 1/(2 gap),
                    swaps them with a factor i.
        Returns {'delta2' [GHz], 'gap' [GHz], 't_on' [ns]}.
        """
        e1, e2 = self.energies
        if gate == 'cz':
            pair, guess = ((1, 1), (0, 2)), e1[1] + e2[1] - e2[2]
        elif gate == 'iswap':
            pair, guess = ((1, 0), (0, 1)), e1[1] - e2[1]
        else:
            raise ValueError("gate must be 'cz' or 'iswap'")
        bare = [self.bare(*l) for l in pair]

        def gap(d):
            evals, evecs = self.hamiltonian(d).eigenstates()
            weight = [sum(abs(b.overlap(v)) ** 2 for b in bare) for v in evecs]
            i, j = np.argsort(weight)[-2:]  # the two states living in span{pair}
            return abs(evals[i] - evals[j])

        res = minimize_scalar(gap, bounds=(guess - window, guess + window), method='bounded',
                              options={'xatol': 1e-7})
        g_min = res.fun
        return {'delta2': res.x, 'gap': g_min,
                't_on': 1 / g_min if gate == 'cz' else 1 / (2 * g_min)}

    def gate_unitary(self, t, delta_t):
        """
        Closed-system propagator of the flux pulse delta2(t) (lab frame),
        restricted to the dressed computational states of the idle qubits:
        M[i, j] = <d_i| U |d_j>, basis |00>, |01>, |10>, |11>.
        """
        H = [self.H0, [TWO_PI * self.level_op[1], delta_t]]
        U = qt.propagator(qt.QobjEvo(H, tlist=t), t[-1], options={'nsteps': 10 ** 7})
        d = self.dressed_states()
        basis = [d[l][1] for l in COMP_LABELS]
        return np.array([[bi.overlap(U * bj) for bj in basis] for bi in basis]), U

    @staticmethod
    def gate_metrics(M, gate):
        """
        Quality of the computational-subspace map M against `gate` ('cz' or
        'iswap'), allowing free single-qubit Z rotations before and after
        (virtual Z gates, free on hardware; they also absorb the lab-frame
        dynamic phases). Average gate fidelity
            F = (|Tr(V^dag M)|^2 + Tr(M^dag M)) / 20,
        leakage = 1 - Tr(M^dag M) / 4 (population lost from the subspace),
        and the conditional phase phi_00 - phi_01 - phi_10 + phi_11 (pi for CZ).
        """
        G = TARGET_GATES[gate]

        def fidelity(p):
            V = local_z(p[2], p[3]) @ G @ local_z(p[0], p[1])
            return (abs(np.trace(V.conj().T @ M)) ** 2 + np.trace(M.conj().T @ M).real) / 20

        rng = np.random.default_rng(0)
        best = max((minimize(lambda p: -fidelity(p), rng.uniform(-np.pi, np.pi, 4),
                             method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-12})
                    for _ in range(6)), key=lambda r: -r.fun)
        ph = np.angle(np.diag(M))
        cond = np.angle(np.exp(1j * (ph[0] - ph[1] - ph[2] + ph[3])))
        return {'fidelity': -best.fun, 'leakage': 1 - np.trace(M.conj().T @ M).real / 4,
                'conditional_phase': cond,
                'ideal': local_z(*best.x[2:]) @ G @ local_z(*best.x[:2])}

    def tune_gate(self, gate='cz', t_ramp=2.0, n_scan=21, span=0.3):
        """
        Calibrate (`calibrate`), then scan the flat-top length around
        t_on - t_ramp (the cosine ramps add about half their length on
        resonance) by +/- span and keep the best fidelity. Returns a dict with
        the calibration, t, delta2(t), M, U and gate_metrics of the best pulse.
        """
        cal = self.calibrate(gate)
        t_guess = max(cal['t_on'] - t_ramp, 0.1)
        best = None
        for t_flat in np.linspace(t_guess * (1 - span), t_guess * (1 + span), n_scan):
            t, delta_t = self.flux_pulse(cal['delta2'], t_flat, t_ramp)
            M, U = self.gate_unitary(t, delta_t)
            m = self.gate_metrics(M, gate)
            if best is None or m['fidelity'] > best['fidelity']:
                best = {**cal, **m, 't_flat': t_flat, 't_ramp': t_ramp,
                        't': t, 'delta_t': delta_t, 'M': M, 'U': U}
        return best

    def run_flux_pulse(self, t, delta_t, psi0):
        """Flux pulse delta2(t) with decoherence (mesolve), from psi0. Returns the states."""
        H = [self.H0, [TWO_PI * self.level_op[1], delta_t]]
        return qt.mesolve(H, psi0, t, c_ops=self.c_ops,
                          options={'nsteps': 10 ** 6, 'store_states': True}).states
