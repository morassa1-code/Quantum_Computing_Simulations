import numpy as np
import qutip as qt
import scqubits as scq

from simulation_tools import TransmonQubit
from decoherence import Decoherence

scq.settings.PROGRESSBAR_DISABLED = True
scq.settings.T1_DEFAULT_WARNING = False
scq.set_units('GHz')  # energies/frequencies in GHz, coherence times in ns

TWO_PI = 2 * np.pi


def make_transmon(ej=20.0, ec=0.25, ng=0.0, ncut=30, n_levels=4):
    """
    scqubits Transmon from first principles (cosine potential, charge basis),
    not the Duffing approximation. ej, ec in GHz; ng is the offset charge
    (0 = the charge-insensitive sweet spot for the 1/f charge noise).
    """
    return scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=ncut, truncated_dim=n_levels)



def to_transmon_qubit(tmon):
    """
    TransmonQubit for the existing pulse simulation, with omega_q and alpha
    taken from the exact scqubits spectrum. Angular units (rad/ns), so pulse
    times are in ns.
    """
    e = tmon.eigenvals(max(tmon.truncated_dim, 3))
    f01, f12 = e[1] - e[0], e[2] - e[1]
    return TransmonQubit(omega_q=TWO_PI * f01, alpha=TWO_PI * (f12 - f01),
                         n_levels=tmon.truncated_dim)


def coherence_times(tmon, noise_channels=None):
    """
    T1, T_phi and T2 (ns) of the 0<->1 transition from scqubits' noise
    models: dielectric loss for T1, and 1/f charge and critical-current noise
    for dephasing. `noise_channels` picks channels by scqubits name (see
    tmon.supported_noise_channels()); by default every supported channel but
    't1_charge_impedance', which couples the bare charge operator to a 50 Ohm
    line and gives an unphysical T1 of ~0.2 ns for a transmon. Returns a dict
    with 't1', 't_phi', 't2' and a `Decoherence` (ready for decoherence.py's
    apply_gate_open / measure_t1 / measure_t2).
    """
    supported = tmon.supported_noise_channels()
    if noise_channels is None:
        noise_channels = [c for c in supported if c != 't1_charge_impedance']
    wanted = [c for c in noise_channels if c in supported]
    t1_channels = [c for c in wanted if c.startswith('t1')]
    tphi_channels = [c for c in wanted if c.startswith('tphi')]

    t1 = tmon.t1_effective(noise_channels=t1_channels) if t1_channels else None
    rate_phi = sum(1.0 / getattr(tmon, c)() for c in tphi_channels)
    t_phi = 1.0 / rate_phi if rate_phi else None

    deco = Decoherence(t1=t1, t_phi=t_phi)
    return {'t1': t1, 't_phi': t_phi, 't2': deco.t2, 'decoherence': deco}


def transmon_resonator(tmon, f_res, g, n_res=12):
    """
    Transmon coupled to a resonator, H = sum_j E_j |j><j| + f_res a_dag a
    + g * n_transmon * (a + a_dag), with the transmon's exact eigenenergies
    and charge-operator matrix elements from scqubits (so the coupling is
    not restricted to the qubit's two levels). f_res, g in GHz.

    Returns a dict: dressed 'f01' (resonator in |0>), 'f_res_0' / 'f_res_1'
    (resonator frequency with the qubit in |0> / |1>) and the dispersive shift
    'chi' = (f_res_1 - f_res_0) / 2, all in GHz.
    """
    n_q = tmon.truncated_dim
    energies = tmon.eigenvals(n_q)
    n_op = qt.Qobj(tmon.matrixelement_table('n_operator', evals_count=n_q))
    a = qt.tensor(qt.qeye(n_q), qt.destroy(n_res))
    H = (qt.tensor(qt.Qobj(np.diag(energies)), qt.qeye(n_res))
         + f_res * a.dag() * a + g * qt.tensor(n_op, qt.qeye(n_res)) * (a + a.dag()))
    evals, evecs = H.eigenstates()

    def dressed(j, k):
        """Energy of the eigenstate with the largest overlap with the bare |j, k>."""
        bare = qt.tensor(qt.basis(n_q, j), qt.basis(n_res, k))
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


if __name__ == '__main__':
    from simulation_tools import XGate, save_text
    from decoherence import apply_gate_open, measure_t1, measure_t2

    ej, ec, f_res, g, kappa = 20.0, 0.25, 7.0, 0.1, 0.001  # GHz
    tmon = make_transmon(ej, ec)
    qubit = to_transmon_qubit(tmon)
    times = coherence_times(tmon)
    deco = times['decoherence']
    f01 = qubit.omega_q / TWO_PI
    delta = f_res - f01

    # T1 / T2 experiments on the exact transmon (plots saved in the data folder)
    t1_fit, _, _ = measure_t1(qubit, deco, save_as='scq_t1')
    t2_fit, _, _ = measure_t2(qubit, deco, save_as='scq_t2')

    # pi pulse with and without T1/T2 (RWA drive, time in ns)
    pulse = XGate(angle=np.pi, sigma=5.0, sample_rate=50.0)
    t_vals, psi = qubit.apply_gate(pulse, rwa=True)
    pop_closed = qubit.populations(psi)[-1]
    _, rho = apply_gate_open(qubit, pulse, deco, rwa=True)
    pop_open = np.real(np.diagonal(rho[-1]))
    qubit.plot_populations(t_vals, psi, save_as='scq_pi_pulse')

    disp = transmon_resonator(tmon, f_res, g)
    report = '\n'.join([
        f'Transmon: EJ = {ej} GHz, EC = {ec} GHz (EJ/EC = {ej / ec:.0f})',
        f'  f01 = {f01:.4f} GHz, anharmonicity = {qubit.alpha / TWO_PI * 1e3:.1f} MHz',
        '',
        'Coherence (scqubits noise models):',
        f'  T1 = {times["t1"] / 1e3:.2f} us, T_phi = {times["t_phi"] / 1e3:.2f} us, '
        f'T2 = {times["t2"] / 1e3:.2f} us',
        f'  simulated fits: T1 = {t1_fit / 1e3:.2f} us, T2 = {t2_fit / 1e3:.2f} us',
        '',
        f'pi pulse (Gaussian, sigma = 5 ns), final populations |0>,|1>,|2>,...:',
        f'  closed system: {np.round(pop_closed, 5)}',
        f'  with T1/T2:    {np.round(pop_open, 5)}',
        '',
        f'Resonator: f_res = {f_res} GHz, g = {g * 1e3:.0f} MHz, kappa = {kappa * 1e3:.1f} MHz',
        f'  detuning = {delta * 1e3:.0f} MHz, dressed f01 = {disp["f01"]:.4f} GHz',
        f'  f_res(|0>) = {disp["f_res_0"]:.5f} GHz, f_res(|1>) = {disp["f_res_1"]:.5f} GHz',
        f'  dispersive shift chi = {disp["chi"] * 1e3:.2f} MHz',
        f'  Purcell-limited T1 = {purcell_t1(g, delta, kappa) / 1e3:.1f} us',
        ''])
    print(report)
    print('report saved to', save_text(report, 'scq_report'))
