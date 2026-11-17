import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from scqubits_tools import make_transmon, save_output
from simulation_tools import XGate
from two_transmon_tools import TwoTransmonDynamics


if __name__ == '__main__':
    # GHz. Qubit 1 fixed, qubit 2 flux tunable and parked above it.
    ej1, ec1 = 17.0, 0.25    # f01 ~ 5.6 GHz
    ej2, ec2 = 20.0, 0.25    # f01 ~ 6.1 GHz
    g = 0.02                 # 20 MHz exchange coupling
    n_levels = 4

    dyn = TwoTransmonDynamics(make_transmon(ej1, ec1, n_levels=n_levels),
                              make_transmon(ej2, ec2, n_levels=n_levels), g=g)
    spec = dyn.spectrum_summary()

    # ---- 1. spectrum vs qubit-2 frequency: avoided crossing and ZZ
    deltas = np.linspace(-0.8, 0.3, 400)
    sweep = dyn.sweep_delta2(deltas)

    # ---- 2. swap oscillations at resonance
    t_swap, p_swap = dyn.swap_oscillations()

    # ---- 3. X pi pulse on qubit 1, with qubit 2 in |0> and in |1>
    x_runs = {}
    for q2 in (0, 1):
        t_x, states = dyn.apply_gate(XGate(np.pi, sigma=5.0), qubit=0, psi0=dyn.bare(0, q2))
        x_runs[q2] = dyn.qubit_populations(states, 0)

    # ---- 4. two-qubit gates by flux pulses on qubit 2
    gates = {name: dyn.tune_gate(name) for name in ('cz', 'iswap')}
    plus = (dyn.bare(0, 0) + dyn.bare(0, 1) + dyn.bare(1, 0) + dyn.bare(1, 1)).unit()  # |++>
    noisy = {}
    for name, gt in gates.items():
        psi0 = dyn.bare(1, 1) if name == 'cz' else dyn.bare(1, 0)
        noisy[name] = dyn.joint_populations(dyn.run_flux_pulse(gt['t'], gt['delta_t'], psi0))
        # effect of decoherence alone: noisy final state vs the closed-system one, from |++>
        rho = dyn.run_flux_pulse(gt['t'], gt['delta_t'], plus)[-1]
        psi_closed = gt['U'] * plus
        gt['noise_fidelity'] = qt.expect(rho, psi_closed)

    # ---- figures
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    ax.plot(sweep['f2'], sweep['E10'], label='|10> (dressed)')
    ax.plot(sweep['f2'], sweep['E01'], label='|01> (dressed)')
    ax.set(xlabel='bare f2 (GHz)', ylabel='E - E00 (GHz)', title=f'avoided crossing, gap 2g = {2 * g * 1e3:.0f} MHz')
    ax.legend()
    ax = axes[0, 1]
    ax.plot(sweep['f2'], sweep['zz'] * 1e3, label='numerical')
    ax.plot(sweep['f2'], sweep['zz_analytic'] * 1e3, 'k--', label='2nd-order analytic')
    ax.axvline(dyn.f01[1], color='gray', lw=0.8, label='park point')
    ax.set(xlabel='bare f2 (GHz)', ylabel='ZZ (MHz)', title='ZZ = E11 - E10 - E01 + E00', ylim=(-10, 10))
    ax.legend()
    ax = axes[0, 2]
    ax.plot(t_swap, p_swap[:, 1, 0], label='|10>')
    ax.plot(t_swap, p_swap[:, 0, 1], label='|01>')
    ax.set(xlabel='t (ns)', ylabel='population', title='swap oscillations at f2 = f1')
    ax.legend()
    ax = axes[1, 0]
    for q2, pops in x_runs.items():
        ax.plot(t_x, pops[:, 1], label=f'qubit 2 in |{q2}>')
    ax.set(xlabel='t (ns)', ylabel='P(qubit 1 in |1>)', title='X pi on qubit 1 (tuned for qubit 2 in |0>)')
    ax.legend()
    for ax, (name, gt), (a, b) in zip(axes[1, 1:], gates.items(), [((1, 1), (0, 2)), ((1, 0), (0, 1))]):
        pops = noisy[name]
        ax.plot(gt['t'], pops[:, a[0], a[1]], label=f'|{a[0]}{a[1]}>')
        ax.plot(gt['t'], pops[:, b[0], b[1]], label=f'|{b[0]}{b[1]}>')
        ax2 = ax.twinx()
        ax2.plot(gt['t'], gt['delta_t'] * 1e3, 'k:', lw=0.8)
        ax2.set_ylabel('delta2 (MHz)')
        ax.set(xlabel='t (ns)', ylabel='population',
               title=f'{name.upper()}: F = {gt["fidelity"]:.5f}, leakage = {gt["leakage"]:.1e}')
        ax.legend(loc='center right')
    plt.tight_layout()
    fig_path = save_output('two_transmon_dynamics')

    # ---- report
    try:
        ev_scq, ev_ours = dyn.scqubits_check()
        check = [f'  scqubits HilbertSpace: {np.round(ev_scq, 5)}',
                 f'  this model          : {np.round(ev_ours, 5)}']
    except Exception as err:
        # scqubits 4.3.1 + recent SciPy: HilbertSpace's own bare Hamiltonian fails
        # (SciPy now returns csc_array where scqubits expects csc_matrix)
        check = [f'  scqubits HilbertSpace cross-check unavailable ({type(err).__name__}: {err});',
                 '  likely a scqubits / SciPy version mismatch, not this model']

    def gate_lines(name, gt):
        return [f'{name.upper()}: delta2 = {gt["delta2"] * 1e3:.2f} MHz, '
                f'min gap = {gt["gap"] * 1e3:.3f} MHz, t on resonance = {gt["t_on"]:.2f} ns',
                f'  pulse: ramps {gt["t_ramp"]:.1f} ns + flat {gt["t_flat"]:.2f} ns '
                f'(total {gt["t"][-1]:.2f} ns)',
                f'  average gate fidelity (closed, up to virtual Z) = {gt["fidelity"]:.6f}, '
                f'leakage = {gt["leakage"]:.2e}',
                f'  conditional phase = {np.degrees(gt["conditional_phase"]):.2f} deg',
                f'  decoherence alone, |++> input: state fidelity = {gt["noise_fidelity"]:.6f}']

    report = '\n'.join([
        f'Qubit 1 (fixed):   EJ = {ej1}, EC = {ec1} GHz: f01 = {dyn.f01[0]:.4f} GHz, '
        f'alpha = {dyn.anharmonicity[0] * 1e3:.1f} MHz',
        f'Qubit 2 (tunable): EJ = {ej2}, EC = {ec2} GHz: f01 = {dyn.f01[1]:.4f} GHz, '
        f'alpha = {dyn.anharmonicity[1] * 1e3:.1f} MHz',
        f'Coupling g = {g * 1e3:.1f} MHz, detuning f1 - f2 = {(dyn.f01[0] - dyn.f01[1]) * 1e3:.0f} MHz',
        f'Coherence: T1 = {dyn.t1[0] / 1e3:.1f}, {dyn.t1[1] / 1e3:.1f} us; '
        f'T_phi = {dyn.t_phi[0] / 1e3:.1f}, {dyn.t_phi[1] / 1e3:.1f} us',
        '',
        'Dressed spectrum at the park point:',
        f'  f1 = {spec["f1"]:.5f} GHz (q2 in |0>), {spec["f1_q2_excited"]:.5f} GHz (q2 in |1>)',
        f'  f2 = {spec["f2"]:.5f} GHz (q1 in |0>), {spec["f2_q1_excited"]:.5f} GHz (q1 in |1>)',
        f'  ZZ = {spec["zz"] * 1e3:.4f} MHz (analytic {spec["zz_analytic"] * 1e3:.4f} MHz)',
        'Lowest levels above ground (GHz):', *check,
        '',
        f'X pi on qubit 1: final P(q1 = 1) = {x_runs[0][-1, 1]:.5f} (q2 in |0>), '
        f'{x_runs[1][-1, 1]:.5f} (q2 in |1>, drive detuned by ZZ)',
        '',
        *gate_lines('cz', gates['cz']), '',
        *gate_lines('iswap', gates['iswap']),
        ''])
    print(report)
    print('report saved to', save_output('two_transmon_report', text=report))
    print('figure saved to', fig_path)
