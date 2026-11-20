from scqubits_tools import *

rise_fall = {'t_rise': 10.0, 't_fall': 10.0}

if __name__ == '__main__':
    # GHz
    ej = 30
    ec = 0.25
    f_res = 5.0
    kappa = 0.001
    g = 0.1
    detuning_factor = 0.00002
    flux = 1.0

    tmon = make_transmon(ej, ec, flux=flux)
    resonator = make_resonator(f_res=f_res)
    dyn = Transmon_Resonator_Dynamics(tmon, resonator=resonator, g=g, kappa=kappa, q_cap=5e6)
    readout = dyn.readout_summary()
    disp = readout['jc']

    x_gate = XGate(np.pi, sigma=5.0, **rise_fall)
    t1_fit, t_t1, p_t1, fit_t1 = dyn.measure_t1(**rise_fall, pulse=x_gate)
    t2_fit, t_t2, p_t2, fit_t2, env_t2 = dyn.measure_t2(**rise_fall, detuning=dyn.f01 * detuning_factor)

    t_pulse, traj = dyn.apply_gate(x_gate)
    pops = dyn.populations(traj)
    # x_gate.frequency_spectrum(omega_q=dyn.f01*TWO_PI, save_as='x_pi_frequency_spectrum')
    # x_gate.plot_bloch_trajectory()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(t_pulse, pops[:, :3])
    axes[0].set(xlabel='t (ns)', ylabel='population', title='pi pulse')
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
    fft = dyn.ramsey_fft
    f_mhz = fft['freqs'] * 1e3  # GHz -> MHz
    axes[3].plot(f_mhz, fft['spectrum'], label='FFT of Ramsey P(|1>)')
    axes[3].plot([], [], ' ', label=f'f01 dressed = {fft["f01_dressed"]:.6f} GHz')
    axes[3].plot([], [], ' ', label=f'drive frequency = {fft["f_drive"]:.6f} GHz')
    axes[3].axvline(fft['detuning'] * 1e3, color='g', ls=':', lw=2,
                    label=f'detuning (drive - f01 dressed) = {fft["detuning"] * 1e3:.4f} MHz')
    axes[3].axvline(fft['f_peak'] * 1e3, color='orange', ls='--',
                    label=f'FFT peak = {fft["f_peak"] * 1e3:.4f} MHz')
    axes[3].axvline(fft['f_fit'] * 1e3, color='r', ls='-', lw=0.8,
                    label=f'fitted = {fft["f_fit"] * 1e3:.4f} MHz')
    axes[3].set(xlim=(0, 5 * fft['detuning'] * 1e3), xlabel='fringe frequency (MHz)',
                ylabel='normalized |FFT|', title='Ramsey spectrum (frame of the drive)')
    axes[3].legend(fontsize=7)
    plt.tight_layout()
    fig_path = save_output('scq_dynamics')

    report = '\n'.join([
        f'Transmon: EJ = {ej} GHz, EC = {ec} GHz (EJ/EC = {ej / ec:.0f})',
        f'  f01 = {dyn.f01:.4f} GHz, anharmonicity = {dyn.anharmonicity * 1e3:.1f} MHz',
        '',
        'Coherence (scqubits noise models):',
        f'  T1 = {dyn.t1 / 1e3:.2f} us, T_phi = {dyn.t_phi / 1e3:.2f} us, '
        f'T2 = {dyn.t2 / 1e3:.2f} us',
        f'  with Purcell decay (expected): T1 = {dyn.t1_expected / 1e3:.2f} us, '
        f'T2 = {dyn.t2_expected / 1e3:.2f} us',
        f'  simulated fits (transmon + resonator): T1 = {t1_fit / 1e3:.2f} us, T2 = {t2_fit / 1e3:.2f} us',
        '',
        'Ramsey fringe frequency (the qubit seen from the frame rotating at the drive):',
        f'  drive = {fft["f_drive"]:.6f} GHz, dressed f01 = {fft["f01_dressed"]:.6f} GHz, '
        f'set detuning = {fft["detuning"] * 1e3:.4f} MHz',
        f'  FFT peak = {fft["f_peak"] * 1e3:.4f} MHz, fitted = {fft["f_fit"] * 1e3:.4f} MHz '
        f'(fitted - set = {(fft["f_fit"] - fft["detuning"]) * 1e6:.2f} kHz)',
        f'  => qubit frequency seen by the Ramsey = drive - fitted = {fft["f_drive"] - fft["f_fit"]:.6f} GHz',
        '',
        *dyn.noise_channel_report(),
        '',
        'pi pulse final populations |0>,|1>,|2>,...:',
        f'  {np.round(pops[-1], 5)}',
        '',
        f'Resonator: f_res = {f_res} GHz, g = {g * 1e3:.0f} MHz, kappa = {kappa * 1e3:.1f} MHz',
        f'  detuning = {dyn.tr_detuning * 1e3:.0f} MHz, dressed f01 = {disp["f01"]:.4f} GHz',
        f'  f_res(|0>) = {disp["f_res_0"]:.5f} GHz, f_res(|1>) = {disp["f_res_1"]:.5f} GHz',
        f'  dispersive shift chi: exact JC = {disp["chi"] * 1e3:.3f} MHz, '
        f'H_disp = {readout["disp"]["chi"] * 1e3:.3f} MHz, '
        f'analytic chi_0 - chi_1/2 = {readout["analytic"]["chi"] * 1e3:.3f} MHz',
        f'  critical photon number n_crit = Delta^2/(4 g^2) = {readout["n_crit"]:.0f}',
        f'  Purcell-limited T1 = {dyn.purcell_t1() / 1e3:.1f} us',
        '      Purcell decay: the qubit is slightly hybridized with the resonator, so it',
        '      leaks energy out through the resonator\'s output line at kappa (g/delta)^2.',
        '      Not a scqubits noise channel; the simulation gets it from the resonator',
        '      decay sqrt(2 pi kappa) a, see "with Purcell decay" above.',
        ''])
    print(report)
    print('report saved to', save_output('scq_report', text=report))
    print('figure saved to', fig_path)
