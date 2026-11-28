"""
Simulations for Prep Report 4 - Relaxation Times.

Lindblad master-equation models of a qubit (and of a three-level transmon) in the
frame rotating at the qubit frequency. Units: times in us, rates in 1/us,
frequencies in MHz (angular frequencies 2 pi f, in rad/us). Pulses are ideal,
instantaneous rotations: the report is about what happens between them.

    Experiment 1 (T1)       fig_t1 (Q2), fig_t2_vs_tphi (Q3)
    Experiment 2 (Ramsey)   fig_ramsey_shape (Q2); fitting the measured CSVs (Q3-Q4): ramsey_fit.py
    Experiment 3 (echo)     fig_ramsey_vs_echo (Q3)
    The f state             fig_f_state_decay (Q2)

    python relaxation_times.py                  # every figure + report.txt in data/<date_time>/
    python relaxation_times.py --no-show        # save without opening windows
    python relaxation_times.py --t1 30 --t-phi 60

Conventions (as in the report's QuTiP example): |g> = basis(2, 0), |e> = basis(2, 1),
destroy(2) = |g><e| = sigma^-, sigma_z|g> = +|g> (so |g> is the north pole of the
Bloch sphere).
"""
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.linalg import expm
from scipy.optimize import curve_fit

from simulation_tools import save_and_show, save_text, get_output_dir

TWO_PI = 2 * np.pi

# the report's example values (us)
T1 = 50.0
T_PHI = 80.0

G, E = qt.basis(2, 0), qt.basis(2, 1)
SM = qt.destroy(2)  # |g><e|
SX, SY, SZ = qt.sigmax(), qt.sigmay(), qt.sigmaz()
P_E = E * E.dag()
RHO_G = (G * G.dag()).full()


# ------------------------------------------------------------------ two-level model

def t2_theory(t1, t_phi):
    """1/T2 = 1/(2 T1) + 1/T_phi (Eq. 4)."""
    return 1.0 / (0.5 / t1 + 1.0 / t_phi)


def collapse_ops(t1=T1, t_phi=T_PHI):
    """
    L1 = sqrt(1/T1) sigma^- (Eq. 2) and L_phi = sqrt(1/(2 T_phi)) sigma_z (Eq. 3).
    np.inf for either time leaves that operator out. L_phi damps rho_eg at
    2 * 1/(2 T_phi) = 1/T_phi, since sigma_z rho sigma_z flips the sign of rho_eg.
    """
    c_ops = []
    if np.isfinite(t1):
        c_ops.append(np.sqrt(1.0 / t1) * SM)
    if np.isfinite(t_phi):
        c_ops.append(np.sqrt(0.5 / t_phi) * SZ)
    return c_ops


def rotation(angle, phase=0.0):
    """Ideal pulse (2x2 array): rotation by `angle` about cos(phase) x + sin(phase) y."""
    return (-0.5j * angle * (np.cos(phase) * SX + np.sin(phase) * SY)).expm().full()


def _apply(U, rho):
    """U rho U^dag, for one rho (2, 2) or a stack (n, 2, 2)."""
    return U @ rho @ U.conj().T


def _vec(rho):
    """Column-stack density matrices (n, 2, 2) -> (n, 4), as qutip's superoperators expect."""
    return np.swapaxes(rho, -1, -2).reshape(rho.shape[:-2] + (4,))


def _unvec(v):
    return np.swapaxes(v.reshape(v.shape[:-1] + (2, 2)), -1, -2)


def free_propagators(times, detuning=0.0, t1=T1, t_phi=T_PHI):
    """
    Lindblad propagators of the free evolution, one for each time in `times`
    (starting at 0), as an array (len(times), 4, 4) acting on column-stacked
    density matrices. H = (2 pi detuning / 2) sigma_z: the qubit seen from a
    frame rotating `detuning` MHz away from its frequency. Here (and in
    ramsey_signal, echo_signal) `detuning` is the total qubit - drive offset in
    one shot, Delta + delta: the drive detuning Delta you choose plus the
    unknown frequency shift delta of that shot. One ODE solve gives
    every propagator, so a pulse sequence with a different wait per point
    (the echo) needs no extra solves.
    """
    L = qt.liouvillian(0.5 * TWO_PI * detuning * SZ, collapse_ops(t1, t_phi))
    S = qt.propagator(L, np.asarray(times, dtype=float))
    return np.array([s.full() for s in S])


def ramsey_signal(taus, detuning=0.0, t1=T1, t_phi=T_PHI):
    """
    P_e after pi/2_x - wait tau - pi/2_x, starting from |g>, for a qubit
    `detuning` MHz away from the drive. Exact result:
        P_e = 1/2 + 1/2 exp(-tau/T2) cos(2 pi detuning tau)
    (T1 relaxation during the wait moves the Bloch vector along z, which the
    second pulse turns into y, so it does not change P_e).
    """
    x90 = rotation(np.pi / 2)
    S = free_propagators(taus, detuning, t1, t_phi)
    rho = _unvec(S @ _vec(_apply(x90, RHO_G)))
    return np.real(_apply(x90, rho)[:, 1, 1])


def echo_signal(taus, detuning=0.0, t1=T1, t_phi=T_PHI):
    """
    P_e after pi/2_x - tau/2 - pi_x - tau/2 - pi/2_(-x), from |g>. The pi pulse
    mirrors the phase gained in the first half, so the second half undoes it:
    a fixed detuning drops out, P_e = 1/2 + 1/2 exp(-tau/T2). The last pulse is
    about -x so P_e starts at 1, like the Ramsey signal.
    """
    taus = np.asarray(taus, dtype=float)
    S = free_propagators(taus / 2, detuning, t1, t_phi)
    rho = _unvec(S @ _vec(_apply(rotation(np.pi / 2), RHO_G)))
    rho = _apply(rotation(np.pi), rho)
    rho = _unvec(np.einsum('kij,kj->ki', S, _vec(rho)))
    return np.real(_apply(rotation(np.pi / 2, np.pi), rho)[:, 1, 1])


def quasi_static_average(signal, taus, drive_detuning, sigma, n_samples=500, seed=0, **kwargs):
    """
    Monte Carlo average over shots with quasi-static frequency noise, like the
    experiment. Two different detunings:
        drive_detuning  Delta (MHz): chosen by you, the same in every shot
        delta           the qubit's unknown frequency shift: fixed during one
                        shot, different from shot to shot, delta ~ Normal(0, sigma)
    Each shot is `signal(taus, Delta + delta, **kwargs)`; the measurement is
    their mean. The statistical error is ~0.5/sqrt(n_samples) (~2 % for 500);
    `seed` makes the result reproducible.
    """
    if sigma == 0:
        return signal(taus, drive_detuning, **kwargs)
    deltas = np.random.default_rng(seed).normal(0.0, sigma, n_samples)
    return np.mean([signal(taus, drive_detuning + d, **kwargs) for d in deltas], axis=0)


def ramsey_theory(t, drive_detuning, t2, sigma=0.0):
    """
    <P_e> of the Ramsey with drive detuning Delta and delta ~ Normal(0, sigma):
        1/2 + 1/2 exp(-t/T2) exp(-(2 pi sigma t)^2 / 2) cos(2 pi Delta t)
    The Gaussian factor falls to 1/e at t = sqrt(2) / (2 pi sigma).
    """
    return 0.5 + 0.5 * np.exp(-t / t2 - 0.5 * (TWO_PI * sigma * t) ** 2) * np.cos(TWO_PI * drive_detuning * t)


def dephasing_time(sigma):
    """1/e time of the Gaussian dephasing factor exp(-(2 pi sigma t)^2 / 2): sqrt(2) / (2 pi sigma)."""
    return np.sqrt(2) / (TWO_PI * sigma)


def _exp_fit(t, y, with_offset=True):
    """Fit y = a exp(-t/tau) (+ c); the tau guess is where y crosses 1/e of its start."""
    tau0 = t[np.argmin(np.abs(y - y[0] / np.e))] or t[-1] / 3
    if with_offset:
        model = lambda x, a, tau, c: a * np.exp(-x / tau) + c
        p0 = [y[0], tau0, 0.0]
    else:
        model = lambda x, a, tau: a * np.exp(-x / tau)
        p0 = [y[0], tau0]
    popt, pcov = curve_fit(model, t, y, p0=p0)
    return popt, np.sqrt(np.diag(pcov)), model


# ------------------------------------------------------------ Experiment 1 - T1

def fig_t1(t1=T1):
    """Q2: start from |e>, only L1; rho_ee(t) decays as exp(-t/T1)."""
    t = np.linspace(0, 5 * t1, 400)
    res = qt.mesolve(0 * SZ, E, t, c_ops=collapse_ops(t1, np.inf), e_ops=[P_E])
    p_e = np.real(res.expect[0])
    (a, tau, c), (da, dtau, dc), model = _exp_fit(t, p_e)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, log in zip(axes, (False, True)):
        ax.plot(t, p_e, lw=2, label=r'simulation $\rho_{ee}(t)$')
        ax.plot(t, model(t, a, tau, c), 'k--', label=rf'fit $a\,e^{{-t/T_1}}+c$: $T_1$ = {tau:.3f} µs')
        ax.axvline(t1, color='gray', ls=':', label=f't = T1 (set) = {t1:g} µs')
        ax.axhline(np.exp(-1), color='r', ls=':', lw=0.8, label='1/e')
        ax.set(xlabel='t (µs)', ylabel=r'$\rho_{ee}$', yscale='log' if log else 'linear',
               title='log scale: a straight line of slope -1/T1' if log else 'T1 decay from |e⟩')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)
    save_and_show('exp1_q2_t1_decay')
    return ['Experiment 1, Q2 - T1 decay (start in |e>, collapse operator L1 only):',
            f'  set T1 = {t1:g} us, fitted T1 = {tau:.4f} +- {dtau:.1e} us, a = {a:.4f}, c = {c:.1e}']


def fig_t2_vs_tphi(t1=T1, tphi_over_t1=(np.inf, 8, 4, 2, 1, 0.5, 0.2)):
    """
    Q3: start from (|g> + |e>)/sqrt2, L1 + L_phi; fit |rho_eg| = a exp(-t/T2)
    for several T_phi and compare with Eq. 4 and the bound T2 <= 2 T1.
    """
    psi0 = (G + E).unit()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows = []
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(tphi_over_t1)))
    for k, (r, col) in enumerate(zip(tphi_over_t1, colors)):
        t_phi = r * t1
        t2_th = t2_theory(t1, t_phi)
        t = np.linspace(0, 4 * t2_th, 300)
        res = qt.mesolve(0 * SZ, psi0, t, c_ops=collapse_ops(t1, t_phi), e_ops=[SM])
        coh = np.abs(res.expect[0])  # <sigma^-> = Tr(|g><e| rho) = rho_eg
        (a, t2_fit), (_, dt2), model = _exp_fit(t, coh, with_offset=False)
        rows.append((t_phi, t2_fit, dt2, t2_th))
        label = r'$T_\varphi\to\infty$' if np.isinf(t_phi) else rf'$T_\varphi$ = {r:g} $T_1$'
        axes[0].plot(t / t1, coh, color=col, lw=2, label=f'{label}: T2 = {t2_fit / t1:.3f} T1')
        axes[0].plot(t / t1, model(t, a, t2_fit), 'k--', lw=0.7)
    axes[0].set(xlabel='t / T1', ylabel=r'$|\rho_{eg}(t)|$', yscale='log', ylim=(1e-3, 0.6),
                title=r'coherence decay from $(|g\rangle+|e\rangle)/\sqrt{2}$ (dashed: fits)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which='both')

    t_phi, t2_fit, dt2, t2_th = map(np.array, zip(*rows))
    x = t1 / t_phi  # T1 / T_phi = 0 for T_phi -> inf
    xs = np.linspace(0, x.max() * 1.05, 200)
    axes[1].plot(xs, 1 / (0.5 + xs), 'b-', label=r'Eq. 4: $T_2/T_1 = 1/(1/2 + T_1/T_\varphi)$')
    axes[1].errorbar(x, t2_fit / t1, yerr=dt2 / t1, fmt='ro', label='fitted T2 / T1')
    axes[1].axhline(2, color='k', ls='--', label=r'bound $T_2 = 2T_1$ (reached only at $T_\varphi\to\infty$)')
    axes[1].set(xlabel=r'$T_1/T_\varphi$  (pure-dephasing strength; 0 = no pure dephasing)',
                ylabel='T2 / T1', title=r'$T_2 \leq 2T_1$')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    save_and_show('exp1_q3_t2_vs_tphi')

    lines = ['Experiment 1, Q3 - coherence |rho_eg| from (|g>+|e>)/sqrt2, L1 + L_phi:',
             f'  T1 = {t1:g} us;  {"T_phi (us)":>11s} {"T2 fit (us)":>12s} {"Eq.4 (us)":>10s} {"T2/2T1":>8s}']
    for tp, t2f, _, t2t in rows:
        lines.append(f'  {"":13s}{tp:11.4g} {t2f:12.4f} {t2t:10.4f} {t2f / (2 * t1):8.4f}')
    lines.append('  T2/2T1 = 1 only for T_phi -> inf (no pure dephasing); any L_phi makes it smaller.')
    return lines


# --------------------------------------------------------- Experiment 2 - Ramsey

def fig_ramsey_shape(t1=T1, t_phi=T_PHI, drive_detuning=0.2, sigma=0.004, shots=2000, seed=1):
    """
    Q2: the expected Ramsey signal with the drive detuned by drive_detuning
    (Delta, MHz), averaged over a quasi-static frequency shift delta ~
    Normal(0, sigma MHz) and sampled with a finite number of shots,
    fitted with ramsey_fit.fit_ramsey, with the fit parameters marked. The
    data are also saved as a CSV in the format ramsey_fit.py reads.
    """
    from ramsey_fit import fit_ramsey, ramsey_model  # lazy: ramsey_fit sets a matplotlib backend on import

    t = np.linspace(0, 6 * t2_theory(t1, t_phi), 401)
    p = quasi_static_average(ramsey_signal, t, drive_detuning, sigma, t1=t1, t_phi=t_phi)
    y = np.random.default_rng(seed).binomial(shots, np.clip(p, 0, 1)) / shots
    path = save_text('time_us,P_e\n' + '\n'.join(f'{a:.6g},{b:.6g}' for a, b in zip(t, y)),
                     'exp2_q2_simulated_ramsey.csv')
    r = fit_ramsey(t, y, envelope='best')
    A, T2, f, phi, B = r['params']

    tt = np.linspace(0, t[-1], 3000)
    env = A * np.exp(-(tt / T2) ** r['n'])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, y, '.', ms=3, color='gray', label=f'simulated data ({shots} shots per point)')
    ax.plot(tt, ramsey_model(tt, *r['params'], n=r['n']), 'r-', lw=1, label='fit')
    ax.plot(tt, B + env, 'b--', lw=1, label=r'envelope $B \pm A\,e^{-(t/T_2^*)^n}$')
    ax.plot(tt, B - env, 'b--', lw=1)
    ax.axhline(B, color='k', ls=':', lw=1, label=f'B = {B:.3f}')
    # one fringe period, drawn just above the envelope at t = T2* (inside the axes)
    y_mark = B + A / np.e + 0.08
    ax.annotate('', xy=(T2 + 1 / f, y_mark), xytext=(T2, y_mark), arrowprops=dict(arrowstyle='<->'))
    ax.text(T2 + 0.5 / f, y_mark + 0.02, r'period $1/\Delta f$', ha='center', va='bottom', fontsize=9)
    ax.set(xlabel='free-evolution time t (µs)', ylabel=r'$P_e$',
           title=r'Ramsey: $P(t) = B + A\,e^{-(t/T_2^*)^n}\cos(2\pi\Delta f\,t+\varphi)$' '\n'
                 f'fit: A = {A:.3f}, T2* = {T2:.1f} µs, Δf = {f * 1e3:.1f} kHz, '
                 f'φ = {phi:.2f}, B = {B:.3f}, n = {r["n"]:g}')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    save_and_show('exp2_q2_ramsey_shape')
    return ['Experiment 2, Q2 - expected Ramsey signal (simulated, quasi-static noise + shot noise):',
            f'  set: drive detuning Delta = {drive_detuning * 1e3:g} kHz, T1 = {t1:g} us, T_phi = {t_phi:g} us '
            f'(T2 = {t2_theory(t1, t_phi):.2f} us), frequency noise delta ~ N(0, {sigma * 1e3:g} kHz)',
            f'  fit ({r["envelope"]} envelope): A = {A:.4f}, T2* = {T2:.3f} us, df = {f * 1e3:.3f} kHz, '
            f'phi = {phi:.3f}, B = {B:.4f}; RMSE = {r["rmse"]:.4f}',
            f'  data saved to {path} (fit it with: python ramsey_fit.py {path})']


# ------------------------------------------------------ Experiment 3 - Hahn echo

def fig_ramsey_vs_echo(t1=T1, t_phi=T_PHI, fixed_deltas=(0.05, 0.1, 0.2), drive_detuning=0.1, sigma=0.02):
    """
    Q3, with two different detunings kept apart:
        Delta  drive detuning, chosen by you, the same in every shot
        delta  the qubit's unknown frequency shift (quasi-static noise)
    (a) Drive on resonance (Delta = 0), qubit shifted by one fixed delta at a
        time (fixed_deltas, MHz): the Ramsey oscillates at delta, the echo is
        the same for every delta (refocused).
    (b) Drive detuned by Delta (so the fringes are visible) and delta
        different in every shot, delta ~ Normal(0, sigma): the averaged Ramsey
        dephases within sqrt(2)/(2 pi sigma), the echo still decays only with
        the Lindblad T2.
    """
    t2 = t2_theory(t1, t_phi)
    t = np.linspace(0, 3 * t2, 1201)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for d, col in zip(fixed_deltas, ('tab:blue', 'tab:orange', 'tab:green')):
        ax.plot(t, ramsey_signal(t, d, t1, t_phi), color=col, lw=0.8, label=f'Ramsey, δ = {d * 1e3:g} kHz')
        ax.plot(t, echo_signal(t, d, t1, t_phi), color=col, lw=3, ls='--', alpha=0.8,
                label=f'echo, δ = {d * 1e3:g} kHz')
    ax.plot(t, 0.5 + 0.5 * np.exp(-t / t2), 'k:', label=r'$\frac{1}{2}(1+e^{-t/T_2})$')
    ax.set(xlabel='total free-evolution time (µs)', ylabel=r'$P_e$',
           title='drive on resonance (Δ = 0), qubit shifted by a fixed δ:\n'
                 'Ramsey oscillates at δ, echo refocuses (the echo curves lie on top of each other)')
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ram = quasi_static_average(ramsey_signal, t, drive_detuning, sigma, t1=t1, t_phi=t_phi)
    echo = quasi_static_average(echo_signal, t, drive_detuning, sigma, t1=t1, t_phi=t_phi)
    t_deph = dephasing_time(sigma)
    ax.plot(t, ram, 'tab:blue', lw=1, label='Ramsey, mean of 500 shots')
    ax.plot(t, ramsey_theory(t, drive_detuning, t2, sigma), 'k:', lw=1,
            label=r'theory $\frac{1}{2}+\frac{1}{2}e^{-t/T_2}e^{-(2\pi\sigma t)^2/2}\cos 2\pi\Delta t$')
    ax.plot(t, echo, 'tab:red', lw=3, alpha=0.8, label='echo, mean of the same 500 shots')
    ax.plot(t, 0.5 + 0.5 * np.exp(-t / t2), 'k--', lw=1, label=r'$\frac{1}{2}(1+e^{-t/T_2})$')
    ax.set(xlabel='total free-evolution time (µs)', ylabel=r'$P_e$',
           title=f'Δ = {drive_detuning * 1e3:g} kHz, δ random in every shot, δ ~ N(0, σ = {sigma * 1e3:g} kHz):\n'
                 f'Ramsey dephases (√2/2πσ = {t_deph:.1f} µs), echo still decays with T2 = {t2:.1f} µs')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    save_and_show('exp3_q3_ramsey_vs_echo')

    err_r = np.max(np.abs(ram - ramsey_theory(t, drive_detuning, t2, sigma)))
    err_e = np.max(np.abs(echo - (0.5 + 0.5 * np.exp(-t / t2))))
    spread = max(np.max(np.abs(echo_signal(t, d, t1, t_phi) - echo_signal(t, 0.0, t1, t_phi)))
                 for d in fixed_deltas)
    return ['Experiment 3, Q3 - Ramsey vs Hahn echo with a quasi-static frequency shift delta:',
            f'  T1 = {t1:g} us, T_phi = {t_phi:g} us -> T2 = {t2:.2f} us',
            f'  Delta = 0, fixed delta in {[d * 1e3 for d in fixed_deltas]} kHz: '
            f'max |echo(delta) - echo(0)| = {spread:.1e} (refocused)',
            f'  Delta = {drive_detuning * 1e3:g} kHz, delta ~ N(0, sigma = {sigma * 1e3:g} kHz) per shot: Ramsey '
            f'decays with an extra Gaussian factor, 1/e at sqrt(2)/(2 pi sigma) = {t_deph:.2f} us;',
            f'    max |simulation - theory| = {err_r:.1e} (Ramsey), {err_e:.1e} (echo vs exp(-t/T2)) '
            '(Monte Carlo over 500 random delta: Ramsey error ~0.5/sqrt(500) = 2e-2 expected)']


# ---------------------------------------------------------------- the f state

def three_level_hamiltonian(f01=5.0, alpha=-0.25):
    """Eq. 5 in frequency units (GHz, H/h): H = f01 |e><e| + (2 f01 + alpha) |f><f|, basis |g>, |e>, |f>."""
    return f01 * qt.fock_dm(3, 1) + (2 * f01 + alpha) * qt.fock_dm(3, 2)


def three_level_collapse_ops(gamma_eg, gamma_fe, gamma_fg=0.0):
    """sqrt(Gamma_i->j) |j><i| for e->g, f->e and (if gamma_fg) f->g; rates in 1/us."""
    ket = lambda j: qt.basis(3, j)
    c_ops = [np.sqrt(gamma_eg) * ket(0) * ket(1).dag(), np.sqrt(gamma_fe) * ket(1) * ket(2).dag()]
    if gamma_fg:
        c_ops.append(np.sqrt(gamma_fg) * ket(0) * ket(2).dag())
    return c_ops


def simulate_f_decay(t, gamma_eg, gamma_fe, gamma_fg=0.0, f01=5.0, alpha=-0.25):
    """
    rho_gg, rho_ee, rho_ff (arrays) at the times `t` (us, t[0] = 0), starting
    from rho(0) = |f><f|, under the Lindblad equation with the Hamiltonian of
    Eq. 5 (three_level_hamiltonian, converted from GHz to rad/us) and the
    collapse operators of three_level_collapse_ops.

    H oscillates at ~2 pi x 10 GHz, i.e. ~10^7 periods over the 300 us decay,
    which an ODE solver (mesolve) would have to follow step by step. Instead
    the equation d vec(rho)/dt = L vec(rho) (L = the Liouvillian, a 9x9
    matrix) is solved exactly: vec(rho(t + dt)) = exp(L dt) vec(rho(t)).
    """
    H = TWO_PI * 1e3 * three_level_hamiltonian(f01, alpha)  # GHz -> rad/us
    L = qt.liouvillian(H, three_level_collapse_ops(gamma_eg, gamma_fe, gamma_fg)).full()
    rho = qt.fock_dm(3, 2).full()  # initial state |f><f|
    vec = rho.ravel(order='F')     # column-stacked, as qutip's Liouvillian expects
    pops = []
    for dt in np.diff(np.asarray(t, dtype=float), prepend=0.0):
        vec = expm(L * dt) @ vec
        pops.append(np.real(np.diag(vec.reshape(3, 3, order='F'))))
    return list(np.array(pops).T)  # [rho_gg, rho_ee, rho_ff]


def f_decay_theory(t, gamma_eg, gamma_fe, gamma_fg=0.0):
    """
    Closed-form populations from the rate equations, i.e. the diagonal of the
    Lindblad equation (H drops out of the diagonal since it is diagonal itself):
        d rho_ff / dt = -(gamma_fe + gamma_fg) rho_ff
        d rho_ee / dt =  gamma_fe rho_ff - gamma_eg rho_ee
        d rho_gg / dt =  gamma_eg rho_ee + gamma_fg rho_ff
    with rho_ff(0) = 1. With G = gamma_fe + gamma_fg:
        rho_ff = e^{-G t}
        rho_ee = gamma_fe / (G - gamma_eg) (e^{-gamma_eg t} - e^{-G t})
        rho_gg = 1 - rho_ee - rho_ff
    """
    big = gamma_fe + gamma_fg
    p_f = np.exp(-big * t)
    p_e = gamma_fe / (big - gamma_eg) * (np.exp(-gamma_eg * t) - p_f)
    return 1 - p_e - p_f, p_e, p_f


def transmon_rate_ratios(ej=30.0, ec=0.25, ngs=(0.0, 0.25)):
    """
    Charge matrix elements of a transmon (scqubits), which set the decay rates
    for charge-coupled loss (Gamma_i->j ~ |<j|n|i>|^2 for a frequency-flat
    environment): |n_ef|^2/|n_ge|^2 (~2, the sqrt(n) ladder of an oscillator)
    and |n_gf|^2/|n_ge|^2 (zero at ng = 0 by parity, tiny elsewhere).
    """
    import scqubits as scq
    out = []
    for ng in ngs:
        n = np.abs(scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=30, truncated_dim=3)
                   .matrixelement_table('n_operator', evals_count=3))
        out.append((ng, (n[1, 2] / n[0, 1]) ** 2, (n[0, 2] / n[0, 1]) ** 2))
    return out


def fig_f_state_decay(t1=T1, fe_ratio=2.0, fg_ratio=0.2, f01=5.0, alpha=-0.25):
    """
    Q2: start from |f>; (a) f -> e -> g only, (b) also f -> g directly. Rates:
    Gamma_eg = 1/T1, Gamma_fe = fe_ratio Gamma_eg, Gamma_fg = fg_ratio Gamma_eg
    (fg_ratio is exaggerated so the effect is visible; the transmon value,
    printed in the report, is orders of magnitude smaller).
    """
    g_eg = 1.0 / t1
    g_fe, g_fg = fe_ratio * g_eg, fg_ratio * g_eg
    t = np.linspace(0, 6 * t1, 600)
    cases = [('(a) f → e → g', 0.0), (f'(b) f → e → g and f → g (Γfg = {fg_ratio:g} Γeg)', g_fg)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sims = []
    for ax, (title, gfg) in zip(axes, cases):
        sim = simulate_f_decay(t, g_eg, g_fe, gfg, f01, alpha)
        th = f_decay_theory(t, g_eg, g_fe, gfg)
        sims.append(sim)
        for p, p_th, lab, col in zip(sim, th, ('g', 'e', 'f'), ('tab:blue', 'tab:orange', 'tab:green')):
            ax.plot(t, p, color=col, lw=2, label=rf'$\rho_{{{lab}{lab}}}$')
            ax.plot(t, p_th, 'k:', lw=1)
        ax.set(xlabel='t (µs)', ylabel='population', title=title + '\n(dotted: rate equations)')
        ax.legend()
        ax.grid(alpha=0.3)

    # how to tell them apart: rho_gg at short times, t^2 (two steps) vs t (one step)
    ax = axes[2]
    ts = np.logspace(-2, 1.5, 200)
    for (title, gfg), ls in zip(cases, ('-', '--')):
        ax.loglog(ts, simulate_f_decay(np.concatenate([[0], ts]), g_eg, g_fe, gfg, f01, alpha)[0][1:], ls, lw=2,
                  label=r'$\rho_{gg}$ ' + title.split(' ')[0])
    ax.loglog(ts, 0.5 * g_fe * g_eg * ts ** 2, 'k:', label=r'$\frac{1}{2}\Gamma_{fe}\Gamma_{eg}t^2$ (two steps)')
    ax.loglog(ts, g_fg * ts, 'k-.', label=r'$\Gamma_{fg}\,t$ (direct)')
    ax.set(xlabel='t (µs)', ylabel=r'$\rho_{gg}$', title='short times: ground-state population grows as t² vs t')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    save_and_show('f_state_q2_decay')

    i_a, i_b = (np.argmax(s[1]) for s in sims)
    H = three_level_hamiltonian(f01, alpha)
    lines = ['The f state, Q2 - decay from |f>:',
             f'  H/h (Eq. 5, GHz) = diag{tuple(np.round(np.real(H.diag()), 4).tolist())}  '
             f'(f01 = {f01:g} GHz, alpha = {alpha * 1e3:g} MHz); Lindblad evolution with this H, '
             'solved exactly with exp(L dt)',
             f'  Gamma_eg = 1/T1 = {g_eg:.4g}/us, Gamma_fe = {fe_ratio:g} Gamma_eg = {g_fe:.4g}/us, '
             f'case (b): Gamma_fg = {fg_ratio:g} Gamma_eg = {g_fg:.4g}/us',
             f'  rho_ee peak: (a) {sims[0][1][i_a]:.3f} at t = {t[i_a]:.1f} us, '
             f'(b) {sims[1][1][i_b]:.3f} at t = {t[i_b]:.1f} us',
             '  transmon charge matrix elements (scqubits, EJ = 30 GHz, EC = 0.25 GHz):']
    for ng, r_fe, r_fg in transmon_rate_ratios():
        lines.append(f'    ng = {ng:g}: |n_ef|^2/|n_ge|^2 = {r_fe:.3f},  |n_gf|^2/|n_ge|^2 = {r_fg:.2e}')
    return lines


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-show', action='store_true', help='save the figures without opening windows')
    ap.add_argument('--t1', type=float, default=T1, help='T1 in us (default %(default)s)')
    ap.add_argument('--t-phi', type=float, default=T_PHI, help='T_phi in us (default %(default)s)')
    args = ap.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
        warnings.filterwarnings('ignore', message='.*non-interactive.*')

    lines = []
    for block in (fig_t1(args.t1),
                  fig_t2_vs_tphi(args.t1),
                  fig_ramsey_shape(args.t1, args.t_phi),
                  fig_ramsey_vs_echo(args.t1, args.t_phi),
                  fig_f_state_decay(args.t1)):
        lines += block + ['']
    report = '\n'.join(lines)
    print(report)
    print('report saved to', save_text(report, 'report'))
    print('figures in', get_output_dir())


if __name__ == '__main__':
    main()
