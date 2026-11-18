"""
Full parameter test of the transmon + resonator model in transmon_resonator_times.py.

Every run writes to test_results/<YYYY-MM-DD_HH-MM>/ :
    01_energy_levels_transmon.png   transmon levels vs flux, EJ, EC, ng
    02_energy_levels_resonator.png  joint (Jaynes-Cummings) levels vs f_res, g, flux
    03_spectroscopy_vs_flux.png     qubit spectroscopy map and resonator lines vs flux
    04_sweep_<param>.png            f01, theory residuals, chi, T1, T_phi, T2 vs one parameter
    05_map_flux_vs_ec.png           f01, T1, T_phi, T2 over (flux, EC)
    06_theory_vs_calculated_f01.png f01: numerical vs theory, with error vs EJ/EC, g and detuning
    07_simulated_t1_*.png           (only with --simulate) Lindblad-simulated T1 vs expected
    sweep_<param>.csv, summary.txt  the raw numbers and the theory-vs-calculation errors

Every other parameter is held at BASE while one is swept.

    python test_parameter_sweeps.py              # analytic sweeps (about a few minutes)
    python test_parameter_sweeps.py --quick      # coarser grids
    python test_parameter_sweeps.py --simulate   # also run the master-equation T1 fits
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')  # save figures only, no windows

import argparse
import sys
import csv
import datetime
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as sc
import scqubits as scq
from scipy.optimize import least_squares
from scqubits.core import units as scq_units
from scqubits.core.noise import CONSTANTS, NOISE_PARAMS

from transmon_resonator_times import (Transmon_Resonator_Dynamics, make_transmon,
                                      make_resonator, TWO_PI)

scq.settings.PROGRESSBAR_DISABLED = True

# GHz; times in ns. d = 0.1 so that EJ(flux) never reaches zero (needed for the flux sweeps).
BASE = dict(ej=30.0, ec=0.25, flux=0.0, d=0.1, ng=0.0,
            f_res=5.0, g=0.1, kappa=0.001, n_levels=4, n_res=12, ncut=30)

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')


# ---------------------------------------------------------------------- helpers

def make_run_dir():
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    path = os.path.join(OUT_ROOT, stamp)
    os.makedirs(path, exist_ok=True)
    return path


def build(**kw):
    p = {**BASE, **kw}
    tmon = make_transmon(p['ej'], p['ec'], ng=p['ng'], ncut=p['ncut'],
                         n_levels=p['n_levels'], flux=p['flux'], d=p['d'])
    res = make_resonator(p['f_res'], p['n_res'])
    return Transmon_Resonator_Dynamics(tmon, res, g=p['g'], kappa=p['kappa'])


def ej_eff(ej, flux, d):
    return ej * np.sqrt(np.cos(np.pi * flux) ** 2 + d ** 2 * np.sin(np.pi * flux) ** 2)


def measure(**kw):
    """Everything we plot, for one parameter set (overrides of BASE)."""
    p = {**BASE, **kw}
    dyn = build(**kw)
    rs = dyn.readout_summary()
    delta = dyn.tr_detuning
    f01 = dyn.f01
    out = dict(
        f01=f01, alpha=dyn.anharmonicity,
        # theory for the bare qubit: Koch et al. E_01 = sqrt(8 EJ EC) - EC
        f01_koch=np.sqrt(8 * ej_eff(p['ej'], p['flux'], p['d']) * p['ec']) - p['ec'],
        f01_dressed=dyn.f01_dressed,
        # dressed qubit frequency, theory: 2nd-order perturbation and exact one-excitation JC
        f01_pert2=f01 + p['g'] ** 2 / delta,
        f01_jc_exact=0.5 * (f01 + p['f_res']) + np.sign(delta) * np.sqrt(delta ** 2 / 4 + p['g'] ** 2),
        chi_jc=rs['jc']['chi'], chi_disp=rs['disp']['chi'], chi_analytic=dyn.chi,
        f_res_0=rs['jc']['f_res_0'], f_res_1=rs['jc']['f_res_1'],
        n_crit=dyn.n_crit, detuning=delta,
        t1=dyn.t1, t_phi=dyn.t_phi, t2=dyn.t2,
        t1_purcell=dyn.purcell_t1(), t1_expected=dyn.t1_expected, t2_expected=dyn.t2_expected,
    )
    for c in dyn.noise_channels:
        out[c] = getattr(dyn.tmon, c)()
    return out


def sweep(param, values, **fixed):
    """Run measure() over `values` of one parameter; failed points become NaN."""
    rows = []
    for v in values:
        try:
            rows.append(measure(**{**fixed, param: v}))
        except Exception as e:  # keep going: report and leave a gap
            print(f'    {param}={v:.4g} failed: {type(e).__name__}: {e}')
            rows.append({})
    keys = sorted({k for r in rows for k in r})
    rec = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}
    rec[param] = np.asarray(values, dtype=float)
    return rec


def save_csv(path, rec, first):
    keys = [first] + [k for k in rec if k != first]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(rec[first])):
            w.writerow([rec[k][i] for k in keys])


def finite(a):
    a = np.array(a, dtype=float)
    a[~np.isfinite(a)] = np.nan
    return a


def save_fig(fig, run_dir, name):
    fig.tight_layout()
    path = os.path.join(run_dir, name)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('  saved', name)


# ------------------------------------------------------- 01 transmon energy levels

def fig_transmon_levels(run_dir, quick):
    n_pts = 40 if quick else 120
    n = BASE['n_levels']

    def levels(**kw):
        p = {**BASE, **kw}
        t = make_transmon(p['ej'], p['ec'], ng=p['ng'], ncut=p['ncut'], n_levels=n,
                          flux=p['flux'], d=p['d'])
        e = t.eigenvals(n)
        return e - e[0]

    panels = [('flux', np.linspace(0, 0.5, n_pts), 'flux (Φ0)'),
              ('ej', np.linspace(5, 60, n_pts), 'EJ (GHz)'),
              ('ec', np.linspace(0.1, 0.6, n_pts), 'EC (GHz)'),
              ('ng', np.linspace(-1, 1, n_pts), 'offset charge ng')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (name, xs, xl) in zip(axes.ravel(), panels):
        e = np.array([levels(**{name: x}) for x in xs])
        for j in range(1, n):
            ax.plot(xs, e[:, j], label=f'E{j} - E0')
        ax.axhline(BASE['f_res'], color='gray', ls='--', lw=0.8, label='f_res')
        ax.set(xlabel=xl, ylabel='energy (GHz)', title=f'transmon levels vs {name}')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(fig, run_dir, '01_energy_levels_transmon.png')


# ------------------------------------------------------ 02 resonator-coupled levels

# bare states |transmon level q, photon number n> whose dressed energies are tracked and labelled
TRACKED_STATES = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]


def joint_levels(**kw):
    """Dressed energies (GHz, relative to the dressed |0,0>) of the tracked bare states.

    Each bare state |q, n> is followed through the sweep as the eigenstate of the full
    transmon + resonator Hamiltonian with the largest overlap with it, so every curve keeps
    its label through the avoided crossings.
    """
    dyn = build(**kw)
    n_q, n_r = BASE['n_levels'], BASE['n_res']
    evals, evecs = dyn.readout_hamiltonian().eigenstates()
    bare = lambda q, n: qt.tensor(qt.basis(n_q, q), qt.basis(n_r, n))
    dressed = lambda q, n: evals[int(np.argmax([abs(bare(q, n).overlap(v)) for v in evecs]))]
    e0 = dressed(0, 0)
    return [dressed(q, n) - e0 for q, n in TRACKED_STATES]


def fig_resonator_levels(run_dir, quick):
    n_pts = 40 if quick else 100
    panels = [('f_res', np.linspace(3, 11, n_pts), 'resonator frequency f_res (GHz)'),
              ('g', np.linspace(0.0, 0.4, n_pts), 'coupling g (GHz)'),
              ('flux', np.linspace(0, 0.5, n_pts), 'flux (Φ0)')]
    colors = plt.cm.tab10(np.arange(len(TRACKED_STATES)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (name, xs, xl) in zip(axes, panels):
        e = np.array([joint_levels(**{name: x}) for x in xs])
        for k, (q, n) in enumerate(TRACKED_STATES):
            ax.plot(xs, e[:, k], lw=1.4, color=colors[k], label=f'|{q},{n}⟩')
        ax.set(xlabel=xl, ylabel='E - E(|0,0⟩) (GHz)',
               title=f'transmon ⊗ resonator levels vs {name}\n(avoided crossings at f01 = f_res)')
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=len(labels),
               title='|q, n⟩ = dominant bare state: transmon level q, n resonator photons '
                     '(tracked through the avoided crossings)')
    save_fig(fig, run_dir, '02_energy_levels_resonator.png')


# ------------------------------------------------------- 03 spectroscopy vs flux

def fig_spectroscopy(run_dir, quick):
    n_pts = 60 if quick else 201
    fluxes = np.linspace(0, 0.5, n_pts)
    n = BASE['n_levels']
    lines = np.zeros((n_pts, n - 1))      # E_j - E_0
    weights = np.zeros((n_pts, n - 1))    # |<0|n|j>|^2
    for i, fl in enumerate(fluxes):
        t = make_transmon(BASE['ej'], BASE['ec'], ng=BASE['ng'], ncut=BASE['ncut'],
                          n_levels=n, flux=fl, d=BASE['d'])
        e = t.eigenvals(n)
        nm = t.matrixelement_table('n_operator', evals_count=n)
        lines[i] = e[1:] - e[0]
        weights[i] = np.abs(nm[0, 1:]) ** 2

    freqs = np.linspace(0, 1.1 * lines.max(), 600)
    width = 0.03  # GHz, Lorentzian linewidth
    img = np.zeros((len(freqs), n_pts))
    for k in range(n - 1):
        w = weights[:, k] / weights[:, 0].max()
        img += w[None, :] * width ** 2 / ((freqs[:, None] - lines[None, :, k]) ** 2 + width ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    m = ax.pcolormesh(fluxes, freqs, np.log10(img + 1e-4), shading='auto', cmap='magma')
    ax.axhline(BASE['f_res'], color='c', ls='--', lw=0.8, label='bare resonator')
    ax.set(xlabel='flux (Φ0)', ylabel='drive frequency (GHz)',
           title='qubit spectroscopy vs flux (line weight = |<0|n|j>|²)')
    ax.legend(loc='upper right')
    fig.colorbar(m, ax=ax, label='log10 absorption')

    # resonator lines from the full JC model: shows the avoided crossing at f01 = f_res
    rec = sweep('flux', fluxes[::2 if not quick else 1])
    ax = axes[1]
    ax.plot(rec['flux'], rec['f_res_0'], 'b.-', ms=3, label='resonator, qubit in |0>')
    ax.plot(rec['flux'], rec['f_res_1'], 'r.-', ms=3, label='resonator, qubit in |1>')
    ax.plot(rec['flux'], rec['f01'], 'k--', label='bare f01')
    ax.plot(rec['flux'], rec['f01_dressed'], 'g--', label='dressed f01')
    ax.axhline(BASE['f_res'], color='gray', ls=':', label='bare f_res')
    ax.set(xlabel='flux (Φ0)', ylabel='frequency (GHz)',
           title='resonator spectroscopy vs flux (exact JC)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, run_dir, '03_spectroscopy_vs_flux.png')


# ----------------------------------------------------------- 04 parameter sweeps

def plot_sweep(run_dir, param, rec, xlabel, xlog=False):
    x = rec[param]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    ax = ax.ravel()

    ax[0].plot(x, rec['f01'], label='bare f01 (numerical diagonalization)')
    ax[0].plot(x, rec['f01_koch'], '--', label='theory sqrt(8 EJ EC) - EC')
    ax[0].plot(x, rec['f01_dressed'], label='dressed f01 (JC)')
    ax[0].set(ylabel='f01 (GHz)', title='qubit frequency')

    ax[1].plot(x, (rec['f01'] - rec['f01_koch']), label='bare: numerical - Koch theory')
    ax[1].plot(x, (rec['f01_dressed'] - rec['f01_pert2']), label='dressed: JC - (f01 + g²/Δ)')
    ax[1].plot(x, (rec['f01_dressed'] - rec['f01_jc_exact']), '--',
               label='dressed: JC - exact 2-level JC')
    ax[1].set(ylabel='difference (GHz)', title='theory vs calculated f01')

    ax[2].plot(x, rec['chi_jc'], label='exact JC')
    ax[2].plot(x, rec['chi_disp'], '--', label='dispersive H')
    ax[2].plot(x, rec['chi_analytic'], ':', label='analytic χ0 - χ1/2')
    ax[2].set(ylabel='χ (GHz)', title='dispersive shift')

    for k in rec:
        if k.startswith('t1_') and k not in ('t1_purcell', 't1_expected'):
            ax[3].plot(x, finite(rec[k]) / 1e3, ':', label=k)
    ax[3].plot(x, finite(rec['t1']) / 1e3, 'k-', label='T1 total (noise models, no Purcell)')
    ax[3].plot(x, finite(rec['t1_purcell']) / 1e3, 'r--', label='T1 Purcell')
    ax[3].plot(x, finite(rec['t1_expected']) / 1e3, 'b-', lw=2, label='T1 expected (both)')
    ax[3].set(ylabel='T1 (µs)', yscale='log', title='relaxation')

    for k in rec:
        if k.startswith('tphi_'):
            ax[4].plot(x, finite(rec[k]) / 1e3, ':', label=k)
    ax[4].plot(x, finite(rec['t_phi']) / 1e3, 'k-', label='T_phi total')
    ax[4].set(ylabel='T_phi (µs)', yscale='log', title='pure dephasing')

    ax[5].plot(x, finite(rec['t2']) / 1e3, 'k-', label='T2 (noise models, no Purcell)')
    ax[5].plot(x, finite(rec['t2_expected']) / 1e3, 'b-', lw=2, label='T2 expected (with Purcell)')
    ax[5].plot(x, 2 * finite(rec['t1_expected']) / 1e3, 'r:', label='2 T1 limit')
    ax[5].set(ylabel='T2 (µs)', yscale='log', title='decoherence')

    for a in ax:
        a.set_xlabel(xlabel)
        a.grid(alpha=0.3, which='both')
        a.legend(fontsize=7)
        if xlog:
            a.set_xscale('log')
    fig.suptitle(f'sweep of {param}; everything else at BASE = '
                 + ', '.join(f'{k}={v}' for k, v in BASE.items() if k != param), fontsize=9)
    save_fig(fig, run_dir, f'04_sweep_{param}.png')


def run_sweeps(run_dir, quick):
    s = (lambda full, small: small if quick else full)
    sweeps = [
        ('flux', np.linspace(0, 2.5, s(36, 11)), 'flux (Φ0)', False),
        ('ej', np.linspace(20, 60, s(21, 9)), 'EJ (GHz)', False),
        ('ec', np.linspace(0.15, 1.5, s(18, 8)), 'EC (GHz)', False),
        ('ng', np.linspace(0, 1, s(11, 6)), 'offset charge ng', False),
        ('g', np.linspace(0.02, 1, s(15, 8)), 'g (GHz)', False),
        ('kappa', np.logspace(-4, -1, s(13, 7)), 'kappa (GHz)', True),
        ('f_res', np.linspace(3, 11, s(15, 8)), 'f_res (GHz)', False),
    ]
    results = {}
    for param, values, xlabel, xlog in sweeps:
        t0 = time.time()
        print(f'  sweeping {param} ({len(values)} points)...')
        rec = sweep(param, values)
        results[param] = rec
        save_csv(os.path.join(run_dir, f'sweep_{param}.csv'), rec, param)
        plot_sweep(run_dir, param, rec, xlabel, xlog)
        print(f'    done in {time.time() - t0:.0f} s')
    return results


# --------------------------------------------------------------- 05 2D map

def fig_map(run_dir, quick):
    fluxes = np.linspace(0, 0.45, 8 if quick else 12)
    ecs = np.linspace(0.15, 0.5, 6 if quick else 9)
    names = ['f01', 't1', 't_phi', 't2']
    grids = {k: np.full((len(ecs), len(fluxes)), np.nan) for k in names}
    for i, ec in enumerate(ecs):
        for j, fl in enumerate(fluxes):
            try:
                m = measure(ec=ec, flux=fl)
                for k in names:
                    grids[k][i, j] = m[k]
            except Exception as e:
                print(f'    map point ec={ec:.3g} flux={fl:.3g} failed: {e}')
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, k in zip(axes, names):
        z = grids[k] if k == 'f01' else np.log10(finite(grids[k]) / 1e3)
        m = ax.pcolormesh(fluxes, ecs, z, shading='auto', cmap='viridis')
        fig.colorbar(m, ax=ax, label='GHz' if k == 'f01' else f'log10 {k} (µs)')
        ax.set(xlabel='flux (Φ0)', ylabel='EC (GHz)', title=k)
    save_fig(fig, run_dir, '05_map_flux_vs_ec.png')


# ------------------------------------------------- 06 theory vs calculated f01

def fig_theory_vs_calc(run_dir, results, quick):
    """f01 against theory over a wide EJ/EC range, plus the dressed-frequency checks."""
    ratio = np.linspace(5, 120, 12 if quick else 40)  # EJ/EC
    ec = 0.25
    num, koch = [], []
    for r in ratio:
        t = make_transmon(r * ec, ec, ng=0.0, ncut=BASE['ncut'], n_levels=4, flux=0.0, d=0.0)
        e = t.eigenvals(4)
        num.append(e[1] - e[0])
        koch.append(np.sqrt(8 * r * ec * ec) - ec)
    num, koch = np.array(num), np.array(koch)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    ax.plot(ratio, num, label='numerical (diagonalization)')
    ax.plot(ratio, koch, '--', label='theory sqrt(8 EJ EC) - EC')
    ax.set(xlabel='EJ/EC', ylabel='f01 (GHz)', title='bare f01 (EC = 0.25 GHz, ng = 0)')
    ax.legend()
    ax.grid(alpha=0.3)
    axes[0].set_title('bare f01 vs EJ/EC')

    # dressed f01 vs g
    g_rec = results['g']
    ax = axes[1]
    ax.plot(g_rec['g'], (g_rec['f01_dressed'] - g_rec['f01']), 'k-', label='JC numerical')
    ax.plot(g_rec['g'], (g_rec['f01_pert2'] - g_rec['f01']), '--', label='2nd order g²/Δ')
    ax.plot(g_rec['g'], (g_rec['f01_jc_exact'] - g_rec['f01']), ':', lw=2, label='exact 2-level JC')
    ax.set(xlabel='g (GHz)', ylabel='Lamb shift of f01 (GHz)', title='dressed f01 - bare f01 vs g')
    ax.legend()
    ax.grid(alpha=0.3)

    # dressed f01 vs resonator detuning
    f_rec = results['f_res']
    ax = axes[2]
    ax.plot(f_rec['detuning'], (f_rec['f01_dressed'] - f_rec['f01']), 'k-', label='JC numerical')
    ax.plot(f_rec['detuning'], (f_rec['f01_pert2'] - f_rec['f01']), '--', label='2nd order g²/Δ')
    ax.plot(f_rec['detuning'], (f_rec['f01_jc_exact'] - f_rec['f01']), ':', lw=2,
            label='exact 2-level JC')
    ax.set(xlabel='detuning Δ = f01 - f_res (GHz)', ylabel='Lamb shift of f01 (GHz)',
           title='dressed f01 - bare f01 vs detuning')
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, run_dir, '06_theory_vs_calculated_f01.png')
    return ratio, num - koch


# ------------------------------------------------- 07 simulated T1 (master eq.)

def fig_simulated_t1(run_dir, quick):
    """Lindblad-simulated T1 fit vs expected (scqubits + Purcell), small resonator space."""
    from transmon_resonator_times import Transmon_Resonator_Dynamics  # noqa: F401
    rise_fall = {'t_rise': 10.0, 't_fall': 10.0}
    n_res = 5  # the Liouvillian is (n_levels * n_res)^2 wide; keep it small
    studies = [('kappa', [3e-4, 1e-3, 3e-3, 1e-2]), ('g', [0.05, 0.1, 0.2])]
    fig, axes = plt.subplots(1, len(studies), figsize=(12, 5))
    lines = []
    for ax, (name, vals) in zip(axes, studies):
        sim, exp = [], []
        for v in vals:
            dyn = build(**{name: v, 'n_res': n_res})
            t1_fit, *_ = dyn.measure_t1(**rise_fall)
            sim.append(t1_fit / 1e3)
            exp.append(dyn.t1_expected / 1e3)
            lines.append(f'{name}={v:g}: T1 fit = {t1_fit / 1e3:.2f} us, expected = '
                         f'{dyn.t1_expected / 1e3:.2f} us, noise models only = {dyn.t1 / 1e3:.2f} us')
            print('    ' + lines[-1])
        ax.plot(vals, exp, 'bo-', label='expected (noise-model T1 + Purcell)')
        ax.plot(vals, sim, 'rx--', label='master-equation fit')
        ax.set(xlabel=name, ylabel='T1 (µs)', xscale='log' if name == 'kappa' else 'linear',
               yscale='log', title=f'simulated T1 vs {name}')
        ax.legend()
        ax.grid(alpha=0.3, which='both')
    save_fig(fig, run_dir, '07_simulated_t1.png')
    return lines


# ------------------------------------------------ 08 noise channels vs theory (fits)
#
# For every scqubits noise channel the rate is computed numerically (scqubits, full
# charge-basis diagonalisation) over sweeps of EJ, EC, flux, asymmetry d and offset charge,
# and compared with the closed-form physical formula evaluated with the transmon
# approximations (Koch et al. 2007):
#     f01 = sqrt(8 EJ EC) - EC,  <0|n|1> = (EJ/8EC)^(1/4)/sqrt2,  <0|sin phi|1> = (2EC/EJ)^(1/4)
# The formula's physical constant(s) (Q_cap, M, Z, A_flux, ...) are then fitted to the
# scqubits numbers (log-space least squares over all sweeps of a channel at once) and
# compared with the scqubits defaults. Rates are in 1/ns.

LOG_FACTOR = np.sqrt(2 * abs(np.log(NOISE_PARAMS['omega_low'] * NOISE_PARAMS['t_exp'])))
T_KELVIN = NOISE_PARAMS['T']


def _eje(pt):
    return pt['ej'] * np.sqrt(np.cos(np.pi * pt['flux']) ** 2 + pt['d'] ** 2 * np.sin(np.pi * pt['flux']) ** 2)


def _f01(pt):
    return np.sqrt(8 * _eje(pt) * pt['ec']) - pt['ec']


def _coth(f_ghz):
    """coth(h f / 2 k T): the sum of the absorption and emission spectral densities."""
    return 1.0 / np.tanh(sc.h * f_ghz * 1e9 / (2 * sc.k * T_KELVIN))


def _n01_sq(pt):
    return np.sqrt(_eje(pt) / (8 * pt['ec'])) / 2


def th_capacitive(pt, q0, power):
    # Gamma = |<0|n|1>|^2 * 2*8 EC * 2pi / Q(f) * coth,  Q(f) = Q0 (6 GHz / f)^power
    f = _f01(pt)
    return _n01_sq(pt) * 32 * np.pi * pt['ec'] / (q0 * (6.0 / f) ** power) * _coth(f)


def th_flux_bias_line(pt, m):
    # Gamma = |<0|dH/dflux|1>|^2 * 2 (2pi)^2 M^2 omega hbar / Z * coth, with
    # dH/dflux = -pi EJmax d / sqrt(cos^2 + d^2 sin^2) sin(phi) + (cos phi term, no 0-1 element)
    f = _f01(pt)
    c2 = np.cos(np.pi * pt['flux']) ** 2 + pt['d'] ** 2 * np.sin(np.pi * pt['flux']) ** 2
    m01_sq = (np.pi * pt['ej'] * pt['d'] / np.sqrt(c2)) ** 2 * np.sqrt(2 * pt['ec'] / _eje(pt))
    return (m01_sq * 2 * (2 * np.pi) ** 2 * m ** 2 * (2 * np.pi * f) * sc.hbar
            / NOISE_PARAMS['R_0'] * _coth(f) * scq_units.to_standard_units(1) ** 2)


def th_charge_impedance(pt, z):
    # Gamma = |<0|n|1>|^2 * 2 omega / Q_c * coth,  Q_c = R_k / (8 pi Z)
    f = _f01(pt)
    q_c = CONSTANTS['R_k'] / (8 * np.pi * z)
    return _n01_sq(pt) * 2 * (2 * np.pi * f) / q_c * _coth(f)


def th_tphi_flux(pt, a_flux):
    # Gamma = 2pi A sqrt(2 |ln(w_low t_exp)|) |d f01 / d flux|
    c2 = np.cos(np.pi * pt['flux']) ** 2 + pt['d'] ** 2 * np.sin(np.pi * pt['flux']) ** 2
    dej = np.pi * pt['ej'] * np.cos(np.pi * pt['flux']) * np.sin(np.pi * pt['flux']) * (pt['d'] ** 2 - 1) / np.sqrt(c2)
    df01 = np.sqrt(2 * pt['ec'] / _eje(pt)) * dej  # d/dEJ [sqrt(8 EJ EC)] = sqrt(2 EC / EJ)
    return 2 * np.pi * a_flux * LOG_FACTOR * np.abs(df01)


def th_tphi_cc(pt, a_cc):
    # Gamma = 2pi A sqrt(2 |ln|) |d f01 / d EJ|,  d f01 / d EJ = sqrt(2 EC / EJ)
    return 2 * np.pi * a_cc * LOG_FACTOR * np.sqrt(2 * pt['ec'] / _eje(pt))


def th_tphi_ng(pt, a_ng):
    # Koch charge dispersion: E_m(ng) = E_m(1/4) - eps_m/2 cos(2 pi ng),
    #   eps_m = (-1)^m EC 2^(4m+5)/m! sqrt(2/pi) (EJ/2EC)^(m/2+3/4) exp(-sqrt(8 EJ/EC))
    # Gamma = 2pi A sqrt(2 |ln|) |d f01 / d ng| = 2pi A sqrt(2 |ln|) pi |eps_1 - eps_0| |sin(2 pi ng)|
    eje, ec = _eje(pt), pt['ec']
    eps = lambda m: ((-1) ** m * ec * 2 ** (4 * m + 5) / math.factorial(m) * np.sqrt(2 / np.pi)
                     * (eje / (2 * ec)) ** (m / 2 + 0.75) * np.exp(-np.sqrt(8 * eje / ec)))
    return 2 * np.pi * a_ng * LOG_FACTOR * np.pi * np.abs(eps(1) - eps(0)) * np.abs(np.sin(2 * np.pi * pt['ng']))


_OP = dict(ej=30.0, ec=0.25, flux=0.2, d=0.1, ng=0.0)


def _sweeps(quick, **ranges):
    n = (lambda full: max(6, full // 2) if quick else full)
    return {k: np.linspace(lo, hi, n(cnt)) for k, (lo, hi, cnt) in ranges.items()}


def noise_channel_specs(quick):
    common = dict(ej=(20, 60, 17), ec=(0.15, 0.5, 15), flux=(0.0, 0.45, 16), d=(0.02, 0.5, 14))
    return {
        't1_capacitive': dict(
            theory=th_capacitive, params=dict(Q_cap0=1e6, power=0.7), op=_OP,
            sweeps=_sweeps(quick, **common),
            formula='Γ = |<0|n|1>|² · 16 EC · 2π / Q(f) · coth(hf/2kT),   Q(f) = Q0 (6 GHz / f)^power'),
        't1_flux_bias_line': dict(
            theory=th_flux_bias_line, params=dict(M=NOISE_PARAMS['M']), op=_OP,
            sweeps=_sweeps(quick, **common),
            formula='Γ = |<0|dH/dΦ|1>|² · 2 (2π)² M² ω ħ / Z · coth(hf/2kT);  '
                    '<0|dH/dΦ|1> = π EJmax d / sqrt(cos² + d² sin²) · (2EC/EJ)^(1/4)'),
        't1_charge_impedance': dict(
            theory=th_charge_impedance, params=dict(Z=NOISE_PARAMS['R_0']), op=_OP,
            sweeps=_sweeps(quick, **common),
            formula='Γ = |<0|n|1>|² · 2 ω / Q_c · coth(hf/2kT),   Q_c = R_k / (8π Z)'),
        'tphi_1_over_f_flux': dict(
            theory=th_tphi_flux, params=dict(A_flux=NOISE_PARAMS['A_flux']), op=_OP,
            sweeps=_sweeps(quick, ej=(20, 60, 17), ec=(0.15, 0.5, 15), flux=(0.02, 0.48, 16), d=(0.02, 0.5, 14)),
            formula='Γφ = 2π A_flux sqrt(2|ln(ω_low t_exp)|) |∂f01/∂Φ|'),
        'tphi_1_over_f_cc': dict(
            theory=th_tphi_cc, params=dict(A_cc=NOISE_PARAMS['A_cc']), op=_OP,
            sweeps=_sweeps(quick, ej=(20, 60, 17), ec=(0.15, 0.5, 15), flux=(0.0, 0.45, 16)),
            formula='Γφ = 2π A_cc sqrt(2|ln(ω_low t_exp)|) |∂f01/∂EJ|,   ∂f01/∂EJ = sqrt(2 EC / EJ)'),
        'tphi_1_over_f_ng': dict(
            theory=th_tphi_ng, params=dict(A_ng=NOISE_PARAMS['A_ng']),
            op=dict(ej=8.0, ec=0.25, flux=0.0, d=0.1, ng=0.25),  # low EJ/EC: measurable charge dispersion
            sweeps=_sweeps(quick, ej=(5, 12, 15), ec=(0.2, 0.5, 13), flux=(0.0, 0.3, 10), ng=(0.02, 0.45, 15)),
            formula='Γφ = 2π A_ng sqrt(2|ln(ω_low t_exp)|) π |ε1 - ε0| |sin 2π ng|,  '
                    'ε_m = (-1)^m EC 2^(4m+5)/m! sqrt(2/π) (EJ/2EC)^(m/2+3/4) exp(-sqrt(8 EJ/EC))'),
    }


def _scqubits_rate(channel, pt, i):
    t = make_transmon(pt['ej'][i], pt['ec'][i], ng=pt['ng'][i], ncut=BASE['ncut'],
                      n_levels=BASE['n_levels'], flux=pt['flux'][i], d=pt['d'][i])
    return getattr(t, channel)(get_rate=True)


def fit_noise_channel(name, spec):
    """Collect scqubits rates over the channel's sweeps and fit the formula's constants."""
    data = {}  # sweep variable -> (pt arrays, rates)
    for var, vals in spec['sweeps'].items():
        pt = {k: np.full(len(vals), v) for k, v in spec['op'].items()}
        pt[var] = np.asarray(vals, dtype=float)
        rates = np.array([_scqubits_rate(name, pt, i) for i in range(len(vals))])
        data[var] = (pt, rates)

    all_pt = {k: np.concatenate([d[0][k] for d in data.values()]) for k in spec['op']}
    all_rate = np.concatenate([d[1] for d in data.values()])
    ok = np.isfinite(all_rate) & (all_rate > 1e-30)  # drop exact zeros (e.g. sweet spots)
    names = list(spec['params'])
    p0 = np.array([spec['params'][k] for k in names], dtype=float)

    def model(pt, p):
        return spec['theory'](pt, *p)

    sel = lambda pt, m: {k: v[m] for k, v in pt.items()}
    resid = lambda th: np.log10(model(sel(all_pt, ok), p0 * np.exp(th))) - np.log10(all_rate[ok])
    # start away from the defaults (x0.5) so the fit has to find them
    sol = least_squares(resid, np.full(len(p0), np.log(0.5)))
    p_fit = p0 * np.exp(sol.x)
    err_default = np.log10(model(sel(all_pt, ok), p0)) - np.log10(all_rate[ok])
    err_fit = np.log10(model(sel(all_pt, ok), p_fit)) - np.log10(all_rate[ok])
    ss_tot = np.sum((np.log10(all_rate[ok]) - np.log10(all_rate[ok]).mean()) ** 2)
    return dict(name=name, names=names, default=p0, fit=p_fit, data=data, model=model,
                rms_default=np.sqrt(np.mean(err_default ** 2)), rms_fit=np.sqrt(np.mean(err_fit ** 2)),
                max_fit=np.max(np.abs(err_fit)), r2=1 - np.sum(err_fit ** 2) / ss_tot,
                n_points=int(ok.sum()), n_dropped=int((~ok).sum()), all_rate=all_rate[ok],
                all_model_fit=model(sel(all_pt, ok), p_fit))


def fig_noise_eval(run_dir, quick):
    specs = noise_channel_specs(quick)
    fits, lines, rows = [], [], []
    lines += ['Noise channel evaluation: software vs closed-form theory',
              'rms = rms of log10(theory/software): 0.02 ~ 5 %, 0.1 ~ 26 %;  fit started at 0.5 x default', '']
    for name, spec in specs.items():
        print(f'  fitting {name}...')
        try:
            r = fit_noise_channel(name, spec)
        except Exception as e:
            print(f'    {name} failed: {type(e).__name__}: {e}')
            lines.append(f'{name}: FAILED ({type(e).__name__}: {e})')
            continue
        fits.append(r)
        n_sw = len(spec['sweeps'])
        n_rows = (n_sw + 1) // 2
        label = 'T1' if name.startswith('t1') else 'Tφ'
        fig = plt.figure(figsize=(13, 5.2 * n_rows))
        gs = fig.add_gridspec(2 * n_rows, 2, height_ratios=[3, 1] * n_rows, hspace=0.45)
        for n_i, (var, (pt, rates)) in enumerate(r['data'].items()):
            row, col = divmod(n_i, 2)
            ax = fig.add_subplot(gs[2 * row, col])
            ax_res = fig.add_subplot(gs[2 * row + 1, col], sharex=ax)
            x = pt[var]
            t_scq = 1 / np.where(rates > 0, rates, np.nan) / 1e3
            t_fit = 1 / r['model'](pt, r['fit']) / 1e3
            ax.plot(x, t_scq, 'ko', ms=5, label='software evaluation')
            ax.plot(x, 1 / r['model'](pt, r['default']) / 1e3, 'b--', label='theory, default constants')
            ax.plot(x, t_fit, 'r-', label='theory, constants matched to software')
            ax.set(ylabel=f'{label} (µs)', yscale='log', title=f'{label} vs {var}')
            ax.grid(alpha=0.3, which='both')
            ax.legend(fontsize=7)
            ax_res.plot(x, 100 * (t_fit / t_scq - 1), 'r.-')
            ax_res.axhline(0, color='k', lw=0.6)
            ax_res.set(xlabel=var, ylabel='evaluation error (%)')
            ax_res.grid(alpha=0.3)
        fit_txt = ', '.join(f'{k} = {v:.4g} (default {d:.4g})'
                            for k, v, d in zip(r['names'], r['fit'], r['default']))
        fig.suptitle('\n'.join([name, spec['formula'], f'constants matched to software: {fit_txt};  R² = {r["r2"]:.4f}']),
                     fontsize=9, y=0.995)
        save_fig(fig, run_dir, f'08_noise_eval_{name}.png')
        lines += [f'{name}', f'  formula: {spec["formula"]}']
        for k, v, d in zip(r['names'], r['fit'], r['default']):
            lines.append(f'  {k}: implied by software {v:.4g}, documented default {d:.4g}, ratio {v / d:.3f}')
            rows.append([name, k, d, v, v / d, r['rms_default'], r['rms_fit'], r['r2']])
        lines.append(f'  log10 error rms: {r["rms_default"]:.3f} with defaults -> {r["rms_fit"]:.3f} with matched constants '
                     f'(max {r["max_fit"]:.3f}), R² = {r["r2"]:.5f}, {r["n_points"]} points'
                     + (f', {r["n_dropped"]} zero-rate points dropped' if r['n_dropped'] else ''))

    with open(os.path.join(run_dir, 'noise_eval.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channel', 'constant', 'documented_default', 'matched_to_software', 'ratio', 'rms_log10_default',
                    'rms_log10_fit', 'R2'])
        w.writerows([[row[0], row[1]] + [f'{v:.6g}' for v in row[2:]] for row in rows])
    return lines


# ---------------------------------------------------------------------- summary

def write_summary(run_dir, results, ratio, koch_err, sim_lines, elapsed):
    L = [f'Parameter test, {datetime.datetime.now():%Y-%m-%d %H:%M}, {elapsed:.0f} s', '',
         'BASE: ' + ', '.join(f'{k}={v}' for k, v in BASE.items()), '',
         'Theory vs calculated f01 (max |difference| over each sweep, GHz):',
         f'{"sweep":8s} {"bare-Koch":>10s} {"dressed-2nd order":>18s} {"dressed-exact JC":>17s}']
    for name, r in results.items():
        d = lambda a, b: np.nanmax(np.abs(r[a] - r[b]))
        L.append(f'{name:8s} {d("f01", "f01_koch"):10.6f} {d("f01_dressed", "f01_pert2"):18.6f} '
                 f'{d("f01_dressed", "f01_jc_exact"):17.6f}')
    L += ['', f'Bare f01 vs sqrt(8 EJ EC) - EC over EJ/EC = {ratio[0]:.0f}..{ratio[-1]:.0f} (d=0):',
          f'  error {koch_err[0]:.5f} GHz at EJ/EC={ratio[0]:.0f}, '
          f'{koch_err[-1]:.5f} GHz at EJ/EC={ratio[-1]:.0f} (numerical - theory)',
          '', 'Notes:',
          '  - dressed vs exact JC should be ~0: |1,0> only couples to |0,1>, so the single-excitation',
          '    block of the multilevel model is exactly the 2-level JC (differences = state labelling',
          '    near the crossing f01 = f_res).',
          '  - dressed vs 2nd order is the perturbation-theory error, O(g^4 / Delta^3).',
          '  - bare vs Koch: the formula drops the next term in sqrt(EC/EJ); error grows at small EJ/EC.',
          '  - kappa has no effect on f01 (only on Purcell T1); the kappa sweep shows this.',
          '  - at flux = 0 (sweet spot) the flux-noise dephasing is infinite and is drawn as a gap.']
    # if eval_lines:
    #     L += [''] + eval_lines
    if sim_lines:
        L += ['', 'Simulated T1 (master equation) vs expected:'] + ['  ' + s for s in sim_lines]
    path = os.path.join(run_dir, 'summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='coarser grids')
    ap.add_argument('--simulate', action='store_true', help='also run master-equation T1 fits')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Greek letters on a cp1252 console

    t0 = time.time()
    run_dir = make_run_dir()
    print('results ->', run_dir)
    fig_transmon_levels(run_dir, args.quick)
    fig_resonator_levels(run_dir, args.quick)
    fig_spectroscopy(run_dir, args.quick)
    results = run_sweeps(run_dir, args.quick)
    fig_map(run_dir, args.quick)
    ratio, koch_err = fig_theory_vs_calc(run_dir, results, args.quick)
    sim_lines = fig_simulated_t1(run_dir, args.quick) if args.simulate else []
    write_summary(run_dir, results, ratio, koch_err, sim_lines, time.time() - t0)


if __name__ == '__main__':
    main()
