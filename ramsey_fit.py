"""
Fit Ramsey measurements (CSV files) to                (Prep Report 4, Experiment 2, Q3-Q4)

    P(t) = B + A exp(-(t / T2*)^n) cos(2 pi df t + phi)

    A    fringe amplitude (contrast: pulse quality, readout fidelity)
    T2*  dephasing time (envelope decay)
    df   fringe frequency = |drive - qubit| detuning
    phi  phase (pulse phase errors, a delay offset in t)
    B    baseline (the mixed-state value, readout offset)
    n    1: exponential envelope (Markovian dephasing, the Lindblad model; default)
         2: Gaussian envelope (quasi-static / 1/f frequency noise)

For every file: the least-squares fit with 1-sigma uncertainties, SSR = sum (y - f)^2,
RMSE = sqrt(SSR / N), a figure (data + fit + envelope, residuals, FFT) and diagnostics
that say what probably went wrong in the measurement and what to change. A table of
every file (ramsey_fits.csv) and a text summary (ramsey_fits.txt) go next to the figures.

    python ramsey_fit.py path/to/folder                 # every .csv in the folder
    python ramsey_fit.py a.csv b.csv --envelope best    # fit both envelopes, keep the lower RMSE
    python ramsey_fit.py folder --t-col delay --y-col 2 # choose columns by name or index
    python ramsey_fit.py --demo                         # make synthetic good / bad files, fit them

The time column is the one whose name looks like t / time / tau / delay / wait (else
the first numeric column); the signal is the next numeric column, or, when there are
columns named I and Q, their projection on the main axis of the IQ cloud. Results are
in the CSV's own units (a time in us gives T2* in us and df in MHz).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')  # figures are saved, not shown

import argparse
import csv
import glob
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert

from simulation_tools import get_output_dir

# Folder of Ramsey CSVs fitted when no path is given on the command line
# (e.g. when running this file with the editor's Run button). None = no default.
DEFAULT_CSV_FOLDER = r"C:\uni\Quantum_Computing_Lab\csv_4_simulation"  # e.g. r'C:\Users\me\Documents\lab\ramsey_csvs'

TWO_PI = 2 * np.pi
ENVELOPES = {'exp': 1.0, 'gauss': 2.0}
_TIME_NAME = re.compile(r'^t($|[\s_(\[])|time|delay|tau|wait|idle', re.I)


# ---------------------------------------------------------------------- loading

def _to_float(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


def _column(spec, names):
    """Column index from an index ('2') or a name (case-insensitive)."""
    if str(spec).lstrip('-').isdigit():
        return int(spec)
    low = [n.strip().lower() for n in names]
    if str(spec).strip().lower() not in low:
        raise ValueError(f'no column {spec!r}; columns are {names}')
    return low.index(str(spec).strip().lower())


def iq_projection(i, q):
    """Project IQ points on the main axis of their cloud (the axis the qubit state moves along)."""
    z = np.column_stack([i, q])
    z = z - np.nanmean(z, axis=0)
    _, _, vt = np.linalg.svd(z[np.isfinite(z).all(axis=1)], full_matrices=False)
    return z @ vt[0]


def load_csv(path, t_col=None, y_col=None):
    """Return (t, y, info) sorted by t; info has the column names used."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        text = f.read()
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=',;\t')
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(text.splitlines(), dialect)
            if any(c.strip() for c in r) and not r[0].lstrip().startswith('#')]
    if not rows:
        raise ValueError('empty file')
    first = [c.strip() for c in rows[0]]
    header = first if any(c and np.isnan(_to_float(c)) for c in first) else None
    if header:
        rows = rows[1:]
    width = max(len(r) for r in rows)
    data = np.full((len(rows), width), np.nan)
    for k, r in enumerate(rows):
        data[k, :len(r)] = [_to_float(c) for c in r]
    names = (header or []) + [f'col{i}' for i in range(len(header or []), width)]
    numeric = [i for i in range(width) if np.isfinite(data[:, i]).mean() > 0.5]
    if len(numeric) < 2:
        raise ValueError(f'need a time and a signal column, found numeric columns {[names[i] for i in numeric]}')

    ti = _column(t_col, names) if t_col is not None else \
        next((i for i in numeric if _TIME_NAME.search(names[i])), numeric[0])
    others = [i for i in numeric if i != ti]
    if y_col is not None:
        yi = _column(y_col, names)
        y, y_name = data[:, yi], names[yi]
    else:
        low = {names[i].strip().lower(): i for i in others}
        if 'i' in low and 'q' in low:
            y, y_name = iq_projection(data[:, low['i']], data[:, low['q']]), 'IQ projection'
        else:
            y, y_name = data[:, others[0]], names[others[0]]
    t = data[:, ti]
    ok = np.isfinite(t) & np.isfinite(y)
    order = np.argsort(t[ok], kind='stable')
    return t[ok][order], y[ok][order], {'t_name': names[ti], 'y_name': y_name, 'columns': names}


# ---------------------------------------------------------------------- fitting

def ramsey_model(t, A, T2, f, phi, B, n=1.0):
    return B + A * np.exp(-(np.abs(t) / T2) ** n) * np.cos(TWO_PI * f * t + phi)


def _amp_phase(t, y, f, T2, n):
    """A, phi for fixed f, T2 by linear least squares: y = A e cos(wt + phi) = c e cos - s e sin."""
    e = np.exp(-(np.abs(t) / T2) ** n)
    basis = np.column_stack([e * np.cos(TWO_PI * f * t), -e * np.sin(TWO_PI * f * t)])
    c, s = np.linalg.lstsq(basis, y, rcond=None)[0]
    return np.hypot(c, s), np.arctan2(s, c)


def initial_guess(t, y):
    """
    Initial guesses, all from the data:
      B   the mean of the signal
      df  the highest peak of the FFT (zero-padded 16x, on a uniform grid), above
          the lowest frequency the window can resolve (1/span), so the decay itself
          (power near 0) is not taken for the fringes
      T2* the time the analytic-signal (Hilbert) envelope falls to 1/e of its start
      A, phi  linear least squares with df and T2* fixed (in fit_ramsey)
    """
    tu, inv = np.unique(t, return_inverse=True)  # average repeated delays
    yu = np.bincount(inv, weights=y) / np.bincount(inv)
    span = tu[-1] - tu[0]
    dt = np.median(np.diff(tu))
    B = yu.mean()
    tg = np.arange(tu[0], tu[-1] + dt / 2, dt)
    yg = np.interp(tg, tu, yu) - B
    n_fft = 1 << int(np.ceil(np.log2(16 * len(tg))))
    spec = np.abs(np.fft.rfft(yg, n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    search = spec * (freqs >= 1.0 / span)
    f0 = freqs[np.argmax(search)] if search.any() else 0.0

    env = np.abs(hilbert(yg))
    k = max(1, int(round(1.0 / (f0 * dt)))) if f0 > 0 else max(1, len(tg) // 20)
    env = np.convolve(env, np.ones(k) / k, mode='same')
    start = env[:max(k, 2)].max()
    below = np.nonzero(env < start / np.e)[0]
    T2 = tg[below[0]] - tg[0] if below.size else span
    return dict(B=B, f=f0, T2=max(T2, 2 * dt), dt=dt, span=span, fft=(freqs, spec))


def fit_ramsey(t, y, envelope='exp'):
    """
    Least-squares fit (scipy curve_fit, trust-region reflective with bounds:
    A >= 0, 0 <= df <= Nyquist, T2* > 0). Started from several T2* guesses
    (the envelope estimate, and span/10, span/3, span), keeping the lowest SSR,
    so a poor first guess does not trap the fit in a local minimum.
    envelope: 'exp', 'gauss' or 'best' (both, the lower RMSE wins).
    Returns a dict (params A, T2, f, phi, B; errors; ssr, rmse, r2; ...).
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    if envelope == 'best':
        fits = [fit_ramsey(t, y, e) for e in ENVELOPES]
        ok = [f for f in fits if f['ok']] or fits
        return min(ok, key=lambda f: f['rmse'])
    n = ENVELOPES[envelope]
    g = initial_guess(t, y)
    nyq = 0.5 / g['dt']
    model = lambda x, A, T2, f, phi, B: ramsey_model(x, A, T2, f, phi, B, n)
    lo = [0, g['dt'] / 4, 0, -np.inf, -np.inf]
    hi = [np.inf, 100 * g['span'], nyq, np.inf, np.inf]

    best = None
    for T2_0 in (g['T2'], g['span'] / 10, g['span'] / 3, g['span']):
        T2_0 = float(np.clip(T2_0, lo[1] * 1.01, hi[1] * 0.99))
        A0, phi0 = _amp_phase(t, y - g['B'], g['f'], T2_0, n)
        p0 = [max(A0, 1e-12), T2_0, min(g['f'], nyq), phi0, g['B']]
        try:
            popt, pcov = curve_fit(model, t, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        ssr = float(np.sum((y - model(t, *popt)) ** 2))
        if best is None or ssr < best[2]:
            best = (popt, pcov, ssr, p0)

    out = dict(envelope=envelope, n=n, N=len(t), dt=g['dt'], span=g['span'], nyquist=nyq,
               fft=g['fft'], guess=g, t=t, y=y)
    if best is None:
        return {**out, 'ok': False, 'params': np.full(5, np.nan), 'errors': np.full(5, np.nan),
                'ssr': np.nan, 'rmse': np.nan, 'r2': np.nan, 'residuals': np.full(len(t), np.nan),
                'at_bound': []}
    popt, pcov, ssr, p0 = best
    popt[3] = (popt[3] + np.pi) % TWO_PI - np.pi  # phi in (-pi, pi]
    err = np.sqrt(np.abs(np.diag(pcov))) if np.all(np.isfinite(pcov)) else np.full(5, np.inf)
    resid = y - model(t, *popt)
    names = ['A', 'T2', 'f', 'phi', 'B']
    at_bound = [names[i] for i in (1, 2)
                if np.isclose(popt[i], lo[i], rtol=1e-3) or np.isclose(popt[i], hi[i], rtol=1e-3)]
    return {**out, 'ok': True, 'params': popt, 'errors': err, 'p0': p0, 'ssr': ssr,
            'rmse': float(np.sqrt(ssr / len(t))), 'r2': 1 - ssr / np.sum((y - y.mean()) ** 2),
            'residuals': resid, 'at_bound': at_bound}


def diagnose(r):
    """What probably went wrong in the measurement, and what to change. Empty = looks good."""
    if not r['ok']:
        return ['fit did not converge: check the file (columns, units) and whether there is a Ramsey signal at all']
    A, T2, f, _, _ = r['params']
    dA, dT2, df, _, _ = r['errors']
    dt, span, nyq = r['dt'], r['span'], r['nyquist']
    out = []
    snr = A / r['rmse'] if r['rmse'] > 0 else np.inf
    if snr < 5:
        out.append(f'low signal-to-noise (A / RMSE = {snr:.1f}): average more shots per point, '
                   'improve the readout (frequency, power, integration time), and check the pi/2 '
                   'amplitude - a small A can also mean wrong pulses')
    if f > 0 and 1 / (f * dt) < 4:
        out.append(f'fringes under-sampled ({1 / (f * dt):.1f} points per period, Nyquist = {nyq:.3g}): '
                   'the true detuning may be above Nyquist and aliased to the fitted one - use a '
                   'smaller time step or a smaller detuning, and check by changing the detuning')
    elif f > 0 and 1 / (f * dt) < 8:
        out.append(f'only {1 / (f * dt):.1f} points per fringe: one file cannot rule out aliasing (a true '
                   f'detuning of k / dt +- {f:.3g} looks the same) - repeat with a different detuning or '
                   'time step and check that the fitted df follows')
    if f * T2 < 1:
        out.append(f'fewer than one fringe within T2* (df T2* = {f * T2:.2f}): the drive is too close '
                   'to resonance, so the oscillation and the decay cannot be told apart - detune the '
                   'drive so that ~3-10 fringes fit within T2*')
    if span < 1.5 * T2:
        out.append(f'time window ({span:.3g}) shorter than 1.5 T2*: the decay is barely seen and T2* '
                   'is an extrapolation - extend the maximum delay to ~3-5 T2*')
    elif span > 10 * T2:
        out.append(f'time window = {span / T2:.0f} T2*: most points are on the baseline after the '
                   'signal has decayed - shorten the window to ~3-5 T2* and spend the points there')
    n_in = int(np.sum(r['t'] - r['t'][0] < T2))
    if n_in < 15:
        out.append(f'only {n_in} points within T2*: use a finer time step')
    if np.isfinite(dT2) and dT2 > 0.2 * T2:
        out.append(f'T2* poorly constrained (+-{100 * dT2 / T2:.0f} %)')
    if not np.isfinite(dT2):
        out.append('covariance undefined: the parameters are not independent with these data')
    if r['at_bound']:
        out.append(f'fit stopped at a bound of {", ".join(r["at_bound"])} (no visible decay or frequency)')
    res = r['residuals']
    rho1 = np.corrcoef(res[:-1], res[1:])[0, 1] if len(res) > 3 and res.std() > 0 else 0.0
    if rho1 > 0.5:
        out.append(f'residuals are correlated (lag-1 r = {rho1:.2f}): the model misses structure - '
                   'a beating (two fringe frequencies: charge-parity switching, a TLS, a drifting '
                   'flux bias), frequency drift during the scan, or the envelope shape '
                   '(try --envelope gauss or best)')
    return out


# ---------------------------------------------------------------------- output

PARAM_LABELS = ['A', 'T2*', 'df', 'phi', 'B']


def plot_fit(name, r, info, diagnostics, out_dir):
    t, y = r['t'], r['y']
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[2.2, 1], hspace=0.08,
                          bottom=0.12 + 0.025 * len(diagnostics))
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_f = fig.add_subplot(gs[:, 1])
    ax.plot(t, y, 'o', ms=3, color='gray', label='data')
    if r['ok']:
        A, T2, f, phi, B = r['params']
        tt = np.linspace(t.min(), t.max(), 3000)
        env = A * np.exp(-(np.abs(tt) / T2) ** r['n'])
        ax.plot(tt, ramsey_model(tt, *r['params'], n=r['n']), 'r-', lw=1, label=f'fit ({r["envelope"]} envelope)')
        ax.plot(tt, B + env, 'b--', lw=0.8, label='envelope')
        ax.plot(tt, B - env, 'b--', lw=0.8)
        txt = '\n'.join(f'{k} = {v:.4g} ± {e:.2g}' for k, v, e in zip(PARAM_LABELS, r['params'], r['errors']))
        txt += f'\nN = {r["N"]}\nSSR = {r["ssr"]:.4g}\nRMSE = {r["rmse"]:.4g}\nR² = {r["r2"]:.4f}'
        ax.text(1.0, 1.0, txt, transform=ax.transAxes, ha='right', va='top', fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
        ax_r.plot(t, r['residuals'], '.', ms=3, color='k')
    ax_r.axhline(0, color='r', lw=0.8)
    ax.set(ylabel=info['y_name'], title=name)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax_r.set(xlabel=info['t_name'], ylabel='residual')
    ax_r.grid(alpha=0.3)

    freqs, spec = r['fft']
    ax_f.plot(freqs, spec / (spec.max() or 1), 'k-', lw=0.8)
    ax_f.axvline(r['guess']['f'], color='orange', ls='--', label=f'FFT peak (guess) = {r["guess"]["f"]:.4g}')
    if r['ok']:
        ax_f.axvline(r['params'][2], color='r', lw=0.8, label=f'fitted df = {r["params"][2]:.4g}')
    ax_f.set(xlabel=f'frequency (1 / {info["t_name"]})', ylabel='normalized |FFT|', title='spectrum',
             xlim=(0, r['nyquist']))
    ax_f.legend(fontsize=8)
    ax_f.grid(alpha=0.3)
    fig.text(0.01, 0.01, '\n'.join(['diagnostics:'] + [f'  - {d}' for d in diagnostics or ['none: the fit looks good']]),
             fontsize=8, va='bottom', wrap=True)
    path = os.path.join(out_dir, f'{name}_fit.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def write_summary(results, out_dir):
    with open(os.path.join(out_dir, 'ramsey_fits.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'envelope', 'A', 'dA', 'T2', 'dT2', 'df', 'd_df', 'phi', 'dphi', 'B', 'dB',
                    'N', 'SSR', 'RMSE', 'R2', 'diagnostics'])
        for name, r, diag in results:
            pe = [v for pair in zip(r['params'], r['errors']) for v in pair]
            w.writerow([name, r['envelope'], *[f'{v:.6g}' for v in pe], r['N'], f'{r["ssr"]:.6g}',
                        f'{r["rmse"]:.6g}', f'{r["r2"]:.6g}', ' | '.join(diag)])
    lines = [f'{"file":28s} {"T2*":>18s} {"df":>18s} {"A":>8s} {"N":>5s} {"SSR":>10s} {"RMSE":>9s}  diagnostics']
    for name, r, diag in results:
        A, T2, f = r['params'][:3]
        dT2, df = r['errors'][1:3]
        lines.append(f'{name[:28]:28s} {T2:9.4g} ± {dT2:<6.2g} {f:9.4g} ± {df:<6.2g} {A:8.3g} {r["N"]:5d} '
                     f'{r["ssr"]:10.4g} {r["rmse"]:9.4g}  {len(diag)} issue(s)')
        lines += [f'{"":30s}- {d}' for d in diag]
    text = '\n'.join(lines)
    with open(os.path.join(out_dir, 'ramsey_fits.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    return text


# ---------------------------------------------------------------------- demo data

# synthetic measurements: T1 = 50 us, T_phi = 80 us, quasi-static detuning noise of
# 4 kHz, readout contrast 0.9 (P_meas = 0.05 + 0.9 P_e), binomial shot noise
# drive_detuning = Delta (MHz), the detuning you set on the drive; the frequency
# noise delta ~ N(0, DEMO_SIGMA) is added shot by shot on top of it
DEMO_SIGMA = 0.004
DEMO_CASES = [
    ('good', dict(drive_detuning=0.2, t_max=150, n=301, shots=2000)),
    ('undersampled', dict(drive_detuning=0.85, t_max=150, n=151, shots=2000)),   # 1 us step: Nyquist 0.5 MHz
    ('window_too_short', dict(drive_detuning=0.2, t_max=15, n=151, shots=2000)),
    ('window_too_long', dict(drive_detuning=0.05, t_max=1500, n=301, shots=2000)),
    ('few_shots', dict(drive_detuning=0.2, t_max=150, n=301, shots=10)),
    ('on_resonance', dict(drive_detuning=0.0, t_max=150, n=301, shots=2000)),
]


def make_demo_files(folder, seed=0):
    from relaxation_times import quasi_static_average, ramsey_signal
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for name, c in DEMO_CASES:
        t = np.linspace(0, c['t_max'], c['n'])
        p = quasi_static_average(ramsey_signal, t, c['drive_detuning'], DEMO_SIGMA, t1=50.0, t_phi=80.0)
        y = rng.binomial(c['shots'], np.clip(0.05 + 0.9 * p, 0, 1)) / c['shots']
        path = os.path.join(folder, f'ramsey_{name}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_us', 'P_e'])
            w.writerows(zip(np.round(t, 6), y))
        paths.append(path)
    return paths


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*', help='CSV files and/or folders of CSV files')
    ap.add_argument('--envelope', choices=['exp', 'gauss', 'best'], default='exp')
    ap.add_argument('--t-col', help='time column (name or index)')
    ap.add_argument('--y-col', help='signal column (name or index)')
    ap.add_argument('--out', help='output folder (default data/<date_time>)')
    ap.add_argument('--demo', action='store_true', help='make synthetic good and bad files and fit them')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    out_dir = args.out or get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    files = make_demo_files(os.path.join(out_dir, 'demo_data')) if args.demo else []
    paths = args.paths
    if not paths and not args.demo and DEFAULT_CSV_FOLDER:
        if not os.path.isdir(DEFAULT_CSV_FOLDER):
            ap.error(f'DEFAULT_CSV_FOLDER does not exist: {DEFAULT_CSV_FOLDER}')
        print('fitting DEFAULT_CSV_FOLDER =', DEFAULT_CSV_FOLDER)
        paths = [DEFAULT_CSV_FOLDER]
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, '*.csv'))) if os.path.isdir(p) else [p]
    if not files:
        ap.error('give CSV files or a folder, set DEFAULT_CSV_FOLDER at the top of ramsey_fit.py, '
                 'or use --demo')

    results = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            t, y, info = load_csv(path, args.t_col, args.y_col)
            r = fit_ramsey(t, y, args.envelope)
        except Exception as e:
            print(f'{name}: skipped ({type(e).__name__}: {e})')
            continue
        diag = diagnose(r)
        plot_fit(name, r, info, diag, out_dir)
        results.append((name, r, diag))
    print(write_summary(results, out_dir))
    print('\nfigures and tables in', out_dir)


if __name__ == '__main__':
    main()
