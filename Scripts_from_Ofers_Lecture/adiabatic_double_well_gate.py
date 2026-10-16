"""
Adiabatic single qubit gate for a double-well (delta-well) pair.

Python translation of the MATLAB script "Adiabatic single qubit gate for a
delta-well pair" from the Appendix of "Introduction to Quantum
Technologies", Lecture 1 (Dr. Ofer Kfir, 2026).

Two coupled wells (e.g. waveguides / quantum wells) have a time-dependent
separation d(t) that is brought adiabatically from a large separation
(df, weak coupling) down to a small separation (dn, strong coupling) and
back out again, following a smooth tanh profile. The inter-well tunneling
coupling r(t) = exp(-k * d(t)) drives Rabi-like population transfer
between the two wells. Depending on (dn, df, k) this adiabatic sweep can
realize different single-qubit gates (e.g. an X gate, or a 50/50
beamsplitter-like gate) -- see the commented-out parameter sets below.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def main():
    draw_final_only = True

    t = np.arange(-10, 10 + 0.1, 0.1)

    # parameters for X gate:
    # k = 0.9999; dn = 1.58; df = 8 / k
    # parameters for 50/50 gate:
    # k = 0.9999; dn = 2.3136; df = 8 / k
    k = 0.9999
    dn = 0.5
    df = 8 / k

    d = df - (df - dn) * 0.5 * (1 + np.tanh(t + 5)) * 0.5 * (1 - np.tanh(t - 5))
    r = np.exp(-k * d)

    dt = t[1] - t[0]

    A = np.full_like(t, np.nan, dtype=complex)  # amplitude alpha
    B = np.full_like(t, np.nan, dtype=complex)  # amplitude beta
    A0 = 0.0
    B0 = 1.0
    A[0] = A0
    B[0] = B0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Top panel: well separation and coupling strength vs. time
    ax1.plot(t, d, 'b', label='Well separation, d')
    ax1.plot(t, -d, 'b')
    ax1.plot(t, 10 * r, 'r', label=r'coupling r (or $\Omega_R$)')
    ax1.plot(t, np.zeros_like(t), 'k')
    ax1.legend()

    # Bottom panel: live (or final) population transfer
    ax2.plot(t, np.zeros_like(t), 'k')
    ax2.plot(t, np.ones_like(t), 'k')
    if not draw_final_only:
        ax2.plot(t[0], np.abs(A0) ** 2, 'ro')
        ax2.plot(t[0], np.abs(B0) ** 2, 'bo')
    ax2.set_xlabel('time (arb.u)')
    ax2.set_ylabel(r'$|\alpha(t)|^2$ and $|\beta(t)|^2$')

    for kt in range(len(t) - 1):
        Hkt = np.array([
            [0,               r[kt]],
            [np.conj(r[kt]),  0],
        ], dtype=complex)

        v_before = np.array([A[kt], B[kt]])
        v_after = expm(-1j * Hkt * dt) @ v_before
        A[kt + 1] = v_after[0]
        B[kt + 1] = v_after[1]

        if not draw_final_only:
            ax2.plot(t[kt + 1], np.abs(A[kt + 1]) ** 2, 'r.')
            ax2.plot(t[kt + 1], np.abs(B[kt + 1]) ** 2, 'b.')

    ax2.plot(t, np.abs(A) ** 2, 'r', label=r'$|\alpha(t)|^2$')
    ax2.plot(t, np.abs(B) ** 2, 'b', label=r'$|\beta(t)|^2$')
    ax2.legend()

    fig.tight_layout()
    plt.show()

    print(np.abs(B[-1]) ** 2)


if __name__ == "__main__":
    main()
