"""
Rotation by a Hamiltonian.

Python translation of the MATLAB script "Rotation by a Hamiltonian" from
the Appendix of "Introduction to Quantum Technologies", Lecture 1
(Dr. Ofer Kfir, 2026).

A fixed two-level Hamiltonian H = [[E1, e], [conj(e), E2]] is used to
propagate an initial state Psi via the time-evolution operator
U(t) = exp(-i H t / hbar). The projections of U(t)|Psi> onto four
reference vectors (the computational basis rotated by 45 degrees, and the
computational basis itself) are plotted as a function of time.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def main():
    hbar = 1.0

    V1 = np.array([1, 1], dtype=complex)
    V2 = np.array([1, -1], dtype=complex)
    V3 = np.array([1, 0], dtype=complex)
    V4 = np.array([0, 1], dtype=complex)
    V1 = V1 / np.linalg.norm(V1)
    V2 = V2 / np.linalg.norm(V2)
    V3 = V3 / np.linalg.norm(V3)
    V4 = V4 / np.linalg.norm(V4)

    Psi = V1

    E1 = -0.5
    E2 = 0.5
    e = 0.01

    H = np.array([
        [E1,         e],
        [np.conj(e), E2],
    ], dtype=complex)

    t_vals = np.arange(0.1, 50, 0.2)

    proj_V1 = np.empty_like(t_vals)
    proj_V2 = np.empty_like(t_vals)
    proj_V3 = np.empty_like(t_vals)
    proj_V4 = np.empty_like(t_vals)

    for i, t_final in enumerate(t_vals):
        U = expm(-1j / hbar * H * t_final)
        state = U @ Psi
        proj_V1[i] = np.abs(np.vdot(V1, state)) ** 2
        proj_V2[i] = np.abs(np.vdot(V2, state)) ** 2
        proj_V3[i] = np.abs(np.vdot(V3, state)) ** 2
        proj_V4[i] = np.abs(np.vdot(V4, state)) ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_vals, proj_V1, 'b.', label='V1')
    ax.plot(t_vals, proj_V2, 'rx', label='V2')
    ax.plot(t_vals, proj_V3, 'k.', label='V3')
    ax.plot(t_vals, proj_V4, 'g.', label='V4')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Projection $\langle V_{1,2}|U(t)|\psi\rangle$')
    ax.set_title(r'$|\psi(0)\rangle = \cdots$' + '\n' + r'$\hat{H}=\cdots$')
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
