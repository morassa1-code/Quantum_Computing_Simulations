"""
Rabi-Oscillation calculation.

Python translation of the MATLAB script "Rabi-Oscillation calculation"
from the Appendix of "Introduction to Quantum Technologies", Lecture 1
(Dr. Ofer Kfir, 2026).

Plots the analytic Rabi-oscillation probabilities |alpha(t)|^2 and
|beta(t)|^2 for three values of detuning (Delta*omega = 0, 1, 4, in units
of the Rabi frequency Omega_R), illustrating how detuning both reduces the
oscillation amplitude and increases the effective (generalized) Rabi
frequency Omega = sqrt(Omega_R^2 + Delta*omega^2).
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    detuning_vec = [0, 1, 4]

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=False)

    for kk, delta_omega in enumerate(detuning_vec):
        ax = axes[kk]

        Omega_R = 1.0
        Omega = np.sqrt(Omega_R ** 2 + delta_omega ** 2)
        sin_theta = Omega_R / Omega
        cos_theta = delta_omega / Omega  # noqa: F841 (kept for parity with original script)

        t = np.arange(0, 2 + 0.01, 0.01) * 2 * np.pi / Omega_R

        prob_alpha = sin_theta ** 2 * (np.sin(Omega * t / 2)) ** 2
        prob_beta = sin_theta ** 2 * (np.cos(Omega * t / 2)) ** 2
        t_axis = Omega_R * t / (2 * np.pi)

        ax.plot(t_axis, prob_alpha, 'b')
        ax.plot(t_axis, np.zeros_like(t_axis), 'k')
        ax.plot(t_axis, np.ones_like(t_axis), 'k')
        ax.plot(t_axis, prob_beta, 'r')
        ax.plot(t_axis, np.zeros_like(t_axis), 'k')
        ax.plot(t_axis, np.ones_like(t_axis), 'k')

        ax.set_ylim(-0.04, 1.04)
        ax.set_title(rf'$\Delta\omega = {round(delta_omega, 2)}\ \Omega_R$')

        if kk == 2:
            ax.set_xlabel(r'Cycles of $\Omega_R$, $[\frac{2\pi}{\Omega_R}]$')
        if kk == 1:
            ax.set_ylabel(r'$|\alpha(t)|^2$ (blue) and $|\beta(t)|^2$ (red)')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
