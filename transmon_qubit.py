"""
Transmon qubit: driven Rabi dynamics and leakage to the 2nd excited state.

A transmon is a weakly anharmonic Josephson-junction oscillator. In the
transmon regime (E_J >> E_C) its low-lying energy levels are well
approximated by a Duffing oscillator in the Fock (number) basis
|0>, |1>, |2>, ...:

    E_n = n * hbar*omega_q + (hbar*alpha/2) * n*(n-1)

where omega_q is the 0<->1 transition frequency and alpha < 0 is the
anharmonicity (typically a few percent of omega_q for real devices).
Because alpha != 0, the 1<->2 transition frequency (omega_q + alpha) is
close to, but not exactly equal to, the 0<->1 transition frequency. A
microwave drive tuned to resonantly flip |0> -> |1> (an X gate / pi-pulse)
therefore also weakly drives |1> -> |2>: this is the canonical transmon
"leakage" problem (the same motivation that leads to DRAG pulses in real
devices), directly analogous to the parasitic |psi2> <-> |psi3> coupling
explored for the particle-in-a-box qubit in
Scripts_from_Ofers_Lecture/particle_infinite_well.py.

This script:
  1) Builds a truncated (N-level) transmon Hamiltonian and drives it with
     a resonant, Gaussian-enveloped charge drive ~ Omega(t)*cos(w_d t)*(a+a^),
     sized to implement an ideal pi-pulse (X gate) under the rotating-wave
     approximation (RWA).
  2) Propagates the full N-level state (no RWA, no truncation to 2 levels)
     alongside an idealized 2-level-only RWA model, to show how much the
     real multi-level dynamics departs from the idealized qubit picture:
     population left behind in |1> and population leaked into |2>.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def main():
    hbar = 1.0

    # --- Transmon (Duffing oscillator) parameters, in angular-frequency units ---
    N = 4                 # number of truncated Fock levels: |0>,|1>,|2>,|3>
    omega_q = 5.0          # 0<->1 transition frequency (matches the nu=5 convention
                            # used elsewhere in these lecture scripts)
    alpha = -0.3           # anharmonicity (negative, ~6% of omega_q -- typical for
                            # real transmons, whose anharmonicity is usually a few
                            # hundred MHz against a several-GHz qubit frequency)
    w_d = omega_q           # drive frequency: resonant with the 0<->1 transition

    # --- Drive pulse: Gaussian envelope sized for an ideal RWA pi-pulse (X gate) ---
    # sigma sets the pulse's spectral width (~1/sigma). It must be slow compared to
    # the anharmonicity (|alpha| ~ 0.3) so the pulse's frequency content stays well
    # clear of the |1>-|2> transition at (omega_q + alpha) -- otherwise the "qubit"
    # gate also significantly drives |1> -> |2>, which is unrealistically lossy.
    sigma = 10.0             # pulse width (time units)
    t0 = 0.0                 # pulse center
    # Under the RWA, cos(w_d t)*(a+a^) reduces (within the qubit subspace) to the
    # effective Hamiltonian H_eff = (Omega(t)/2) * sigma_x, i.e. Omega(t) plays the
    # role of the (time-dependent) Rabi frequency. The resulting population
    # transfer is sin^2(theta(t)/2) with pulse area theta(t) = integral(Omega(t))dt,
    # so a pi-pulse (full transfer, an X gate) requires integral(Omega(t))dt = pi.
    # For a Gaussian, integral(exp(-(t-t0)^2/(2 sigma^2))) dt = sigma*sqrt(2*pi),
    # so:
    Omega_max = np.pi / (sigma * np.sqrt(2 * np.pi))

    def Omega_env(t):
        return Omega_max * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))

    # --- Operators in the truncated N-level Fock basis ---
    a = np.diag(np.sqrt(np.arange(1, N)), k=1).astype(complex)  # lowering operator
    a_dag = a.conj().T                                           # raising operator

    energies = np.array([n * omega_q + 0.5 * alpha * n * (n - 1) for n in range(N)])
    H0 = np.diag(energies).astype(complex)

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)  # for the ideal 2-level model

    # --- Time grid ---
    dt = 0.005
    t_vals = np.arange(-4 * sigma, 4 * sigma + dt, dt)
    n_t = len(t_vals)

    # --- Initial states: ground state |0>, for both the full and ideal models ---
    psi_full = np.zeros(N, dtype=complex)
    psi_full[0] = 1.0

    psi_ideal = np.array([1, 0], dtype=complex)  # 2-level-only (RWA) model

    pop_full = np.empty((n_t, N))
    pop_ideal1 = np.empty(n_t)
    envelope = np.empty(n_t)

    pop_full[0] = np.abs(psi_full) ** 2
    pop_ideal1[0] = np.abs(psi_ideal[1]) ** 2
    envelope[0] = Omega_env(t_vals[0])

    for it in range(n_t - 1):
        t = t_vals[it]
        Om = Omega_env(t)
        envelope[it + 1] = Omega_env(t_vals[it + 1])

        # Full lab-frame, multi-level Hamiltonian (no RWA, no 2-level truncation)
        H_full = H0 + Om * np.cos(w_d * t) * (a + a_dag)
        U_full = expm(-1j * H_full * dt / hbar)
        psi_full = U_full @ psi_full

        # Idealized 2-level RWA model: H_eff = (Omega(t)/2) * sigma_x
        H_ideal = (Om / 2) * sigma_x
        U_ideal = expm(-1j * H_ideal * dt / hbar)
        psi_ideal = U_ideal @ psi_ideal

        pop_full[it + 1] = np.abs(psi_full) ** 2
        pop_ideal1[it + 1] = np.abs(psi_ideal[1]) ** 2

    # --- Report final populations ---
    print("Final populations of the full (multi-level) simulation:")
    for n in range(N):
        print(f"  P{n} = {pop_full[-1, n]:.5f}")
    print(f"Ideal (2-level RWA) final P1 = {pop_ideal1[-1]:.5f}")
    print(f"Leakage into |2> at the end of the pulse: {pop_full[-1, 2]:.5e}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(t_vals, envelope, 'm')
    ax1.set_ylabel(r'Drive envelope $\Omega(t)$')
    ax1.set_title('Transmon driven by a resonant Gaussian pi-pulse')

    labels = [r'$P_0$', r'$P_1$', r'$P_2$', r'$P_3$'][:N]
    colors = ['b', 'r', 'g', 'y'][:N]
    for n in range(N):
        ax2.plot(t_vals, pop_full[:, n], colors[n], label=labels[n])
    ax2.plot(t_vals, pop_ideal1, 'k--', label=r'ideal $P_1$ (2-level RWA)')
    ax2.set_xlabel('Time (arb. units)')
    ax2.set_ylabel('Population')
    ax2.set_ylim(-0.04, 1.04)
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
