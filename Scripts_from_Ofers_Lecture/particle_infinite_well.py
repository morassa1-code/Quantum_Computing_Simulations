"""
Effective qubit within a 3-level system (particle in an infinite well).

Python translation of the MATLAB script "Effective qubit within a 3-level
system" from the Appendix of "Introduction to Quantum Technologies",
Lecture 1 (Dr. Ofer Kfir, 2026).

Physical picture
-----------------
The first two eigenstates of a particle in a 1D infinite square well,
|psi1> and |psi2>, are used as an effective qubit. A harmonic dipole drive
resonant with the |psi1> <-> |psi2> transition (coupling V12) is applied.
Because the infinite well has infinitely many levels, the same drive also
couples |psi2> to the third eigenstate |psi3> with a "parasitic" coupling
V23 -- the well is not a true two-level system.

This script numerically integrates the Schrodinger equation for the full
3-level state (including the parasitic coupling to |psi3>) and compares it
to the ideal 2-level (qubit-only) evolution, to quantify:
  1) how much population leaks into the parasitic level |psi3>, and
  2) the resulting phase error accumulated in the qubit subspace.

See Lecture 1, section "An effective qubit from a non-two-level-system"
and its Appendix for the derivation of V12, V23, and alpha (the energy of
|psi3> relative to the qubit's mean energy, in units of hbar*nu/2).
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    hbar = 1.0

    # Measurement / basis vectors (3-level Hilbert space)
    V1 = np.array([1, 1, 0], dtype=complex)
    V2 = np.array([-1, 1, 0], dtype=complex)
    V3 = np.array([0, 0, 1], dtype=complex)
    V1 = V1 / np.linalg.norm(V1)
    V2 = V2 / np.linalg.norm(V2)
    V3 = V3 / np.linalg.norm(V3)

    # Coupling strengths and frequencies (units of hbar = 1)
    V12 = 0.05        # |psi1> <-> |psi2> dipole coupling (the qubit drive)
    V23 = 0.05        # |psi2> <-> |psi3> parasitic dipole coupling
    w = 5.0            # drive frequency
    nu = 5.0           # qubit transition frequency (on resonance: w = nu)
    alpha = 13 / 3     # (E3 - Ebar) / (hbar*nu/2), from the infinite-well energies

    # Initial state: full 3-level state starts in |psi1> (normalized)
    Psi3_t = V1 

    # Ideal 2-level (qubit-only) state: same initial condition, restricted
    # to the {|psi1>, |psi2>} subspace
    V11 = V1[:2]
    Psi2_t = V11 / np.linalg.norm(V11)

    dt = 0.005
    t_max = 5.0
    n_steps = int(round((t_max - 0.01) / dt)) + 1
    t_vals = np.linspace(0.01, t_max, n_steps)

    pop3 = np.empty_like(t_vals)
    phase_error = np.empty_like(t_vals)

    for i, t in enumerate(t_vals):
        # Population left in the parasitic 3rd level
        pop3[i] = np.abs(np.vdot(V3, Psi3_t)) ** 2

        # Phase error between the full (3-level) evolution and the ideal
        # (2-level only) evolution, restricted to the qubit subspace
        phase_error[i] = np.angle(np.vdot(Psi2_t, Psi3_t[:2])) / np.pi

        # Parasitic coupling term (no rotating-wave approximation applied)
        H23 = V23 * np.cos(w * t) * np.exp(-1j * (alpha - 1) / 2 * nu * t)

        H = np.array([
            [0,            V12, 0],
            [V12,          0,   H23],
            [0, np.conj(H23),   0],
        ], dtype=complex)

        # Explicit Euler step of the Schrodinger equation:
        #   d|psi>/dt = -(i/hbar) H |psi>
        Psi3_t = Psi3_t - dt * 1j * (H @ Psi3_t) / hbar
        Psi2_t = Psi2_t - dt * 1j * (H[:2, :2] @ Psi2_t) / hbar

    # --- Plotting ---
    t_axis = w * t_vals / (2 * np.pi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(t_axis, pop3, 'b.')
    ax1.set_ylabel(r'$|\langle \psi_3 | \psi_3 \rangle|^2$', fontsize=14)
    ax1.set_xlim(0, w * t_max / (2 * np.pi))
    ax1.tick_params(labelsize=14)

    ax2.plot(t_axis, phase_error, 'r.')
    ax2.set_ylabel(r'phase error ($\pi$)', fontsize=14)
    ax2.set_xlim(0, w * t_max / (2 * np.pi))
    ax2.set_xlabel(r'Time ($\omega t / 2\pi$)', fontsize=14)
    ax2.tick_params(labelsize=14)

    fig.suptitle('parasitic phase')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
