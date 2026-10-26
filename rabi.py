import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# 1. Define physical parameters (in units where hbar = 1)
omega_rabi = 1.0 * 2 * np.pi  # Rabi frequency (driving strength)
detuning = 0.0 * 2 * np.pi    # Detuning from resonance (Δ = omega_drive - omega_qubit)

# 2. Set up operators and Hamiltonian
# Standard two-level system basis: |0> is ground, |1> is excited
ket_0 = qt.basis(2, 0)
ket_1 = qt.basis(2, 1)

# Hamiltonian under the Rotating Wave Approximation (RWA)
H = 0.5 * detuning * qt.sigmaz() + 0.5 * omega_rabi * qt.sigmax()

# 3. Define initial state and time grid
psi0 = ket_0  # System starts completely in the ground state
tlist = np.linspace(0, 3.0, 300)  # Simulate for 3 time units

# 4. Ideal evolution (Closed System via sesolve)
# We track the population projection operators P0 and P1 as observables
P0 = ket_0 * ket_0.dag()
P1 = ket_1 * ket_1.dag()

result_closed = qt.sesolve(H, psi0, tlist, e_ops=[P0, P1])

# 5. Dissipative evolution (Open System via mesolve)
# Define environmental collapse operators (c_ops)
gamma_relaxation = 0.3  # Rate of decay from |1> to |0>
gamma_dephasing = 0.1   # Pure dephasing rate

c_ops = [
    np.sqrt(gamma_relaxation) * qt.destroy(2),  # Energy relaxation (|1> -> |0>)
    np.sqrt(2 * gamma_dephasing) * qt.sigmaz()  # Phase decoherence
]

result_open = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[P0, P1])

# 6. Plotting the Comparison
plt.figure(figsize=(10, 5))
plt.plot(tlist, result_closed.expect[1], 'b-', label='Ideal (Closed - State $|1\\rangle$)')
plt.plot(tlist, result_open.expect[1], 'r--', label='With Decay (Open - State $|1\\rangle$)')
plt.plot(tlist, result_open.expect[0], 'g:', label='With Decay (Open - State $|0\\rangle$)')

plt.title('Rabi Oscillations: Closed vs. Open Quantum System Dynamics')
plt.xlabel('Time (t)')
plt.ylabel('Population Probability')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
