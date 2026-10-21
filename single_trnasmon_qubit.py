from simulation_tools import *

if __name__ == "__main__":
    q = TransmonQubit(omega_q=2.0, alpha=-0.3, n_levels=3)
    x_gate = XGate(np.pi, drive_frequency=q.omega_q, drag_coeff=0.5, sample_rate=1e3)
    x_gate.DRAG_scheme(q.omega_q, q.alpha)
    x_gate.plot_bloch_trajectory(q, frame='rotating')
    x_gate.frequency_spectrum(omega_q=q.omega_q, alpha=q.alpha, n_fft=2**20)
    # YGate(np.pi/2).plot_bloch_trajectory(q)
    # ZGate(np.pi/2).plot_bloch_trajectory(q)

