from simulation_tools import *

if __name__ == "__main__":
    q = TransmonQubit(omega_q=2.0, alpha=-0.3, n_levels=3)
    # x_gate = XGate(np.pi, drive_frequency=q.omega_q, drag_coeff=0.5, sample_rate=1e3)
    # x_gate.plot_bloch_trajectory(q, frame='rotating')
    # x_gate.frequency_spectrum(omega_q=q.omega_q, alpha=q.alpha, n_fft=2**20)
    y_gate = YGate(np.pi/2, drive_frequency=q.omega_q, drag_coeff=0.0)
    y_gate.plot(q.omega_q, alpha=q.alpha, save_as='y_gate_pulse')
    y_gate.plot_bloch_trajectory(q, frame='rotating', save_as='bloch_trajectory', save_3d='bloch_trajectory.glb', save_gif='bloch_trajectory.gif')
    y_gate.frequency_spectrum(omega_q=q.omega_q, alpha=q.alpha, n_fft=2**20, save_as='frequency_spectrum')
    # ZGate(np.pi/2).plot_bloch_trajectory(q)

