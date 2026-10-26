from simulation_tools import *

if __name__ == "__main__":
    q = TransmonQubit(omega_q=2.0, alpha=-0.3, n_levels=3)
    # x_gate = XGate(np.pi, drive_frequency=q.omega_q, drag_coeff=0.5, sample_rate=1e3)
    # t_vals, psi_traj = q.apply_gate(x_gate)

    rabi = RabiDrive(omega_rabi=0.1, duration=200, drive_frequency=q.omega_q)
    t_vals, psi_traj = q.apply_gate(rabi)
    fit = q.fit_rabi(t_vals, psi_traj, drive=rabi, save_as='rabi_fit')
    q.plot_populations(t_vals, psi_traj, fit=fit, save_as='rabi')


    # x_gate.plot(q.omega_q, alpha=q.alpha, save_as='x_gate_pulse')
    # x_gate.plot_bloch_trajectory(q, frame='lab', save_as='bloch_trajectory', save_3d='bloch_trajectory.glb', save_gif='bloch_trajectory.gif')
    # x_gate.frequency_spectrum(omega_q=q.omega_q, alpha=q.alpha, n_fft=2**20, save_as='frequency_spectrum')
    # y_gate = YGate(np.pi/2, drive_frequency=q.omega_q, drag_coeff=0.0)
    # y_gate.plot(q.omega_q, alpha=q.alpha, save_as='y_gate_pulse')
    # y_gate.plot_bloch_trajectory(q, frame='rotating', save_as='bloch_trajectory', save_3d='bloch_trajectory.glb', save_gif='bloch_trajectory.gif')
    # y_gate.frequency_spectrum(omega_q=q.omega_q, alpha=q.alpha, n_fft=2**20, save_as='frequency_spectrum')
    
    # z_gate = ZGate(np.pi)
    # z_gate.plot_bloch_trajectory(psi0=(qt.basis(2, 0) + qt.basis(2, 1)) / np.sqrt(2), save_as='z_gate', save_3d='z_gate', save_gif='z_gate')

