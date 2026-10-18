from simulation_tools import *

if __name__ == "__main__":
    q = TransmonQubit(omega_q=5.0, alpha=-0.3, n_levels=5)
    XGate(np.pi, drive_frequency=q.omega_q).plot_bloch_trajectory(q, frame='rotating')
    # YGate(np.pi/2).plot_bloch_trajectory(q)
    # ZGate(np.pi/2).plot_bloch_trajectory(q)

