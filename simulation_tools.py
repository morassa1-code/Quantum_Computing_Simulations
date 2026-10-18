import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


class DrivePulse:
    """
    Resonant microwave drive pulse implementing a single-qubit rotation on a
    transmon, under the rotating-wave approximation:

        H_eff(t) = (Omega(t)/2) * [cos(phase)*sigma_x + sin(phase)*sigma_y]

    The envelope is Gaussian, truncated to +/- n_sigma*sigma, with peak
    amplitude chosen so that integral(Omega(t) dt) = angle (the rotation
    angle), since integral(exp(-t^2/(2 sigma^2))) dt = sigma*sqrt(2*pi).
    """

    def __init__(self, angle=np.pi, phase=0.0, sigma=10.0, n_sigma=4,
                 sample_rate=200.0, drive_frequency=None):
        """
        Parameters
        ----------
        angle : float
            The rotation angle (in radians) for the pulse - to rotate the qubit state.
        phase : float
            The phase of the pulse (in radians).
        sigma : float
            The standard deviation of the Gaussian envelope.
        n_sigma : float
            The number of standard deviations to truncate the envelope.
        sample_rate : float
            The sampling rate of the pulse (in arbitrary units).
        drive_frequency : float, optional
            The frequency of the drive field. If None, it is set to the qubit frequency.
        """
        self.angle = angle
        self.phase = phase
        self.sigma = sigma
        self.n_sigma = n_sigma
        self.sample_rate = sample_rate
        self.drive_frequency = drive_frequency  # resonant with the qubit if None

    @property
    def envelope_duration(self):
        return 2 * self.n_sigma * self.sigma

    @property
    def envelope_amplitude(self):
        return self.angle / (self.sigma * np.sqrt(2 * np.pi))

    def envelope(self, t, t0=0.0):
        return self.envelope_amplitude * np.exp(-(t - t0) ** 2 / (2 * self.sigma ** 2))

    def time_array(self, t0=0.0):
        dt = 1.0 / self.sample_rate
        return np.arange(t0 - self.n_sigma * self.sigma,
                          t0 + self.n_sigma * self.sigma + dt, dt)

    def drive_signal(self, omega_q, t0=0.0):
        """Lab-frame signal Omega(t)*cos(w_d t + phase) to add to (a + a_dag)."""
        t = self.time_array(t0)
        w_d = self.drive_frequency if self.drive_frequency is not None else omega_q
        return t, self.envelope(t, t0) * np.cos(w_d * t + self.phase)

    def plot(self, omega_q, t0=0.0):
        t, signal = self.drive_signal(omega_q, t0)
        plt.plot(t, self.envelope(t, t0), 'k--', label='envelope')
        plt.plot(t, signal, label='drive')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Amplitude')
        plt.title(f'{self.__class__.__name__} (angle={self.angle:.3f})')
        plt.legend()
        plt.grid()
        plt.show()


class XGate(DrivePulse):
    def __init__(self, angle=np.pi, **kwargs):
        kwargs.pop('phase', None)
        super().__init__(angle=angle, phase=0.0, **kwargs)


class YGate(DrivePulse):
    def __init__(self, angle=np.pi, **kwargs):
        kwargs.pop('phase', None)
        super().__init__(angle=angle, phase=np.pi / 2, **kwargs)


class ZGate:
    """
    Virtual Z gate: on a transmon, a Z rotation isn't driven with microwaves
    but implemented as a frame change, so it has no envelope or duration -
    it just needs to shift the phase of any X/Y pulses applied after it.
    """

    def __init__(self, angle=np.pi / 2):
        self.angle = angle
        self.envelope_duration = 0.0

    def apply_to(self, pulse):
        pulse.phase += self.angle
        return pulse


class TransmonQubit:
    """
    Truncated (n_levels) transmon: a weakly anharmonic Josephson-junction
    oscillator, approximated as a Duffing oscillator in the Fock basis
    |0>, |1>, ..., |n_levels-1>:

        E_n = n*omega_q + (alpha/2)*n*(n-1)

    omega_q is the 0<->1 transition frequency and alpha < 0 is the
    anharmonicity, which separates the 1<->2 transition from the 0<->1
    one and is what makes single-qubit gates imperfect (leakage into |2>).
    """

    def __init__(self, omega_q=5.0, alpha=-0.3, n_levels=4, hbar=1.0):
        self.omega_q = omega_q
        self.alpha = alpha
        self.n_levels = n_levels
        self.hbar = hbar

    @property
    def energies(self):
        n = np.arange(self.n_levels)
        return n * self.omega_q + 0.5 * self.alpha * n * (n - 1)

    @property
    def H0(self):
        return np.diag(self.energies).astype(complex)

    @property
    def a(self):
        return np.diag(np.sqrt(np.arange(1, self.n_levels)), k=1).astype(complex)

    @property
    def a_dag(self):
        return self.a.conj().T

    @property
    def ground_state(self):
        psi0 = np.zeros(self.n_levels, dtype=complex)
        psi0[0] = 1.0
        return psi0

    def apply_gate(self, pulse, psi0=None, t0=0.0):
        """
        Propagate a state through a drive pulse by direct time integration
        of the full lab-frame Hamiltonian H0 + Omega(t)*cos(w_d t+phase)*(a+a_dag)
        - no RWA, no 2-level truncation, so leakage out of the qubit
        subspace (into |2>, etc.) shows up naturally. Step size is set by
        the pulse's own sample_rate.
        """
        if psi0 is None:
            psi0 = self.ground_state

        t_vals = pulse.time_array(t0)
        dt = t_vals[1] - t_vals[0]
        w_d = pulse.drive_frequency if pulse.drive_frequency is not None else self.omega_q
        drive_op = self.a + self.a_dag

        psi = psi0.copy()
        populations = np.empty((len(t_vals), self.n_levels))
        populations[0] = np.abs(psi) ** 2

        for i in range(len(t_vals) - 1):
            t = t_vals[i]
            Om = pulse.envelope(t, t0) * np.cos(w_d * t + pulse.phase)
            H = self.H0 + Om * drive_op
            psi = expm(-1j * H * dt / self.hbar) @ psi
            populations[i + 1] = np.abs(psi) ** 2

        return t_vals, psi, populations

