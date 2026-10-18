import numpy as np
import matplotlib.pyplot as plt


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

