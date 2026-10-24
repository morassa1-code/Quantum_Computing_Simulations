import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


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
                 sample_rate=200.0, drive_frequency=None, drag_coeff=0.0):
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
        drag_coeff : float
            DRAG coefficient (lambda). 0 (default) disables DRAG. When nonzero,
            a second quadrature proportional to -(drag_coeff/alpha)*dOmega/dt is
            added 90 degrees out of phase with the main drive, cancelling (to
            first order) the AC-Stark phase error from off-resonantly driving
            the |1>-|2> transition. 1.0 is the standard first-order value;
            requires the qubit's anharmonicity (alpha) at simulation time.
        """
        self.angle = angle
        self.phase = phase
        self.sigma = sigma
        self.n_sigma = n_sigma
        self.sample_rate = sample_rate
        self.drive_frequency = drive_frequency  # resonant with the qubit if None
        self.drag_coeff = drag_coeff
        self.total_modulated_signal = None  # set by generate_drive_signal(): Omega(t)*cos(w_d t + phase)
        self.deriviate_modulated_signal = None  # envelope derivative used for DRAG
  
    @property
    def envelope_duration(self):
        return 2 * self.n_sigma * self.sigma

    @property
    def envelope_amplitude(self):
        return self.angle / (self.sigma * np.sqrt(2 * np.pi))

    def envelope(self, t, t0=0.0, shape='gaussian'):
        if shape == 'gaussian':
            return self.envelope_amplitude * np.exp(-(t - t0) ** 2 / (2 * self.sigma ** 2))
        elif shape == 'square':
            return self.envelope_amplitude * np.where(
                np.abs(t - t0) <= self.n_sigma * self.sigma, 1.0, 0.0)
        else:
            raise ValueError("Unsupported envelope shape")

    def time_array(self, t0=0.0):
        dt = 1.0 / self.sample_rate
        return np.arange(t0 - self.n_sigma * self.sigma,
                          t0 + self.n_sigma * self.sigma + dt, dt)

    def generate_drive_signal(self, omega_q, t0=0.0, alpha=None):
        """
        Lab-frame signal Omega(t)*cos(w_d t + phase) to add to (a + a_dag).
        If drag_coeff != 0, the DRAG quadrature is added too, which needs the
        qubit anharmonicity alpha.
        """
        if self.drag_coeff != 0.0 and alpha is None:
            raise ValueError("DRAG needs the qubit anharmonicity: pass alpha")
        t = self.time_array(t0)
        w_d = self.drive_frequency if self.drive_frequency is not None else omega_q
        env = self.envelope(t, t0)
        theta = w_d * t + self.phase
        signal = env * np.cos(theta)
        if self.drag_coeff != 0.0:
            # DRAG: quadrature Omega_y = -(lambda/alpha) * dOmega/dt, 90 deg out of phase (deriviate)
            dt = 1.0 / self.sample_rate
            self.deriviate_modulated_signal = np.gradient(env, dt)
            omega_y = -(self.drag_coeff / alpha) * self.deriviate_modulated_signal
            signal = signal + omega_y * np.sin(theta)  # lab-frame sign for H = Omega(t)(a + a_dag)
        self.total_modulated_signal = signal
        return t, self.total_modulated_signal

    def plot(self, omega_q, t0=0.0, alpha=None):
        t, signal = self.generate_drive_signal(omega_q, t0, alpha)
        plt.plot(t, self.envelope(t, t0), 'k--', label='envelope')
        plt.plot(t, signal, label='drive')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Amplitude')
        plt.title(f'{self.__class__.__name__} (angle={self.angle:.3f})')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_bloch_trajectory(self, qubit, psi0=None, t0=0.0, frame='lab'):
        """
        Simulate this pulse acting on `qubit` and plot the state's path on
        the Bloch sphere, projected onto the |0>,|1> subspace (any
        population leaked to |2>, |3>, ... just shrinks the vector's
        length, since the projection is renormalized at every step).

        frame : 'lab' or 'rotating'
            'lab' plots the raw simulated state: since apply_gate does no
            RWA, this includes the free precession around Z at omega_q
            (from H0 alone, nothing to do with the pulse), which winds the
            trajectory around the sphere many times during a pulse much
            longer than 1/omega_q.
            'rotating' removes that free precession by multiplying the |1>
            amplitude by exp(i*w_d*t) (w_d = this pulse's drive frequency)
            before computing the Bloch vector, leaving just the rotation
            the drive implements - the same demodulation trick used to
            read out Rabi oscillations experimentally. When w_d ==
            qubit.omega_q (resonant drive, the default), this is the frame
            in which H0 is exactly cancelled, labeled the Heisenberg frame.
        """
        t_vals, psi_traj = qubit.apply_gate(self, psi0=psi0, t0=t0)

        w_d = self.drive_frequency if self.drive_frequency is not None else qubit.omega_q
        if frame not in ('lab', 'rotating'):
            raise ValueError("frame must be 'lab' or 'rotating'")

        # Pauli operators on the |0>,|1> subspace of the full n_levels space
        n = qubit.n_levels
        sx = qt.projection(n, 0, 1) + qt.projection(n, 1, 0)
        sy = -1j * qt.projection(n, 0, 1) + 1j * qt.projection(n, 1, 0)
        sz = qt.projection(n, 0, 0) - qt.projection(n, 1, 1)
        p01 = qt.projection(n, 0, 0) + qt.projection(n, 1, 1)

        states = [qt.Qobj(row.reshape(-1, 1)) for row in psi_traj]
        # dividing by <P01> renormalizes the projection onto the qubit subspace
        # (population leaked to |2>, ... is discarded)
        pop = np.real(qt.expect(p01, states))
        bloch_x = np.real(qt.expect(sx, states)) / pop
        bloch_y = np.real(qt.expect(sy, states)) / pop
        bloch_z = np.real(qt.expect(sz, states)) / pop

        if frame == 'rotating':
            # undo the free precession: multiply the |1> amplitude by exp(i w_d t),
            # i.e. rotate (x, y) about Z by w_d t
            phi = w_d * t_vals
            bloch_x, bloch_y = (bloch_x * np.cos(phi) - bloch_y * np.sin(phi),
                                bloch_x * np.sin(phi) + bloch_y * np.cos(phi))

        if frame == 'lab':
            frame_label = 'lab frame'
        elif w_d == qubit.omega_q:
            frame_label = 'Heisenberg frame'
        else:
            frame_label = 'rotating frame (drive)'

        fig = plt.figure(figsize=(6, 6))
        b = qt.Bloch(fig=fig)
        b.xlabel = [r'$|+\rangle$', r'$|-\rangle$']
        b.ylabel = [r'$|{+i}\rangle$', r'$|{-i}\rangle$']
        b.zlabel = [r'$|0\rangle$', r'$|1\rangle$']
        b.point_size = [4, 60, 60]  # trajectory, initial, final
        b.add_points([bloch_x, bloch_y, bloch_z], meth='l', colors=['crimson'])
        b.add_points([[bloch_x[0]], [bloch_y[0]], [bloch_z[0]]], colors=['g'])
        b.add_points([[bloch_x[-1]], [bloch_y[-1]], [bloch_z[-1]]], colors=['r'])
        b.render()
        fig.suptitle(f'{self.__class__.__name__} (angle={self.angle:.3f}) on Bloch sphere '
                     f'[{frame_label}]\n(green: initial, red: final)')
        plt.show()

        return bloch_x, bloch_y, bloch_z

    def frequency_spectrum(self, omega_q=None, alpha=None, plot=True, n_fft=2**17):
        """
        Compute the Fourier transform of the drive signal, centered on the drive frequency (or the qubit frequency if drive_frequency is None).
        Returns the angular frequencies and the corresponding Fourier transform of the signal.
        omega_q : float, optional
            The qubit frequency. If None, it is set to the drive frequency.
        alpha : float, optional
            The anharmonicity of the qubit. If None, it is set to the qubit's alpha.
        plot : bool, optional
            If True, plots the magnitude of the Fourier transform around the drive frequency.
        n_fft : int, optional
            The number of points to use in the FFT. Default is 2**15 for zero padding.
        """
        if self.total_modulated_signal is None:
            self.generate_drive_signal(omega_q, alpha=alpha)
        
        signal_ft = np.fft.fftshift(np.fft.fft(self.total_modulated_signal, n=n_fft))
        freqs = 2 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(n_fft, d=1.0 / self.sample_rate))
        if plot:
            w_d = self.drive_frequency if self.drive_frequency is not None else omega_q
            half_width = 10.0 / self.sigma  # ~10 spectral widths either side
            plt.plot(freqs, np.abs(signal_ft))
            plt.xlim(-(abs(w_d) + half_width), abs(w_d) + half_width)  # both peaks, +/-w_d
            plt.axvline(omega_q if omega_q is not None else w_d, color='gray', linestyle='--',
                        label=r'$\omega_q$')
            if omega_q is not None and alpha is not None:
                plt.axvline(omega_q + alpha, color='red', linestyle='--',
                            label=r'$\omega_q + \alpha$ (1$\leftrightarrow$2)')
            plt.legend()
            plt.xlabel('Angular frequency (arb. units)')
            plt.ylabel('Magnitude')
            plt.title(f'{self.__class__.__name__} frequency spectrum')
            plt.grid()
            plt.show()
        return freqs, signal_ft


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
        return qt.Qobj(np.diag(self.energies))

    @property
    def a(self):
        return qt.destroy(self.n_levels)

    @property
    def a_dag(self):
        return self.a.dag()

    @property
    def ground_state(self):
        return qt.basis(self.n_levels, 0)

    def apply_gate(self, pulse, psi0=None, t0=0.0):
        """
        Propagate a state through a drive pulse with qutip.sesolve on the
        full lab-frame Hamiltonian H0 + Omega(t)*cos(w_d t+phase)*(a+a_dag)
        - no RWA, no 2-level truncation, so leakage out of the qubit
        subspace (into |2>, etc.) shows up naturally. The drive is passed
        to qutip as an array sampled on the pulse's own time grid (set by
        its sample_rate).

        psi0 may be a qutip ket or a plain vector (default: ground state).

        Returns the time grid and the full complex state at every step
        (shape (len(t_vals), n_levels)), so callers can get populations
        (abs(psi_traj)**2), the final state (psi_traj[-1]), or the relative
        phase between levels (needed for a Bloch-sphere trajectory).
        """
        if psi0 is None:
            psi0 = self.ground_state
        elif not isinstance(psi0, qt.Qobj):
            psi0 = qt.Qobj(np.asarray(psi0, dtype=complex).reshape(-1, 1))

        pulse.generate_drive_signal(self.omega_q, t0, self.alpha)
        t_vals = pulse.time_array(t0)

        H = [self.H0, [self.a + self.a_dag, pulse.total_modulated_signal]]
        result = qt.sesolve(H, psi0, t_vals,
                            options={'store_states': True, 'nsteps': 100000})

        psi_traj = np.array([state.full().ravel() for state in result.states])
        return t_vals, psi_traj

