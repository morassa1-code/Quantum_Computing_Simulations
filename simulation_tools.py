import os
from matplotlib import lines
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.optimize import curve_fit
import trimesh
from trimesh.visual.material import PBRMaterial

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def save_and_show(save_as=None, dpi=200):
    """Save the current figure to DATA_DIR/save_as (.png if no extension), then show it."""
    if save_as:
        os.makedirs(DATA_DIR, exist_ok=True)
        name = save_as if os.path.splitext(save_as)[1] else save_as + '.png'
        plt.savefig(os.path.join(DATA_DIR, name), dpi=dpi, bbox_inches='tight')
    plt.show()


def save_text(text, name):
    """Save text to DATA_DIR/name (.txt if no extension) and return the path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    name = name if os.path.splitext(name)[1] else name + '.txt'
    path = os.path.join(DATA_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def export_bloch_glb(x, y, z, name, max_points=400, tube_radius=0.015):
    """
    Save a Bloch-sphere path as a 3D model (.glb) in DATA_DIR: a translucent
    unit sphere, x/y/z axes, the trajectory as a red tube, a green start
    marker and a red end marker. In PowerPoint use Insert > 3D Models, then
    drag to rotate it during the slide show. A .glb has no text, so label
    the axes (|0>, |1>, |+>, ...) with text boxes on the slide.
    """

    os.makedirs(DATA_DIR, exist_ok=True)
    name = name if os.path.splitext(name)[1] else name + '.glb'

    # glTF is Y-up: map Bloch (x, y, z) -> (x, z, -y) so that |0> points up
    pts = np.column_stack([x, z, -np.asarray(y)])
    pts = pts[np.linspace(0, len(pts) - 1, min(max_points, len(pts))).astype(int)]

    def solid(mesh, rgba):
        mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
            baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.8,
            alphaMode='BLEND' if rgba[3] < 1.0 else 'OPAQUE', doubleSided=True))
        return mesh

    scene = trimesh.Scene()
    scene.add_geometry(solid(trimesh.creation.icosphere(subdivisions=4, radius=1.0),
                             [0.8, 0.8, 0.8, 0.25]), geom_name='sphere')
    # axes in glTF coordinates (Y up): Bloch X, Bloch Y (-Z in glTF), Bloch Z (+Y in glTF)
    axes = (('x', [1, 0, 0], [0.12, 0.35, 0.85, 1.0]),
            ('y', [0, 0, -1], [0.10, 0.60, 0.25, 1.0]),
            ('z', [0, 1, 0], [0.25, 0.25, 0.25, 1.0]))
    for label, direction, rgba in axes:
        direction = np.array(direction, dtype=float)
        shaft = trimesh.creation.cylinder(radius=0.02,
                                          segment=[-1.25 * direction, 1.25 * direction])
        scene.add_geometry(solid(shaft, rgba), geom_name=f'axis_{label}')
        tip = trimesh.creation.cone(radius=0.06, height=0.18,
                                    transform=trimesh.geometry.align_vectors([0, 0, 1], direction))
        tip.apply_translation(1.25 * direction)  # cone base at the axis end, pointing outward
        scene.add_geometry(solid(tip, rgba), geom_name=f'arrow_{label}')
        for sign in (1, -1):  # markers on the sphere at +-1, showing the poles
            pole = trimesh.creation.icosphere(subdivisions=2, radius=0.04)
            pole.apply_translation(sign * direction)
            scene.add_geometry(solid(pole, rgba), geom_name=f'pole_{label}{"+" if sign > 0 else "-"}')
    tube = trimesh.util.concatenate([
        trimesh.creation.cylinder(radius=tube_radius, segment=[p0, p1], sections=8)
        for p0, p1 in zip(pts[:-1], pts[1:]) if np.linalg.norm(p1 - p0) > 1e-9])
    scene.add_geometry(solid(tube, [0.86, 0.08, 0.24, 1.0]), geom_name='trajectory')
    for label, point, rgba in (('start', pts[0], [0.0, 0.6, 0.0, 1.0]),
                               ('end', pts[-1], [0.9, 0.0, 0.0, 1.0])):
        marker = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
        marker.apply_translation(point)
        scene.add_geometry(solid(marker, rgba), geom_name=label)

    path = os.path.join(DATA_DIR, name)
    scene.export(path)
    return path


def save_bloch_gif(fig, bloch, name, n_frames=72, elev=25, fps=24):
    """Save a GIF of the rendered qutip Bloch sphere turning once about Z, in DATA_DIR."""
    from PIL import Image

    os.makedirs(DATA_DIR, exist_ok=True)
    name = name if os.path.splitext(name)[1] else name + '.gif'
    frames = []
    for azim in np.linspace(0, 360, n_frames, endpoint=False):
        bloch.axes.view_init(elev=elev, azim=azim)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB'))
    path = os.path.join(DATA_DIR, name)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return path


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
        self.complex_envelope = None  # set by generate_drive_signal(): Omega_x - i*Omega_y, used by the RWA
        self.drive_phase = None  # set by generate_drive_signal(): w_d*t + phase
  
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

    def generate_drive_signal(self, omega_q, t0=0.0, alpha=None, use_envelope=True, rwa=False):
        """
        Lab-frame signal Omega(t)*cos(w_d t + phase) to add to (a + a_dag).
        If drag_coeff != 0, the DRAG quadrature is added too, which needs the
        qubit anharmonicity alpha.
        """
        if self.drag_coeff != 0.0 and alpha is None:
            raise ValueError("DRAG needs the qubit anharmonicity: pass alpha")
        t = self.time_array(t0)
        w_d = self.drive_frequency if self.drive_frequency is not None else omega_q
        theta = w_d * t + self.phase
        if use_envelope:
            env = self.envelope(t, t0)
            signal = env * np.cos(theta)
        else:
            env = np.ones_like(t)
            signal = np.cos(theta)
        omega_y = np.zeros_like(t)
        if self.drag_coeff != 0.0:
            # DRAG: quadrature Omega_y = -(lambda/alpha) * dOmega/dt, 90 deg out of phase (deriviate)
            dt = 1.0 / self.sample_rate
            self.deriviate_modulated_signal = np.gradient(env, dt)
            omega_y = -(self.drag_coeff / alpha) * self.deriviate_modulated_signal
            signal = signal + omega_y * np.sin(theta)  # lab-frame sign for H = Omega(t)(a + a_dag)
        self.total_modulated_signal = signal

        # signal = Re[(Omega_x - i*Omega_y) * exp(i*theta)]
        self.complex_envelope = env - 1j * omega_y
        self.drive_phase = theta
        return t, self.total_modulated_signal

    def plot(self, omega_q, t0=0.0, alpha=None, save_as=None):
        t, signal = self.generate_drive_signal(omega_q, t0, alpha)
        plt.plot(t, self.envelope(t, t0), 'k--', label='envelope')
        plt.plot(t, signal, label='drive')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Amplitude')
        plt.title(f'{self.__class__.__name__} (angle={self.angle:.3f})')
        plt.legend()
        plt.grid()
        save_and_show(save_as)

    def plot_bloch_trajectory(self, qubit, psi0=None, t0=0.0, frame='lab', save_as=None,
                              save_3d=None, save_gif=None):
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

        save_as : str, optional
            File name to save the figure under in the data folder.
        save_3d : str, optional
            File name (.glb) for an interactive 3D model of the sphere and
            trajectory, for PowerPoint's Insert > 3D Models.
        save_gif : str, optional
            File name (.gif) for an animation of the sphere turning about Z.
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
        if save_3d:
            export_bloch_glb(bloch_x, bloch_y, bloch_z, save_3d)
        if save_gif:
            save_bloch_gif(fig, b, save_gif)
            b.axes.view_init(elev=30, azim=-60)  # qutip's default view for the still image
        save_and_show(save_as)

        return bloch_x, bloch_y, bloch_z

    def frequency_spectrum(self, omega_q=None, alpha=None, plot=True, n_fft=2**17,
                           save_as=None):
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
        save_as : str, optional
            File name to save the plot under in the data folder (needs plot=True).
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
            save_and_show(save_as)
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

    def plot_bloch_trajectory(self, psi0=None, n_points=200, save_as=None,
                              save_3d=None, save_gif=None):
        """
        Plot the virtual Z gate on the Bloch sphere. There is no pulse and
        no duration: the gate is a frame change by `angle` about Z (the
        phase added to later X/Y pulses in apply_to), so the trajectory is
        the rotation exp(-i*angle*sigma_z/2), obtained by evolving psi0
        under H = sigma_z/2 for a "time" equal to the angle.

        psi0 : qutip ket, optional
            Initial qubit state. Default is |+>, since the ground state
            sits on the Z axis and would not move.
        save_as, save_3d, save_gif : str, optional
            File names (in the data folder) for the still image, the 3D
            model (.glb) and the rotating GIF, as in DrivePulse.
        """
        if psi0 is None:
            psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit() # |+> state
        tlist = np.linspace(0.0, self.angle, n_points)
        result = qt.sesolve(qt.sigmaz() / 2, psi0, tlist,
                            e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        bloch_x, bloch_y, bloch_z = (np.real(e) for e in result.expect)

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
        fig.suptitle(f'ZGate (angle={self.angle:.3f}) on Bloch sphere [virtual Z]\n'
                     '(green: initial, red: final)')
        if save_3d:
            export_bloch_glb(bloch_x, bloch_y, bloch_z, save_3d)
        if save_gif:
            save_bloch_gif(fig, b, save_gif)
            b.axes.view_init(elev=30, azim=-60)  # qutip's default view for the still image
        save_and_show(save_as)

        return bloch_x, bloch_y, bloch_z



    



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

    def apply_gate(self, pulse, psi0=None, t0=0.0, rwa=False):
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

        if rwa:
            # keep only the co-rotating terms: (1/2) * (c* e^{-i theta} a_dag + c e^{i theta} a)
            c, theta = pulse.complex_envelope, pulse.drive_phase
            H = [self.H0,
                 [self.a_dag, 0.5 * np.conj(c) * np.exp(-1j * theta)],
                 [self.a,     0.5 * c * np.exp(1j * theta)]]
        else:
            H = [self.H0, [self.a + self.a_dag, pulse.total_modulated_signal]]
        result = qt.sesolve(H, psi0, t_vals,
                            options={'store_states': True, 'nsteps': 100000})
        psi_traj = np.array([state.full().ravel() for state in result.states])
        return t_vals, psi_traj


    def populations(self, psi_traj):
        """Population of each level at every time step (shape (n_steps, n_levels))."""
        if not isinstance(psi_traj, np.ndarray):
            raise TypeError("psi_traj must be a numpy array")
        return np.abs(psi_traj)**2


    def fit_rabi(self, t_vals, psi_traj, drive=None, level=1, n_fft=2**15, save_as=None):
        """
        Fit the population of `level` to the two-level Rabi formula

            P(t) = (A/2) * (1 - cos(w * (t - t_vals[0])))

        where A is the maximum population and w the oscillation frequency
        (generalized Rabi frequency sqrt(Omega^2 + Delta^2)). If `drive` is a
        RabiDrive, the fit is compared with the theory A = Omega^2/(Omega^2 +
        Delta^2), w = sqrt(Omega^2 + Delta^2). Returns a dict (max_pop,
        omega_eff, fit, and the theory values). The comparison is saved to
        DATA_DIR/save_as (.txt if no extension), or printed if save_as is None.
        """
        tau = np.asarray(t_vals) - t_vals[0]
        pop = self.populations(psi_traj)[:, level]
        dt = tau[1] - tau[0]

        # initial guess: dominant frequency of the population (FFT peak)
        spectrum = np.abs(np.fft.rfft(pop - pop.mean(), n=n_fft))
        w0 = 2 * np.pi * np.fft.rfftfreq(n_fft, dt)[np.argmax(spectrum)]

        def model(t, A, w, B):
            return 0.5 * A * (1 - np.cos(w * t)) + B

        (A, w, B), _ = curve_fit(model, tau, pop, p0=[pop.max(), w0, pop.mean()])
        result = {'max_pop': pop.max(), 'max_pop_fit': A, 'omega_eff': abs(w), 
                  'offset_fit': B, 'fit': model(tau, A, w, B)}

        report = [f'omega_q = {self.omega_q}, alpha = {self.alpha}, n_levels = {self.n_levels}',
                  f'|{level}> population: max = {pop.max():.4f}, fit amplitude = {A:.4f}, '
                  f'fit omega = {abs(w):.5f} (fit offset = {B:.4f})']
        if drive is not None:
            max_pop_theory, omega_th = drive.expected_rabi(self.omega_q)
            result.update(max_pop_theory=max_pop_theory, omega_eff_theory=omega_th)
            report.insert(1, f'omega_rabi = {drive.omega_rabi}, '
                             f'detuning = {drive.detuning(self.omega_q)}, duration = {drive.duration}')
            report.append(f'theory:        max = {max_pop_theory:.4f}, omega = {omega_th:.5f}  '
                          f'(errors: {abs(A - max_pop_theory) / max_pop_theory:.1%}, '
                          f'{abs(abs(w) - omega_th) / omega_th:.1%})')
        report = '\n'.join(report) + '\n'
        if save_as:
            save_text(report, save_as)
        else:
            print(report)
        return result

    def plot_populations(self, t_vals, psi_traj, fit=None, save_as=None):
        """Plot the probability of each level |0>, |1>, ... against time (fit: dict from fit_rabi)."""
        populations = self.populations(psi_traj)
        for n in range(populations.shape[1]):
            plt.plot(t_vals, populations[:, n], label=f'|{n}>')
        if fit is not None:
            plt.plot(t_vals, fit['fit'], 'k--', label='Rabi fit')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Population')
        plt.title('Rabi oscillations')
        plt.legend()
        plt.grid()
        save_and_show(save_as)


class RabiDrive(DrivePulse):
    """
    A continuous drive with constant amplitude Omega(t) = omega_rabi for
    `duration` (a square envelope), to watch Rabi oscillations.

    omega_rabi : Rabi frequency (the constant envelope amplitude).
    duration : length of the drive.
    drive_frequency : carrier frequency; None means resonant with the qubit.
    """

    def __init__(self, omega_rabi, duration=400.0, phase=0.0, sample_rate=200.0,
                 drive_frequency=None):
        super().__init__(angle=omega_rabi * duration, phase=phase,
                         sigma=duration / 2, n_sigma=1, sample_rate=sample_rate,
                         drive_frequency=drive_frequency)
        self.omega_rabi = omega_rabi
        self.duration = duration

    def envelope(self, t, t0=0.0):
        return self.omega_rabi * np.where(np.abs(t - t0) <= self.duration / 2, 1.0, 0.0)


    def detuning(self, omega_q):
        """Return the detuning of the drive carrier from the qubit's 0<->1 transition."""
        if self.drive_frequency is None:
            return 0.0
        return self.drive_frequency - omega_q

    def expected_rabi(self, omega_q):
        """Two-level RWA theory: (max population, oscillation frequency)."""
        delta = self.detuning(omega_q)
        w = np.hypot(self.omega_rabi, delta) # Rabi frequency sqrt(Omega^2 + Delta^2)
        return self.omega_rabi**2 / w**2, w
