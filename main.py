# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def euler_rhs(omega, I, torque):
    I1, I2, I3 = I
    w1, w2, w3 = omega
    t1, t2, t3 = torque
    dw1 = ((I2 - I3) * w2 * w3 + t1) / I1
    dw2 = ((I3 - I1) * w3 * w1 + t2) / I2
    dw3 = ((I1 - I2) * w1 * w2 + t3) / I3
    return np.array([dw1, dw2, dw3])

def integrate_euler(I, omega0, torque_func, t_span, dt):
    t0, tf = t_span
    n = int(np.ceil((tf - t0) / dt)) + 1
    ts = np.linspace(t0, tf, n)
    omegas = np.empty((n, 3))
    omegas[0] = omega0
    for i in range(1, n):
        t = ts[i-1]
        omega = omegas[i-1]
        torque = torque_func(t)
        k1 = euler_rhs(omega, I, torque)
        k2 = euler_rhs(omega + 0.5*dt*k1, I, torque_func(t + 0.5*dt))
        k3 = euler_rhs(omega + 0.5*dt*k2, I, torque_func(t + 0.5*dt))
        k4 = euler_rhs(omega + dt*k3, I, torque_func(t + dt))
        omegas[i] = omega + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return ts, omegas

def kinetic_energy(I, omega):
    return 0.5 * np.sum(I * omega**2, axis=1)

def angular_momentum_magnitude(I, omega):
    L = I * omega
    return np.linalg.norm(L, axis=1)

def main():
    # Parameters
    I = np.array([2.0, 1.0, 3.0])  # principal moments of inertia
    omega0 = np.array([1.0, 2.0, 3.0])  # initial angular velocity (rad/s)
    dt = 0.001
    t_span = (0.0, 20.0)

    # Experiment 1: torque‑free rotation
    torque_zero = lambda t: np.zeros(3)
    t1, w1 = integrate_euler(I, omega0, torque_zero, t_span, dt)
    KE1 = kinetic_energy(I, w1)

    plt.figure()
    plt.plot(t1, w1[:,0], label=r'$\omega_1$')
    plt.plot(t1, w1[:,1], label=r'$\omega_2$')
    plt.plot(t1, w1[:,2], label=r'$\omega_3$')
    plt.xlabel('Time')
    plt.ylabel('Angular velocity (rad/s)')
    plt.title('Angular velocity components (torque‑free)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('angular_velocity_components.png')
    plt.close()

    plt.figure()
    plt.plot(t1, KE1)
    plt.xlabel('Time')
    plt.ylabel('Rotational kinetic energy')
    plt.title('Kinetic energy (torque‑free)')
    plt.tight_layout()
    plt.savefig('kinetic_energy.png')
    plt.close()

    # Experiment 2: constant torque about the second principal axis
    tau0 = 0.5  # magnitude of constant torque
    torque_const = lambda t: np.array([0.0, tau0, 0.0])
    t2, w2 = integrate_euler(I, omega0, torque_const, t_span, dt)
    KE2 = kinetic_energy(I, w2)
    Lmag2 = angular_momentum_magnitude(I, w2)

    # Approximate Euler angle φ as the time integral of ω₂
    phi = np.cumsum(w2[:,1]) * dt

    plt.figure()
    plt.plot(t2, Lmag2)
    plt.xlabel('Time')
    plt.ylabel('||L||')
    plt.title('Angular momentum magnitude (constant torque)')
    plt.tight_layout()
    plt.savefig('angular_momentum_magnitude.png')
    plt.close()

    plt.figure()
    plt.plot(t2, phi)
    plt.xlabel('Time')
    plt.ylabel(r'$\phi$ (rad)')
    plt.title('Euler angle $\phi$ about torque axis')
    plt.tight_layout()
    plt.savefig('euler_angle_phi.png')
    plt.close()

    # Primary numeric answer: initial kinetic energy (conserved in torque‑free case)
    answer = KE1[0]
    print('Answer:', answer)

if __name__ == '__main__':
    main()

