"""
Displays some of my favorite attractors

Description of the Lorenz equations:
These are highly simplified equations that describe aspects
of fluid motion in a shallow layer.

From wiki:
More specifically, the Lorenz equations are derived from the Oberbeck-Boussinesq
approximation to the equations describing fluid circulation in a shallow layer
of fluid, heated uniformly from below and cooled uniformly from above. This fluid
circulation is known as Rayleigh-Bénard convection. The fluid is assumed to
circulate in two dimensions (vertical and horizontal) with periodic rectangular
boundary conditions.
The Lorenz equations also arise in simplified models for lasers, dynamos,
thermosyphons, brushless DC motors, electric circuits, chemical reactions and
forward osmosis.

Lorenz attractor equations:
dx/dt = σ(y−x)dx/dt     = σ(y−x)
dy/dt = x(ρ−z) − ydy/dt = x(ρ−z)−y
dz/dt                   = xy − βz

x - proportional to the intensity of convection motion.
y - proportional to the temperature difference between the ascending and
    descending currents.
z - proportional to the distortion of the vertical temperature profile
    from linearity.

"""

from uplog import log
import sys
import numpy as np
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, sigma=10, rho=28, beta=2.667):
    """
    Calculates the dt in the lorenz system
    """
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = (x * y) - (beta * z)
    return x_dot, y_dot, z_dot


def rossler(x, y, z, a=0.2, b=0.2, c=5.7):
    """
    Calculates the dt in the Rossler system
    """
    x_dot = (-1.0 * y) - z
    y_dot = x + (a * y)
    z_dot = b + (z * (x - c))
    return x_dot, y_dot, z_dot


def chua(x, y, z, a=15.6, b=32.0, c=0.01):
    """
    Calculates the dt in Chua's circuit system
    """
    x_dot = a * (y - chua_func(x))
    y_dot = x - y + z
    z_dot = ((-1.0 * b) * y) - (c * z)
    return x_dot, y_dot, z_dot


def chua_func(x, m0=-1.14285714285714, m1=-0.71428571428571):
    if x <= -1:
        g = (m1 * x) + m1 - m0
    elif (x > -1) and (x < 1):
        g = m0 * x
    else:  # x >= 1
        g = (m1 * x) + m0 - m1
    return x + g


def init_random(x_mean=0.0, y_mean=0.0, z_mean=0.0,
                x_variance=20.0, y_variance=20.0, z_variance=20.0):
    """
    Used to randomly assign initial values to the attractor
    """
    # Set to random values
    x_init = x_mean + random.uniform(-x_variance, x_variance)
    y_init = y_mean + random.uniform(-y_variance, y_variance)
    z_init = z_mean + random.uniform(-z_variance, z_variance)
    return [x_init, y_init, z_init]


def plot_attractor(steps=15000, dt=0.001, attractor='lorenz', 
                   random=True, color='b'):
    """
    Plots a the chosen attractor
    :param steps: timesteps to integrate
    :param dt: delta t
    :param attractor: type of attractor
    :return:
    """
    # Need one more for the initial values
    xo_vals = np.empty(steps)
    yo_vals = np.empty(steps)
    zo_vals = np.empty(steps)
    xo_dot = 0
    yo_dot = 0
    zo_dot = 0
    if random:
        
        if attractor == 'lorenz':
            xo_vals[0], yo_vals[0], zo_vals[0] = init_random(x_variance=20.0, 
                                                             y_variance=20.0, 
                                                             z_variance=20.0)
        elif attractor == 'rossler':
            xo_vals[0], yo_vals[0], zo_vals[0] = init_random(x_variance=10.0, 
                                                             y_variance=10.0, 
                                                             z_variance=10.0)
        elif attractor == 'chua':
            xo_vals[0], yo_vals[0], zo_vals[0] = init_random(x_variance=2.0, 
                                                             y_variance=0.5, 
                                                             z_variance=2.0)
        else:
            log.out.error("Attractor: " + attractor + " not defined.")
            sys.exit(-1)
    else:
        if attractor == 'lorenz':
            xo_vals[0] = 1.0
            yo_vals[0] = 1.0
            zo_vals[0] = 1.0
        elif attractor == 'rossler':
            xo_vals[0] = 1.0
            yo_vals[0] = 1.0
            zo_vals[0] = 1.0
        elif attractor == 'chua':
            xo_vals[0] = 0.777
            yo_vals[0] = -0.222
            zo_vals[0] = -1.222
        else:
            log.out.error("Attractor: " + attractor + " not defined.")
            sys.exit(-1)

    # Stepping through time
    for i in range(0, steps-1):
        # Calculate the derivatives of the X, Y, Z state
        if attractor == 'lorenz':
            xo_dot, yo_dot, zo_dot = lorenz(xo_vals[i], yo_vals[i], zo_vals[i])
        elif attractor == 'rossler':
            xo_dot, yo_dot, zo_dot = rossler(xo_vals[i], yo_vals[i], zo_vals[i])
        elif attractor == 'chua':
            xo_dot, yo_dot, zo_dot = chua(xo_vals[i], yo_vals[i], zo_vals[i])

        # Propagate the system
        xo_vals[i + 1] = xo_vals[i] + (xo_dot * dt)
        yo_vals[i + 1] = yo_vals[i] + (yo_dot * dt)
        zo_vals[i + 1] = zo_vals[i] + (zo_dot * dt)

    fig3d = plt.figure()
    fig3d.canvas.set_window_title('3D View (model)')
    ax3d = Axes3D(fig3d)
    ax3d.plot(xo_vals, yo_vals, zo_vals, color=color, linewidth=0.5)
    ax3d.set_xlabel("X Axis")
    ax3d.set_ylabel("Y Axis")
    ax3d.set_zlabel("Z Axis")
    if attractor == 'lorenz':
        ax3d.set_title("Lorenz Attractor")
    elif attractor == 'rossler':
        ax3d.set_title("Rössler Attractor")
    elif attractor == 'chua':
        ax3d.set_title("Chua's Circuit Attractor")

    plt.show()


#########################
#         DRIVER        #
#########################
if __name__ == '__main__':
    log.setLevel('INFO')  # Set the logging level
    # Report basic system info
    log.out.info("Python version: " + sys.version)
    log.out.info("Number of CPUs: " + str(mp.cpu_count()))

    # Choose the demo
    # Add observational error
    plot_attractor(steps=125000, dt=0.002, attractor='lorenz', 
                   random=False, color='b')
    # Try the Rossler attractor
    plot_attractor(steps=125000, dt=0.002, attractor='rossler', 
                   random=False, color='r')
    # Try Chua's circuit
    plot_attractor(steps=125000, dt=0.002, attractor='chua', 
                   random=False, color='g')

    # Stop Logging
    log.stopLog()
