"""
Add description dude...
The axes could use some love

"""

from uplog import log
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import math


def newtons_method(f, df, zi, e, max_iteration=1000):
    """
    Find a root using Newtown's method
    :param f: function to search for root on
    :param df: derivative of above
    :param x0: initial guess
    :param e: epsilon to stop iteration at
    :return:
    """
    # Initialize the iteration
    delta = abs(0 - f(zi))
    iteration = 0
    while (delta > e) and (iteration < max_iteration):
        iteration += 1
        zi = zi - f(zi)/df(zi)
        delta = abs(0 - f(zi))
    return zi, iteration


def make_polynomial(coefficients, powers):
    """
    Returns the value of a polynomial "c1*x^p1 + c2*x^p2 +..." at x

    :param coefficients: c1, c2, ...
    :param powers: p1, p2, ...
    :param x: where to evaluate
    :return: value of polynomial at x
    """
    if len(coefficients) != len(powers):
        log.out.error("Length of coefficient array must equal length of power array")
        raise ValueError

    def polynomial(x):
        value = 0
        for i in range(len(coefficients)):
            value += coefficients[i] * x**powers[i]
        return value

    return polynomial


def make_derivative_polynomial(coefficients, powers):
    """
    Returns the derivative of a polynomial "c1*x^p1 + c2*x^p2 +..." at x
    :param coefficients: c1, c2, ...
    :param powers: p1, p2, ...
    :param x: where to evaluate
    :return: derivative of polynomial at x
    """
    if len(coefficients) != len(powers):
        log.out.error("Length of coefficient array must equal length of power array")
        raise ValueError

    def derivative(x):
        value = 0
        for i in range(len(coefficients)):
            if powers[i] != 0:
                value += (coefficients[i] * powers[i]) * x**(powers[i]-1)
        return value

    return derivative


def test_method():
    # Define z^4 -1
    coefs = [1.0, -1.0]
    pows = [4, 0]
    f = make_polynomial(coefs, pows)
    df = make_derivative_polynomial(coefs, pows)
    # Make initial guesses and find roots
    guesses = [0.1+0.0j, -2.2-1.0j, -0.4+1.0j, -1.1-2.0j]
    for guess in guesses:
        root, iterations = newtons_method(f, df, guess, 1e-5)
        log.out.info("Guess | Result | Iterations: " + str(guess) + " | " +
                     str(root) + " | " + str(iterations))


def calc_basins(f, df, x_axis, y_axis):
    # Define test space
    xmat = np.repeat(x_axis[:,np.newaxis], len(y_axis), 1)
    ymat = np.repeat(y_axis[:,np.newaxis], len(x_axis), 1).transpose()
    z = xmat + (ymat * 1j)

    root_map = np.zeros([len(x_axis), len(y_axis)], np.complex64)
    iteration_map = np.zeros([len(x_axis), len(y_axis)])

    # Loop through map and save solution
    for i in tqdm(range(len(x_axis))):
        for j in range(len(y_axis)):
            root, iterations = newtons_method(f, df, z[i, j],
                                              1e-5, max_iteration=1000)
            root_map[i, j] = root
            iteration_map[i, j] = iterations
    return root_map, iteration_map


def plot_basins(xmin, xmax, ymin, ymax, xres, yres,
                f=None, df=None):
    # Define test space
    x_axis = np.arange(xmin, xmax, xres, dtype=np.float32)
    y_axis = np.arange(ymin, ymax, yres, dtype=np.float32)

    # Get the estimated basins
    if (f is None) or (df is None):
        # Define z^4 -1 as a default function
        coefs = [1.0, -1.0]
        pows = [4, 0]
        f = make_polynomial(coefs, pows)
        df = make_derivative_polynomial(coefs, pows)
        
    basin_map, iter_map = calc_basins(f, df, x_axis, y_axis)
    
    # Break it up for visualization
    basin_map_real = np.zeros([len(x_axis), len(y_axis)])
    basin_map_imag = np.zeros([len(x_axis), len(y_axis)])
    for i in range(len(x_axis)):
        for j in range(len(y_axis)):
            basin_map_real[i, j] = basin_map[i, j].real
            basin_map_imag[i, j] = basin_map[i, j].imag
    # Normalize for coloring
    # Shift to positive
    outliers = 1.0
    basin_map_real[basin_map_real > outliers] = outliers
    basin_map_real[basin_map_real < -outliers] = -outliers
    basin_map_imag[basin_map_imag > outliers] = outliers
    basin_map_imag[basin_map_imag < -outliers] = -outliers
    basin_map_real = np.nan_to_num(basin_map_real)
    basin_map_imag = np.nan_to_num(basin_map_imag)

    max_real_val = 1.0
    min_real_val = np.min(basin_map_real)
    basin_map_real = basin_map_real + abs(min_real_val)
    max_imag_val = 1.0
    min_imag_val = np.min(basin_map_imag)
    basin_map_imag = basin_map_imag + abs(min_imag_val)
    basin_map_imag = 2.0 * max_real_val * (basin_map_imag + max_imag_val)
    basin_colors = basin_map_real + basin_map_imag
    
#    pylab.matshow(basin_colors.transpose(), cmap=plt.cm.hot)
    pylab.matshow(basin_colors.transpose(), cmap=plt.get_cmap('viridis'))
#    pylab.matshow(basin_colors.transpose(), cmap=plt.get_cmap('pink'))
    pylab.draw()


def d_arctan():
    def func(x):
        return (1.0 / (1.0 + x**2))
    return func


#########################
#         DRIVER        #
#########################
if __name__ == '__main__':
    log.setLevel('INFO')  # Set the logging level
    # Report basic system info
    log.out.info("Python version: " + sys.version)

    # test_method()

    # Define z^4 -1 as a the function
    coefs = [1.0, -1.0]
    pows = [4, 0]
    f = make_polynomial(coefs, pows)
    df = make_derivative_polynomial(coefs, pows)

#    plot_basins(-1.5, 1.5, -1.5, 1.5, 0.05, 0.05,
#                f=f, df=df)
#    plot_basins(-1.5, 1.5, -1.5, 1.5, 0.01, 0.01,
#                f=f, df=df)
    plot_basins(1.0, 1.35, 1.0, 1.35, 0.001, 0.001,
                f=f, df=df)

#    f = np.sin
#    df = np.cos
#    plot_basins(1.15, 1.4, -0.2, 0.2, 0.003, 0.003, f=f, df=df)

#    f = math.atan
#    df = d_arctan()
#    plot_basins(-2.0, 2.0, -2.0, 2.0, 0.01, 0.01, f=f, df=df)

    # Stop Logging
    log.stopLog()
