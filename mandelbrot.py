"""
Displays the mandelbrot set:
iteration of: z(n+1) = z(n)^2 + c [z(0)=0)] is bounded


"""

from uplog import log
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm


def mandelbrot_set(xmin, xmax, ymin, ymax, 
                   xres, yres, maxiter, horizon=2.0):
    x = np.linspace(xmin, xmax, xres, dtype=np.float32)
    y = np.linspace(ymin, ymax, yres, dtype=np.float32)
    c = x + y[:, None] * 1j
    n = np.zeros(c.shape, dtype=int)
    z = np.zeros(c.shape, np.complex64)
    for this_n in tqdm(range(maxiter)):
        i = np.less(abs(z), horizon)
        n[i] = this_n
        z[i] = z[i]**2 + c[i]
    n[n == maxiter - 1] = 0
    return z, n


def plot_mandelbot(xmin=-2.25, xmax=0.75, xres=3000,
                   ymin=-1.25, ymax=1.25, yres=2500,
                   maxiter=200, horizon=2.0**40):
    
    z, n = mandelbrot_set(xmin, xmax, ymin, ymax, 
                          xres, yres, maxiter, horizon)
    
    # Plot the results... dig the fancy pylab skills!
    xaxis_res = int(xres / 20)
    yaxis_res = int(yres / 20)
    x_axis = np.linspace(xmin, xmax, num=xres)
    y_axis = np.linspace(ymin, ymax, num=yres)
    x_labels = []
    y_labels = []
    for i in range(0, len(x_axis), xaxis_res):
        x_labels.append(str(round(x_axis[i], 2)))
    for i in range(0, len(y_axis), yaxis_res):
        y_labels.append(str(round(y_axis[i], 2)))

    pylab.matshow(n, cmap=plt.cm.hot, interpolation='nearest')
    pylab.xticks(np.arange(0, xres, xaxis_res), x_labels)
    pylab.yticks(np.arange(0, yres, yaxis_res), y_labels)
    pylab.xticks(rotation=90)
    pylab.xlim(0, xres)
    pylab.ylim(yres, 0)
    pylab.xlabel('x')
    pylab.ylabel('y')
    ax = pylab.colorbar()
    ax.set_label('Iterations till divergence')
    pylab.draw()


#########################
#         DRIVER        #
#########################
if __name__ == '__main__':
    log.setLevel('INFO')  # Set the logging level
    # Report basic system info
    log.out.info("Python version: " + sys.version)

    # TEST
    plot_mandelbot(xmin=-2.25, xmax=0.75, xres=300,
                   ymin=-1.25, ymax=1.25, yres=250,
                   maxiter=250, horizon=2.0**40)

    # plot_mandelbot(xmin=-0.35, xmax=0.10, xres=6000,
    #                ymin=-1.12, ymax=-0.63, yres=6000,
    #                maxiter=250, horizon=2.0**40)
    #
    # plot_mandelbot(xmin=-2.25, xmax=0.75, xres=3000,
    #                ymin=-1.25, ymax=1.25, yres=2500,
    #                maxiter=250, horizon=2.0**40)

    
    # Stop Logging
    log.stopLog()
