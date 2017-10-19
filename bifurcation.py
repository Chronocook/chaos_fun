"""
Displays the bifurcations of the logistic map:
x(n+1) = r * x(n) * (1-x(n))

This was popularized in 1976 by Robert May who used it as a simplified 
model of population dynamics. r being the reproduction rate and the 
mortality rate represented as linearly dependent on population density

r>0, r<1: the population will eventually die, independent of the initial population
r>1, r<3: the population reaches r-1/r, independent of the initial population
r>3, r<~3.45: from almost all initial conditions the population will approach 
              permanent oscillations between two values
r>~3.45, r<~3.54: from almost all initial conditions the population will approach 
                  permanent oscillations between four values
r>~3.549, r<4: Crazy time! After this the periods of oscillation bifurcate wildly, 
               sometimes reaching islands of stability then returning to chaos
r>4: After this very few values are stable, almost all diverge quickly

FYI this contains slower code that could be more instructive, then the
roaring fast power of numpy vector math
"""

from uplog import log
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from random import randint
from tqdm import tqdm # Don't really need this but it's cool (pip install tqdm)


def logisticmap(x, r):
    return x * r * (1 - x)


# Return nth iteration of logisticmap(x. r)
def iterate_logistic(n, x, r):
    for i in range(1, n):
        x = logisticmap(x, r)
    return x


def plot_bifurcation_map(min_r=0.0, max_r=4.0, dr=0.0001, fidelity=1):
    """
    
    """
    # Define iteration behavior
    iteration_base = 1000
    cycle_resolution = 32  # Largest cycle size to look for
    iteration_min = iteration_base - (cycle_resolution / 2)  # Minimum number of iterations
    iteration_max = iteration_base + (cycle_resolution / 2)  # Maximum number of iterations

    # Initialize plot
    plt.figure()
    color = (0,0,0)

    # This just goes over the map again with another seed to make it more dense (zoomable)
    for f in range(fidelity):
        # Initialize r and x lists
        r = np.arange(min_r, max_r, dr)
        x = np.empty(r.size)
        
        # Generate list values -- iterate for each value of r
        seed = random.uniform(0.25, 0.75) # Seed value for x in iterations (0, 1)
        for i in tqdm(range(r.size)):
           x[i] = iterate_logistic(randint(iteration_min, iteration_max), seed, r[i])
    
        plt.scatter(r, x, s=0.01, c=color, alpha=1.0)
    plt.show()


def plot_bifurcation_map_quickly(min_r=0.0, max_r=4.0, dr=0.0001, fidelity=1):
    """
    Numpy is almost always the way to go!
    """
    # Define iteration behavior
    iteration_base = 1000
    # Initialize plot
    plt.figure()
    color = (0,0,0)

    # This just goes over the map again with another seed to make it more dense (zoomable)
    for f in tqdm(range(fidelity)):
        # Initialize r and x lists
        r = np.arange(min_r, max_r, dr)
        x = np.random.random_sample(r.size)
        
        for i in range(iteration_base):
           x = x * r * (1 - x)
    
        plt.scatter(r, x, s=0.001, c=color, alpha=1.0)
    plt.show()



#########################
#         DRIVER        #
#########################
if __name__ == '__main__':
    log.setLevel('INFO')  # Set the logging level
    # Report basic system info
    log.out.info("Python version: " + sys.version)

#    # Make crude map
#    plot_bifurcation_map(min_r=0.0, max_r=4.0, dr=0.001, fidelity=5)
#    # Look at interesting bit a little closer
#    plot_bifurcation_map(min_r=2.9, max_r=4.0, dr=0.00005, fidelity=3)
#    # Zoom and enhance! (this takes about 5 minutes)
#    plot_bifurcation_map(min_r=3.55, max_r=3.75, dr=0.000001, fidelity=5) 


    # Make awesome plots (i just do it backwards to not spoiler the presentation)
    plot_bifurcation_map_quickly(min_r=3.55, max_r=3.65, dr=0.000001, fidelity=25) 
    plot_bifurcation_map_quickly(min_r=2.90, max_r=4.00, dr=0.000010, fidelity=5)
    plot_bifurcation_map_quickly(min_r=0.00, max_r=4.00, dr=0.000100, fidelity=5)

    
    # Stop Logging
    log.stopLog()
