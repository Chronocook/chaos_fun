"""
Completely based around:
https://gist.github.com/benrules2/220d56ea6fe9a85a4d762128b11adfba
Cleaned up the methods a bit and edited to follow PEP8
Added movie code to shnaz this up a bit
Added Vesta and a couple of good degenerate orbits
Added option to ignore objects with the same name in gravity calcs
"""

from uplog import log
import math
import sys
import random
import matplotlib.pyplot as plot
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# Only need to run once to download ffmpeg
# import imageio
# imageio.plugins.ffmpeg.download()
import moviepy.editor as mpy  # pip install moviepy


IMAGE_ARRAY = None
TIMESTEP = None
COLOR = {}
COLOR['black'] = [0.0, 0.0, 0.0]
COLOR['light_black'] = [5.0, 5.0, 5.0]
COLOR['white'] = [250.0, 250.0, 250.0]
COLOR['light_white'] = [255.0, 255.0, 255.0]
COLOR['grey'] = [100.0, 100.0, 100.0]
COLOR['light_grey'] = [155.0, 155.0, 155.0]
COLOR['red'] = [220.0, 35.0, 35.0]
COLOR['light_red'] = [250.0, 150.0, 150.0]
COLOR['blue'] = [30.0, 30.0, 150.0]
COLOR['light_blue'] = [75.0, 75.0, 225.0]
COLOR['yellow'] = [200.0, 200.0, 0.0]
COLOR['light_yellow'] = [200.0, 200.0, 100.0]


def make_frame(t):
    index = int(t/TIMESTEP)
    return IMAGE_ARRAY[:, :, index, :]


def draw_circle(image_array, x, y, radius, color='white', fade=True):
    xp = min(x+radius+1, len(image_array[0, :])-1)
    xm = max(x-radius, 0)
    yp = min(y+radius+1, len(image_array[:, 0])-1)
    ym = max(y-radius, 0)
    if fade:
        kernel = np.zeros((2*radius+1, 2*radius+1, 3))
        ymask, xmask = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = xmask**2 + ymask**2 <= radius**2
        kernel[mask] = COLOR['light_'+color]
        mask = xmask**2 + ymask**2 <= (radius/1.3)**2
        kernel[mask] = COLOR[color]
    else:
        kernel = np.zeros((2*radius+1, 2*radius+1, 3))
        ymask, xmask = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = xmask**2 + ymask**2 <= radius**2
        kernel[mask] = COLOR[color]

    image_array[ym:yp, xm:xp, :] = kernel
    return image_array


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Body:
    def __init__(self, location, mass, velocity, name=""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name


# Planet data (location (m),  mass (kg),  velocity (m/s)
sun = {"location": Point(0, 0, 0),  "mass": 2e30,  "velocity": Point(0, 0, 0)}
mercury = {"location": Point(0, 5.7e10, 0),  "mass": 3.285e23,  "velocity": Point(47000, 0, 0)}
venus = {"location": Point(0, 1.1e11, 0),  "mass": 4.8e24,  "velocity": Point(35000, 0, 0)}
earth = {"location": Point(0, 1.5e11, 0),  "mass": 6e24,  "velocity": Point(30000, 0, 0)}
mars = {"location": Point(0, 2.2e11, 0),  "mass": 2.4e24,  "velocity": Point(24000, 0, 0)}
jupiter = {"location": Point(0, 7.7e11, 0),  "mass": 1e28,  "velocity": Point(13000, 0, 0)}
saturn = {"location": Point(0, 1.4e12, 0),  "mass": 5.7e26,  "velocity": Point(9000, 0, 0)}
uranus = {"location": Point(0, 2.8e12, 0),  "mass": 8.7e25,  "velocity": Point(6835, 0, 0)}
neptune = {"location": Point(0, 4.5e12, 0),  "mass": 1e26,  "velocity": Point(5477, 0, 0)}
pluto = {"location": Point(0, 3.7e12, 0),  "mass": 1.3e22,  "velocity": Point(4748, 0, 0)}
vesta = {"location": Point(0, 3.53e11, 0),  "mass": 2.59e20,  "velocity": Point(19340, 0, 0)}
vesta_degenerate_1 = {"location": Point(0, 3.53e11, 0),  "mass": 2.59e20,  "velocity": Point(14350, 0, 0)}
vesta_degenerate_2 = {"location": Point(0, 3.53e11, 0),  "mass": 2.59e20,  "velocity": Point(14500, 0, 0)}


def calculate_single_body_acceleration(bodies, body_index):
    g_const = 6.67408e-11  # m3 kg-1 s-2
    acceleration = Point(0, 0, 0)
    target_body = bodies[body_index]
    epsilon = 1.0
    for index, external_body in enumerate(bodies):
        if (index != body_index) and (target_body.name != external_body.name):
            r = (target_body.location.x - external_body.location.x)**2 + \
                (target_body.location.y - external_body.location.y)**2 + \
                (target_body.location.z - external_body.location.z)**2
            r = math.sqrt(r)
            tmp = g_const * external_body.mass / (epsilon + r**3)
            acceleration.x += tmp * (external_body.location.x - target_body.location.x)
            acceleration.y += tmp * (external_body.location.y - target_body.location.y)
            acceleration.z += tmp * (external_body.location.z - target_body.location.z)

    return acceleration


def compute_velocity(bodies, time_step=1):
    for body_index, target_body in enumerate(bodies):
        acceleration = calculate_single_body_acceleration(bodies, body_index)
        target_body.velocity.x += acceleration.x * time_step
        target_body.velocity.y += acceleration.y * time_step
        target_body.velocity.z += acceleration.z * time_step


def update_location(bodies, time_step=1):
    for target_body in bodies:
        target_body.location.x += target_body.velocity.x * time_step
        target_body.location.y += target_body.velocity.y * time_step
        target_body.location.z += target_body.velocity.z * time_step


def compute_gravity_step(bodies, time_step=1):
    compute_velocity(bodies, time_step=time_step)
    update_location(bodies, time_step=time_step)


def plot_output(bodies, outfile='orbits.png', make_plot=True, make_movie=False):
    num_coords = len(bodies[0]['x'])
    log.out.info("Bodies contain:" + str(num_coords) + " frames.")
    max_range = 0
    for current_body in bodies:
        max_dim = max(max(current_body["x"]), max(current_body["y"]), max(current_body["z"]))
        if max_dim > max_range:
            max_range = max_dim

    if make_plot:
        fig = plot.figure()
        colours = ['r', 'b', 'g', 'y', 'm', 'c']
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for current_body in bodies:
            ax.plot(current_body["x"], current_body["y"], current_body["z"],
                    c=random.choice(colours), label=current_body["name"])

        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.legend()

        if outfile:
            plot.savefig(outfile)
        else:
            plot.show()

    if make_movie:
        max_range = max_range * 2.0
        xrange = max_range
        xres = 600
        xstep = 2.0 * xrange / (xres-1)
        xgrid = np.arange(-1.0*xrange, xrange+xstep, xstep)
        yrange = max_range
        yres = 600
        ystep = 2.0 * yrange / (yres-1)
        ygrid = np.arange(1.0*yrange, -1.0*yrange-ystep, -1.0*ystep)

        global IMAGE_ARRAY
        IMAGE_ARRAY = np.zeros([len(ygrid), len(xgrid), num_coords, 3],  dtype=float)
        for body in bodies:
            if body['name'] == 'sun':
                size = 9
                color = 'yellow'
            elif 'vesta' in body['name']:
                size = 3
                color = 'red'
            else:
                size = 6
                color = 'blue'
            for i in range(len(body['x'])):
                index_x = (np.abs(xgrid - body['x'][i])).argmin()
                index_y = (np.abs(ygrid - body['y'][i])).argmin()
                IMAGE_ARRAY[:, :, i, :] = draw_circle(IMAGE_ARRAY[:, :, i, :], index_x, index_y, size, color)

        global TIMESTEP
        fps = 30.0
        movie_length_sec = 60.0
        TIMESTEP = movie_length_sec / num_coords
        animation = mpy.VideoClip(make_frame, duration=movie_length_sec)
        animation.write_videofile('gravity_rocks.mp4', fps=fps)


def run_simulation(bodies, time_step=1, number_of_steps=10000, report_freq=None):
    if report_freq is None:
        report_freq = time_step
    # Create output container for each body
    body_locations_hist = []
    for current_body in bodies:
        body_locations_hist.append({"x": [], "y": [], "z": [], "name": current_body.name})

    for i in tqdm(range(1, number_of_steps)):
        compute_gravity_step(bodies, time_step=time_step)

        if i % report_freq == 0:
            for index, body_location in enumerate(body_locations_hist):
                body_location["x"].append(bodies[index].location.x)
                body_location["y"].append(bodies[index].location.y)
                body_location["z"].append(bodies[index].location.z)

    return body_locations_hist


def main():
    log.out.info("Starting n-body integration")

    # Build list of planets in the simulation, or create your own
    bodies = [
        Body(location=sun["location"], mass=sun["mass"], velocity=sun["velocity"], name="sun"),
        Body(location=earth["location"], mass=earth["mass"], velocity=earth["velocity"], name="earth"),
        Body(location=mars["location"], mass=mars["mass"], velocity=mars["velocity"], name="mars"),
        Body(location=venus["location"], mass=venus["mass"], velocity=venus["velocity"], name="venus"),
        Body(location=vesta_degenerate_1["location"], mass=vesta_degenerate_1["mass"],
             velocity=vesta_degenerate_1["velocity"], name="vesta"),
        Body(location=vesta_degenerate_2["location"], mass=vesta_degenerate_2["mass"],
             velocity=vesta_degenerate_2["velocity"], name="vesta")
    ]

    motions = run_simulation(bodies, time_step=1000, number_of_steps=500000)
    # motions = run_simulation(bodies, time_step=100, number_of_steps=50000)

    plot_output(motions, make_plot=False, make_movie=True)


if __name__ == "__main__":
    log.setLevel('INFO')  # Set the logging level
    # Report basic system info
    log.out.info("Python version: " + sys.version)

    main()

    log.stopLog()
