# maths libraries
import numpy as np
from scipy.spatial.distance import pdist, squareform

# plotting libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# turning on interactive mode for pycharm
import matplotlib;

matplotlib.use("TkAgg")


class Population:
    """Population class

    init_state is an [N x 6] array, where N is the number of people.
    It takes the form:

       [[x1, y1, vx1, vy1, i_state1, i_time1],
        [x2, y2, vx2, vy2, i_state2, i_time2],
        ...
        [xN, yN, vxN, vyN, i_stateN, i_timeN]]

    where x1 and y2 are the coordinates,
    vx1 and vy1 are the velocities,
    i_state1 is the infection state: 0 = susceptible, 1= infected, -1= immune,
    i_time1 is the time since infection

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """

    def __init__(self,
                 init_state=[[1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                             [-0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                             [-0.5, -0.5, -0.5, 0.5, 1.0, 0.0]],

                 bounds=[-2, 2, -2, 2],
                 size=0.04,
                 M=0.05,
                 T_recover=100):
        """initiate the class"""
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.T_recover = T_recover

    def step(self, dt):
        """step forward by one time step dt"""
        self.time_elapsed += dt

        # update positions
        self.state[:, :2] += dt * self.state[:, 2:4]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:4]
            v2 = self.state[i2, 2:4]

            # get the infection status
            status1 = self.state[i1, 4]
            status2 = self.state[i2, 4]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:4] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:4] = v_cm - v_rel * m1 / (m1 + m2)

            # case where first ball is infected change ball 2 state to 1
            if status1 == 1 and status2 == 0:
                self.state[i2, 4] = 1
            elif status2 == 1 and status1 == 0:
                self.state[i1, 4] = 1

        # get the infection times
        infection_times = self.state[:, 5]
        infection_states = self.state[:, 4]

        # increment infection times
        for i, (state, times) in enumerate(zip(infection_states, infection_times)):
            # increment the time if the person is infected
            if state == 1:
                self.state[i, 5] += 1

            # change the state if the person is infected long enough
            if times > self.T_recover and state == 1:
                self.state[i, 4] = -1

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1


class AnimatedScatter(object):
    """an animated scatter plot using the Population class as data input"""

    def __init__(self, num_people=100, T_recover=100, frac_infect=0.1):
        # set the number of people
        self.num_people = num_people
        self.T_recover = T_recover
        self.frac_infect = frac_infect

        # set up initial state
        np.random.seed(0)

        # set up the coordinates and velocities with uniform random distribution
        self.init_state = -0.5 + np.random.random((self.num_people, 4))

        # scale the velocities
        self.init_state[:, :2] *= 3.9

        # now add the infection state and the time of infection
        self.init_state = np.hstack((self.init_state,
                                     np.random.choice(2, self.num_people,
                                                      p=[1 - self.frac_infect, self.frac_infect]).reshape(
                                         self.num_people, 1),
                                     np.random.randint(9, size=self.num_people).reshape(self.num_people, 1)))

        self.box = Population(self.init_state, T_recover=self.T_recover)
        self.dt = 1. / 30  # 30fps

        # setup the figure and axes
        self.fig = plt.figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

        # setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """initial drawing of the scatter plot."""
        # set up the colour scale we will use Blue (-1 immune) -> Green (0 susceptible) -> Red (1 infected)
        self.colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        self.n_bins = 3
        self.cm = LinearSegmentedColormap.from_list('infections', self.colors, N=self.n_bins)

        self.particles = self.ax.scatter([], [], c=[], s=25, edgecolors=None, vmin=-1, vmax=1, cmap=self.cm)

        # rect is the box edge
        self.rect = plt.Rectangle(self.box.bounds[::2],
                                  self.box.bounds[1] - self.box.bounds[0],
                                  self.box.bounds[3] - self.box.bounds[2],
                                  ec='none', lw=2, fc='none')

        self.ax.add_patch(self.rect)

        return self.particles, self.rect

    def update(self, i):
        """update the scatter plot."""
        self.box.step(self.dt)

        # update pieces of the animation
        self.rect.set_edgecolor('k')
        self.particles.set_offsets(self.box.state[:, :2])
        self.particles.set_array(self.box.state[:, 4])

        return self.particles, self.rect


if __name__ == '__main__':
    a = AnimatedScatter(num_people=100, T_recover=100, frac_infect=0.1)
    plt.show()
