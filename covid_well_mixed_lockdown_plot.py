# maths libraries
import numpy as np
from scipy.spatial.distance import pdist, squareform

# plotting libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

# turning on interactive mode for pycharm
import matplotlib;
matplotlib.use("TkAgg")


class Population:
    """Population class

    init_state is an [N x 6] array, where N is the number of people.
    It takes the form:

       [[x_1, y_1, vx_1, vy_1, i_state1, i_time1, q_state_1, x_home_1, y_home_2],
        [x_2, y_2, vx_2, vy_2, i_state2, i_time2, q_state_2, x_home_2, y_home_2],
        ...
        [x_N, y_N, vx_N, vy_N, i_stateN, i_timeN, q_state_N, x_home_N, y_home_N]]

    where x_i and y_i are the coordinates,
    vx_i and vy_i are the velocities,
    i_state_i is the infection state: 0= susceptible, 1= infected, -1= immune,
    i_time_i is the time since infection
    q_state_i = the quarantine state: 0= not at home, 1= at home,
    x_home_i and y_home_i are the home coordinates for those at home

    bounds is the size of the box: [x_min, x_max, y_min, y_max]
    """

    def __init__(self,
                 init_state=[[ 1.0, 0.0, 0.0,-1.0, 1, 0.0, 1, 1.0, 0.0],
                             [-0.5, 0.5, 0.5, 0.5, 1, 0.0, 0,-0.5, 0.5],
                             [-0.5,-0.5,-0.5, 0.5, 0, 1.0, 1,-0.5,-0.5]],

                 bounds=[-2, 2, -2, 2],
                 size=0.04,
                 M=0.05,
                 T_recover=100,
                 q_rad=0.1,
                 q_frac=0.9,
                 q_inplace=True):
        """initiate the class"""
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.T_recover = T_recover
        self.q_rad = q_rad
        self.q_frac = q_frac
        self.q_inplace = q_inplace
        self.num_people = self.init_state.shape[0]

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
                self.state[i, 5] += 1/30 # frame rate of 30 fps

            # change the state if the person is infected long enough
            if times > self.T_recover and state == 1:
                self.state[i, 4] = -1



        # check for crossing absolute boundary
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

        # check for crossing home boundary if quarantine:
        if self.q_inplace is True:
            # [x_min, x_max, y_min, y_max]
            local_bounds = np.hstack(((self.state[:, 7] - self.q_rad).reshape(self.num_people, 1),
                                      (self.state[:, 7] + self.q_rad).reshape(self.num_people, 1),
                                      (self.state[:, 8] - self.q_rad).reshape(self.num_people, 1),
                                      (self.state[:, 8] + self.q_rad).reshape(self.num_people, 1)))

            local_bounds[self.state[:,6] > 0] = self.bounds[:]

            local_crossed_x1 = (self.state[:, 0] < local_bounds[:, 0] + self.size)
            local_crossed_x2 = (self.state[:, 0] > local_bounds[:, 1] - self.size)
            local_crossed_y1 = (self.state[:, 1] < local_bounds[:, 2] + self.size)
            local_crossed_y2 = (self.state[:, 1] > local_bounds[:, 3] - self.size)

            self.state[local_crossed_x1, 0] = local_bounds[local_crossed_x1, 0] + self.size
            self.state[local_crossed_x2, 0] = local_bounds[local_crossed_x2, 1] - self.size

            self.state[local_crossed_y1, 1] = local_bounds[local_crossed_y1, 2] + self.size
            self.state[local_crossed_y2, 1] = local_bounds[local_crossed_y2, 3] - self.size

            self.state[local_crossed_x1 | local_crossed_x2, 2] *= -1
            self.state[local_crossed_y1 | local_crossed_y2, 3] *= -1

class AnimatedScatter(object):
    """an animated scatter plot using the Population class as data input"""

    def __init__(self, num_people=100, T_recover=100, frac_infect=0.1,
                 q_rad=0.5, q_frac=0.9, q_inplace=True, run_time=40):
        # set the number of people
        self.num_people = num_people
        self.T_recover = T_recover
        self.frac_infect = frac_infect
        self.q_rad = q_rad
        self.q_frac = q_frac
        self.q_inplace = q_inplace
        self.run_time = run_time

        # set up initial state
        np.random.seed(0)

        # set up the coordinates and velocities with uniform random distribution
        self.init_state = -0.5 + np.random.random((self.num_people, 4))

        # scale the locations
        self.init_state[:, :2] *= 4

        # now add the infection state and the time of infection
        self.init_state = np.hstack((self.init_state,
                                     np.random.choice(2, self.num_people,
                                                      p=[1 - self.frac_infect, self.frac_infect]).reshape(
                                         self.num_people, 1),
                                     np.random.randint(9, size=self.num_people).reshape(self.num_people, 1),
                                     np.random.choice(2, self.num_people,
                                                      p=[self.q_frac, 1 - self.q_frac]).reshape(
                                         self.num_people, 1),
                                     self.init_state[:, :2]))

        self.box = Population(self.init_state, T_recover=self.T_recover, q_rad=self.q_rad,
                              q_frac=self.q_frac, q_inplace=self.q_inplace)
        self.dt = 1. / 30  # 30fps
        self.total_frames = int(self.run_time/self.dt)

        # setup the figure and axes
        self.fig = plt.figure(figsize=(12,4))
        self.gs = self.fig.add_gridspec(nrows=1, ncols=3)
        #self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax_1 = self.fig.add_subplot(self.gs[0, 0], aspect='equal', autoscale_on=False,
                                         xlim=(-2.1, 2.1), ylim=(-2.1, 2.1))
        self.ax_2 = self.fig.add_subplot(self.gs[0, 1:], autoscale_on=True,
                                         xlim=(0, self.run_time), ylim=(self.frac_infect, 1))

        asp = (self.run_time/(-np.log(self.frac_infect)))/1.5
        self.ax_2.set_aspect(asp)

        self.fig.tight_layout()

        # setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=30, frames=self.total_frames,
                                           init_func=self.setup_plot, blit=True, repeat = False)

        # saving the animation
        file_name = "num_p-{}_fracinf-{}_Trec-{}_qrad{}_qfrac-{}_q-{}.mp4".format(self.num_people,
                                                                                  self.frac_infect,
                                                                                  self.T_recover,
                                                                                  self.q_rad,
                                                                                  self.q_frac,
                                                                                  self.q_inplace)

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        self.ani.save(file_name, writer=writer)

    def setup_plot(self):
        """initial drawing of the scatter plot."""
        # setup the colour scale we will use Blue (-1 immune) -> Green (0 susceptible) -> Red (1 infected)
        self.colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        self.n_bins = 3
        self.cm = LinearSegmentedColormap.from_list('infections', self.colors, N=self.n_bins)

        # setup the scatter plot for the people
        self.people = self.ax_1.scatter([], [], c=[], s=25, edgecolors=None, vmin=-1, vmax=1, cmap=self.cm)

        # rect is the box edge
        self.rect = plt.Rectangle(self.box.bounds[::2],
                                  self.box.bounds[1] - self.box.bounds[0],
                                  self.box.bounds[3] - self.box.bounds[2],
                                  ec='none', lw=2, fc='none')

        self.ax_1.add_patch(self.rect)

        # figure title
        title_str = "Num. = {}, Init. frac. inf. = {}, Inf. time = {}, Quar. rad. = {}, Quar. frac. = {}".format(
            self.num_people,
            self.frac_infect,
            self.T_recover,
            self.q_rad,
            self.q_frac)
        self.fig.suptitle(title_str, fontsize=16)

        # get rid of the spines for the population
        self.ax_1.set_xticks([])
        self.ax_1.set_yticks([])
        edges = ['left', 'right', 'top', 'bottom']
        for edge in edges:
            self.ax_1.spines[edge].set_visible(False)

        # set up the timeline plot
        self.line, = self.ax_2.plot([], [], lw=2)
        self.ax_2.set_yscale('log')

        # get
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        self.ax_2.yaxis.set_major_formatter(formatter)
        self.ax_2.minorticks_off()

        # set up the time, cumulative infected and infected arrays
        self.t = []
        self.cum_inf = []
        self.inf = []

        return self.people, self.rect, self.line,

    def update(self, i):
        """update the scatter plot."""
        self.box.step(self.dt)

        # update pieces of the animation
        self.rect.set_edgecolor('k')
        self.people.set_offsets(self.box.state[:, :2])
        self.people.set_array(self.box.state[:, 4])

        # get the number infected, and susceptible
        infected_state = self.box.state[:, 4]
        num_sus = infected_state[infected_state == 0].shape[0]
        num_inf = infected_state[infected_state == 1].shape[0]

        # append to the arrays as fractions of population
        self.cum_inf.append(1 - num_sus/self.num_people)
        self.inf.append(num_sus/self.num_people)
        self.t.append(i*self.dt)

        # updated the line plot
        self.line.set_data(self.t, self.cum_inf)

        return self.people, self.rect, self.line,


def main():
    a = AnimatedScatter(num_people=900, T_recover=8, frac_infect=0.01,
                        q_rad=0.05, q_frac=0.75, q_inplace=True, run_time=40)
    #plt.show()


if __name__ == '__main__':
    main()