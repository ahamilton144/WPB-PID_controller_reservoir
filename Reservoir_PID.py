import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
from random import gauss, uniform, seed
from simple_pid import PID

### general parameters
ny = 6
dt = 1/365
time_steps = ny / dt

### inflow parameters
inflow_base = 20
use_inflow_sin = True
inflow_sin_amp = 4
inflow_sin2_amp = 2
use_inflow_rand_walk = True
inflow_rand_walk_sd = 0.1
use_inflow_jump = True
inflow_jump_prob = 0.01
inflow_jump_amp = 2

### reservoir parameters
storage_0 = 0
max_release = np.inf
max_storage = 10
target_base = 5
use_target_sin = True
target_sin_amp = 2

### PID controller parameters
# PID_params = -np.array([0, 0, 0]) ## no control
# PID_params = -np.array([1, 0, 0]) ## insufficient release to avoid filling up
# PID_params = -np.array([15, 0, 0]) ## better control, but leads to steady state error
# PID_params = -np.array([15, 10000, 0]) ## integral control reduces error to 0 over time
# PID_params = -np.array([150, 100000, 0]) ## increase controls more for faster equilibrium
# PID_params = -np.array([15, 10000, 0.0005]) ## sequence to show oscillations in D control
# PID_params = -np.array([15, 10000, 0.001]) ## sequence to show oscillations in D control
# PID_params = -np.array([15, 10000, 0.005]) ## sequence to show oscillations in D control
# PID_params = -np.array([150, 100000, 0])  ## good for sinusoidal inflow
# PID_params = -np.array([150, 100000, 0])  ## good for sinusoidal inflow & set point
# PID_params = -np.array([150, 100000, 0])  ## good for sinusoidal inflow & set point, gauss noise
# PID_params = -np.array([150, 100000, 0])  ## good for sinusoidal inflow & set point, gauss + jump noise
PID_params = -np.array([15, 100, 0])  ## lower variability policy for sinusoidal inflow & set point, gauss + jump noise

### set random seed for consistent results
rseed = 0
seed(rseed)

### reservoir class
class Reservoir_PID:
    '''
    Class to represent simple reservoir operations using PID controller to control reservoir releases,
    attempting to match reservoir storage to seasonally-varying target under random inflows.
    '''
    def __init__(self):
        '''
        Initialization for Reservoir_PID class.
        '''
        self.inflow = 0.
        self.inflow_rand_walk = 0.
        self.storage = storage_0
        ### initialize PID controller
        self.pid = PID(*PID_params, setpoint=target_base, sample_time=None)
        self.pid.output_limits = (0, max_release)

    def update_inflow(self):
        '''
        Method for updating the inflow of the reservoir each time step (self.t). Depending on parameters (see top of page),
        this can be (a) constant inflow, (b) combination of seasonally varying sinusoids,
        (c) b plus noise in the form of a Gaussian random walk, or (d) c plus a random jump process.
        :return: None
        '''
        inflow = inflow_base
        if use_inflow_sin:
            inflow += inflow_sin_amp * sin(self.t * 2 * pi) + \
                      inflow_sin2_amp * sin(self.t * pi)
        if use_inflow_rand_walk:
            self.inflow_rand_walk += gauss(0, inflow_rand_walk_sd)
            if use_inflow_jump:
                if uniform(0,1) < inflow_jump_prob:
                    if uniform(0,1) < 0.5:
                        self.inflow_rand_walk += inflow_jump_amp
                    else:
                        self.inflow_rand_walk -= inflow_jump_amp
            inflow += self.inflow_rand_walk
        self.inflow = max(inflow, 0)

    def update_reservoir(self, t):
        '''
        Method for updating the reservoir each time step. This involves updating the current inflow and
        the seasonally-varying target, getting the prescribed release from the PID controller, potentially
        modifying this release to ensure mass balance, and finally updating the storage.
        :param t: Total elapsed time (years) since simulation began.
        :return: None
        '''
        ### update inflow
        self.t = t
        self.update_inflow()

        ### if using variable target, assume it is sin with opposite phase of inflows
        if use_target_sin:
            self.pid.setpoint = target_base - target_sin_amp * sin(t * 2 * pi)

        ### get target release from PID
        release_target = self.pid(self.storage)

        ### update storage & release to preserve mass balance
        self.storage += (self.inflow - release_target) * dt
        if self.storage < 0:
            self.release = release_target + self.storage / dt
            self.storage = 0
        elif self.storage > max_storage:
            self.release = release_target + (self.storage - max_storage) / dt
            self.storage = max_storage
        else:
            self.release = release_target


### lists to store results
storages, inflows, releases, targets = [storage_0], [], [], []

### initialize reservoir
reservoir = Reservoir_PID()

### now run simulation, calling PID controller each time to keep reservoir storage constant.
for t in np.arange(0, ny+0.01, dt):
    ### update storage & enforce mass balance
    reservoir.update_reservoir(t)
    storages.append(reservoir.storage)
    releases.append(reservoir.release)
    inflows.append(reservoir.inflow)
    targets.append(reservoir.pid.setpoint)

### plot results
fig,axs = plt.subplots(2,1,figsize=(8,5),gridspec_kw={'hspace':0.05})
ts = np.arange(0, ny+0.01, dt)
tp1 = np.arange(0, ny+0.01+dt, dt)
axs[0].plot(ts, inflows, label='inflow')
axs[0].plot(ts, releases, label='release')
axs[1].plot(ts, targets, label='target')
axs[1].plot(tp1, storages, label='storage')
axs[0].set_xticks(np.arange(ny+1), ['']*(ny+1))
axs[1].set_xlabel('years')
axs[1].set_xticks(np.arange(ny+1), np.arange(ny+1))
axs[0].legend()
axs[1].legend()
axs[0].set_ylabel('Flow (MAF/year)')
axs[1].set_ylabel('Storage (MAF)')

# plt.show()
plt.savefig(f'figs/PID_{PID_params[0]}_{PID_params[1]}_{PID_params[2]}_inflowSin{use_inflow_sin}_inflowRand{use_inflow_rand_walk}_inflowJump{use_inflow_jump}_targetSin{use_target_sin}_seed{rseed}.png', dpi=300, bbox_inches='tight')
