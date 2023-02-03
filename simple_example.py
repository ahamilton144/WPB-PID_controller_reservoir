import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, sin, cos
from random import gauss, uniform, seed
from simple_pid import PID
import time

### parameters
storage = 0
total_time = 1
ny = 6
setpoint_base = 5
use_setpoint_sin = True
setpoint_sin_amp = 2
setpoint_sin_period = total_time / ny
sample_time = total_time / ny / 365
max_release = np.inf
max_storage = 10
inflow_base = 20
use_inflow_sin = True
inflow_sin_amp = 4
inflow_sin2_amp = 2
inflow_sin_period = total_time / ny
use_inflow_rand_walk = True
inflow_rand_walk_sd = 0.2
use_inflow_jump = True
inflow_jump_prob = 0.1
inflow_jump_amp = 3
seed(3)

# pid_params = -np.array([10, 100, 0]) ## good for constant inflow, slow approach
# pid_params = -np.array([100, 1000, 0])  ## good for sinusoidal inflow
# pid_params = -np.array([100, 1000, 0])  ## good for sinusoidal inflow & set point
# pid_params = -np.array([100, 1000, 0])  ## good for sinusoidal inflow & set point, gauss noise
pid_params = -np.array([100, 1000, 0])  ## good for sinusoidal inflow & set point, gauss + jump noise


### reservoir class
class Reservoir:
    def __init__(self, storage_0=0):
        self.inflow = 0.
        self.inflow_rand_walk = 0.
        self.storage = storage_0

    def update_inflow(self):
        inflow = inflow_base
        if use_inflow_sin:
            inflow += inflow_sin_amp * sin(self.dt_total * 2 * pi / inflow_sin_period) + \
                      inflow_sin2_amp * sin(self.dt_total * 2 * pi / inflow_sin_period *2)
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

    def update_reservoir(self, release_target, dt, dt_total):
        ### update inflow
        self.dt_total = dt_total
        self.update_inflow()
        ### update storage based on release_target from PID
        self.storage += (self.inflow - release_target) * dt
        if self.storage < 0:
            release = release_target + self.storage / dt
            self.storage = 0
            return self.storage, self.inflow, release
        elif self.storage > max_storage:
            release = release_target + (self.storage - max_storage)
            self.storage = max_storage
        else:
            return self.storage, self.inflow, release_target


### set up for simulation
storages, inflows, releases, setpoints = [storage], [], [], []
reservoir = Reservoir(storage)
pid = PID(*pid_params, setpoint=setpoint_base, sample_time=sample_time)
pid.output_limits = (0, max_release)

start_time = time.time()
last_time = start_time

### now run simulation, calling PID controller each time to keep reservoir storage constant.
while time.time() - start_time < total_time:
    current_time = time.time()
    dt = current_time - last_time
    dy = dt * ny/total_time
    dt_total = current_time - start_time
    if dt > sample_time:
        ### if using variable setpoint, assume it is sin with opposite phase of inflows
        if use_setpoint_sin:
            pid.setpoint = setpoint_base - setpoint_sin_amp * sin(dt_total* 2 * pi / setpoint_sin_period)

        ### get target release from PID
        release_target = pid(storage)

        ### update storage & enforce mass balance
        storage, inflow, release = reservoir.update_reservoir(release_target, dy, dt_total)
        storages.append(storage)
        releases.append(release)
        inflows.append(inflow)
        setpoints.append(pid.setpoint)
        last_time = current_time

### plot results
fig,axs = plt.subplots(2,1,figsize=(8,5),gridspec_kw={'hspace':0.05})
t = np.arange(len(inflows)) * sample_time
tp1 = np.arange(len(inflows)+1) * sample_time
axs[0].plot(t,inflows, label='inflow')
axs[0].plot(t,releases, label='release')
axs[1].plot(t,setpoints, label='target')
axs[1].plot(tp1,storages, label='storage')
axs[1].set_xlabel('years')
axs[1].set_xticks(np.arange(0,total_time+0.01,total_time/ny), np.arange(ny+1))
axs[0].set_xticks(np.arange(0,total_time+0.01,total_time/ny), ['']*(ny+1))
axs[0].legend()
axs[1].legend()
axs[0].set_ylabel('Flow (MAF/year)')
axs[1].set_ylabel('Storage (MAF)')

plt.savefig('figs/pid.png', dpi=300, bbox_inches='tight')
