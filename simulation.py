import casadi as cs
import opengen as og
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

mng = og.tcp.OptimizerTcpManager("python_build/xuhao_drones")
mng.start()

x_state_0 = [1,0,50,0,0,0,0,0,12,0,0,7,0,0]
simulation_steps = 500

state_sequence = x_state_0
input_sequence = []

NU = 7
NX = 14
sampling_time = 0.1

def dynamic_ct(x, u):
    PA = x[:3]
    PB = x[3:6]
    Pb = x[6]
    VA = x[7:10]
    VB = x[10:13]
    Vb = x[13]

    aA = u[:3]
    aB = u[3:6]
    ab = u[6]

    #return cs.vcat([VA,VB,Vb,aA,aB,ab])
    return VA + VB + [Vb] + aA + aB + [ab]
    #return np.vstack([VA,VB,Vb,aA,aB,ab])

def dynamic_dt(x, u):
    dx = dynamic_ct(x, u)
    #return cs.vcat([x[i] + sampling_time * dx[i] for i in range(NX)])
    #return np.vstack([x[i] + sampling_time * dx[i] for i in range(NX)])
    r = []
    for i in range(NX):
        r.append(x[i] + sampling_time * dx[i])
    return r

x = x_state_0
for k in range(simulation_steps):
    #x_np = np.array(x).flatten()
    solver_status = mng.call(x)
    us = solver_status['solution']
    u = us[0:7]
    x_next = dynamic_dt(x, u)
    state_sequence += x_next
    input_sequence += [u]
    x = x_next

mng.kill()

#time = np.arange(0, sampling_time*simulation_steps, sampling_time)

PA_x = state_sequence[0:NX*simulation_steps:NX]
PA_y = state_sequence[1:NX*simulation_steps+1:NX]
PA_z = state_sequence[2:NX*simulation_steps+2:NX]
PB_x = state_sequence[3:NX*simulation_steps+3:NX]
PB_y = state_sequence[4:NX*simulation_steps+4:NX]
PB_z = state_sequence[5:NX*simulation_steps+5:NX]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the trajectory
# Plot the trajectory for PA
ax.plot(PA_x, PA_y, PA_z, label="PA Trajectory")

# Plot the trajectory for PB
ax.plot(PB_x, PB_y, PB_z, label="PB Trajectory")

ax.set_xlim([-0.5, 1.0])
ax.set_ylim([-10, 350])
ax.set_zlim([-10, 60])

# Setting labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Adding a legend
ax.legend()

# Show the plot
plt.show()