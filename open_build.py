import casadi as cs
import opengen as og
import matplotlib.pyplot as plt
import numpy as np

T = 20 # Horizons
NU = 7
NX = 14
sampling_time = 0.1

# weight of the cost function
w = [20,5,1.3,1.3,0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

# param of L1
r_min = 7

# Params for L3
beta = 50
b_max = 4
c = 10
k = -0.5

# Constraints params
a_min = [-2,-2,-2,-2,-2,-2,-2]
a_max = [2,2,2,2,2,2,2]

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

    return cs.vcat([VA,VB,Vb,aA,aB,ab])

def dynamic_dt(x, u):
    dx = dynamic_ct(x, u)
    return cs.vcat([x[i] + sampling_time * dx[i] for i in range(NX)])

def stage_cost(x, u):
    PA = x[:3]
    PB = x[3:6]
    Pb = x[6]
    VA = x[7:10]
    VB = x[10:13]
    Vb = x[13]

    Vd = 7

    r_ABs = 0
    for i in range(3):
        r_ABs += (PA[i]-PB[i])**2

    L1 = w[0]/(r_ABs-r_min**2)
    L2 = (-1) * w[1] * (cs.log(Pb + b_max) + cs.log(b_max - Pb))

    L3_1 = ((k*(PA[1]-beta-Pb)-((k*(PA[1]-beta-Pb))**2+c)**0.5))/2 + beta
    L3_2 = (-k*(PA[1]+beta/k-beta-Pb)+((k*(PA[1]+beta/k-beta-Pb))**2+c)**0.5)/2 + Pb
    L3 = w[2]*(PA[2] - (L3_1 + L3_2)) + w[3]*PA[0]**2 

    L4 = w[4]*(VA[1]-Vd)**2
    L5 = w[5]*(VB[1]-Vd)**2

    W = cs.diag(w[6:])
    L6 = cs.mtimes([u.T, W, u])

    L7 = 1.5*(PA[2]-L3_1-L3_2)**2
    L8 = 5*PB[0]**2+5*PB[2]**2
    return L1+L2+L3+L4+L5+L6+L7+L8

x_0 = cs.MX.sym("x_0", NX)
#x_ref = cs.MX.sym("x_ref", 1)
u_k = [cs.MX.sym('u_' + str(i), NU) for i in range(T)]

x_t = x_0
total_cost = 0
for t in range(0, T):
    #total_cost+=stage_cost(x_t, x_ref, u_k[t])
    total_cost+=stage_cost(x_t, u_k[t])
    x_t = dynamic_dt(x_t, u_k[t])

# Constraints
bounds = og.constraints.Rectangle(a_min*T, a_max*T)

# Code generation
optimization_variables = []
optimization_parameters = []

optimization_variables += u_k
optimization_parameters += [x_0]
#optimization_parameters += [x_ref]
optimization_variables = cs.vertcat(*optimization_variables)
optimization_parameters = cs.vertcat(*optimization_parameters)

problem = og.builder.Problem(optimization_variables,
                             optimization_parameters,
                             total_cost)  \
            .with_constraints(bounds)

build_config = og.config.BuildConfiguration()  \
    .with_build_directory("python_build")      \
    .with_build_mode("release")                \
    .with_tcp_interface_config()

meta = og.config.OptimizerMeta().with_optimizer_name("xuhao_drones")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-6)\
    .with_initial_tolerance(1e-6)

builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config, solver_config)
builder.build()