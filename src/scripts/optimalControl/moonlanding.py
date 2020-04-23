# ==== Libraries ====
import numpy as np
import matplotlib.pyplot as plt
from casadi import *

# ===================

# ==== Functions ====
def f(x,u):
    """ dx = f(x,u)
        The nonlinear state-equation for the moon landing state-space model.
        Arguments
            x: the current state x(t) = [Pₓ(t), Pᵧ(t), Vₓ(t), Vᵧ(t), m(t)] of the system (x.shape = [n,1])
            u: the current applied control u(t) = [Fₜ(t), θ(t)] action (u.shape = [p,1])
    """
    # Compute state-space dimensions
    nx = x.shape[0]; nu = u.shape[0];

    # Computes and returns the time-derivative of the state vector
    return [x[2],
            x[3],
            c1*(u[0]/m)*cos(u[1])+g_x,
            c1*(u[0]/m)*sin(u[1])+g_y,
            -(c1/c2)*u[0]]
# ---

# ===================

# ==== Variables ====
## 1 System Parameters
# Mass (m); Gravity (g_x, g_y); Maximum Thrust (c1); Engine Efficiency (c2)
m = 1; g_x = 0; g_y = -1.6229; c1 = 44000; c2 = 311*9.81
T = 100; N = 50; tspan = np.arange(0, N)

## 2 Casadi Symbolic Definitions
x = MX.sym('x', 5); u = MX.sym('u', 2)

F   = integrator("F", 'cvodes', {'x':x, 'p':u, 'ode':vcat(f(x, u))})
F   = Function("F", [x, u], [F(x0=x, p=u)['xf']], ['x', 'u'], ['x_next'])

## 3 Casadi Optimization Definitions
ocp = casadi.Opti()
x  = ocp.variable(5, N+1);  u  = ocp.variable(2, N);  x0 = ocp.parameter(5, 1)

# ===================

# ===== Script =====
## 1. Definition of the Optimal Control Problem (OCP)
# Cost function
ocp.minimize( sumsqr(x)+sumsqr(u) )

# Path constraints
for k in tspan:
    ocp.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))

# Bound constraints
ocp.subject_to(u[0,:] >= 0); ocp.subject_to(u[0,:] <= 1)
ocp.subject_to(u[1,:] >= 0); ocp.subject_to(u[1,:] <= 2*pi)

ocp.subject_to(x[4,:] >= 0);
ocp.subject_to(x[1,:] >= 0);

ocp.subject_to(x[:,0] == x0);

## 2. Sets and executes the solver
ocp.solver('ipopt')
ocp.set_value(x0, [30, 1000, 10, 20, 10000])

sol  = ocp.solve()
xout = sol.value(x); uout = sol.value(u)

print(xout)

## 3 Plot the results
plt.figure()
plt.subplot(1,2,1)
plt.plot(xout[0,:], xout[1,:])

plt.subplot(1,2,2)
plt.step(tspan, uout[0,:], linestyle='--')
plt.step(tspan, uout[1,:], linestyle='--')
plt.legend(["u_1(t)", "u_2(t)"])

plt.show()
