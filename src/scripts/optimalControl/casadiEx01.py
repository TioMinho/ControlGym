# === Libraries ===
from casadi import *
import matplotlib.pyplot as plt 
import numpy as np

# === Definitions ===
# Variables
x = MX.sym('x', 2)
u = MX.sym('u')

T = 10; N = 20;
tspan = np.arange(0,N)

F_opt = {}
F_opt['tf'] = T/N
F_opt['simplify'] = True
F_opt['number_of_finite_elements'] = 4

# The Differential Algebraic Equations
rhs = vertcat((1-x[1]**2)*x[0]-x[1]+u, x[0])
F   = integrator("F", 'rk', {'x':x, 'p':u, 'ode':rhs}, F_opt)
F   = Function("F", [x, u], [F(x0=x,p=u)['xf']], ['x', 'u'], ['x_next'])


# === Optimization ===
opti = casadi.Opti()

x = opti.variable(2,N+1)
u = opti.variable(1,N)
p = opti.parameter(2,1)

# Cost Function and Constraints
opti.minimize(sumsqr(x)+sumsqr(u))

for k in tspan:
	opti.subject_to(x[:,k+1]==F(x[:,k], u[:,k]))

opti.subject_to(u >= -1)
opti.subject_to(u <=  1)
opti.subject_to(x[:,0] == p)

# Selects the solver and computes the optimization
opti.solver('ipopt')
opti.set_value(p, [0, 1])

sol  = opti.solve()
xout = sol.value(x)
uout = sol.value(u)

print(xout)

# Plot the results
plt.figure()
plt.plot(xout[0,:])
plt.plot(xout[1,:])
plt.step(tspan, uout, linestyle='--')

plt.legend(["x_1(t)", "x_2(t)", "u(t)"])
plt.show()