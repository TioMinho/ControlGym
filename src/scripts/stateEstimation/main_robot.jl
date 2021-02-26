# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats
using ControlSystems, Plots

folders = (@__DIR__).*["/filters", "/utils"]
for folder in folders;	for file in readdir(folder)
	include(folder*"/"*file)
end; end

# ==== Variables ====
# Nonlinear State and Output Equations
Nₓ = 3; Nᵧ = 3; Nᵤ = 2;
Δt = 0.1; # seconds  (Sampling time)

f(x,u) = x + [cos(x[3])Δt * u[1]
			  sin(x[3])Δt * u[1]
			  u[2]*Δt]
g(x) = x

# Linearized System (Jacobians)
xₛₛ = [0; 0; π/4]; uₛₛ = [1; 0];

A(xₛₛ,uₛₛ) = [1  0  -(uₛₛ[1]*Δt)sin(xₛₛ[3]);
	 		  0  1   (uₛₛ[1]*Δt)cos(xₛₛ[3]);
	 		  0  0         1        ]
B(xₛₛ,uₛₛ) = [(Δt)cos(xₛₛ[3])  0 ;
	 		  (Δt)sin(xₛₛ[3])  0 ;
		  		   0       	  Δt ]
C(xₛₛ) = I(Nₓ)

Aₛₛ = A(xₛₛ,uₛₛ); Bₛₛ = B(xₛₛ,uₛₛ); Cₛₛ = C(xₛₛ)
sys  = (f,g,A,B,C,Δt,Nₓ,Nᵧ,Nᵤ)
sysL = (f,g,Aₛₛ,Bₛₛ,Cₛₛ,Δt,Nₓ,Nᵧ,Nᵤ)

# ==== Simulation Parameters ====
# Input signal
t = 0:Δt:20;
u = [2 .+ 0t  -(0.5π)sin.(0.5t)+(π)sin.(t)]';

# Process and Measurement Noise covariances
Q = 0.001I(3); R = 0.01I(3);

# Initial state
x₀ = zeros(3)
X₀ = MvNormal(x₀, 0.001I(3))

# ===================

# ==== Script ====
# Simulates the System
(y, t, x)    = sim(sys,    u, t, x₀, Q=Q, R=R, mode="nonlinear")
(Xₑ, μₑ, Σₑ) = UKF(sys, y, u, t, X₀, Q, R)

# Plot the results
plot_trajectory(t, x, y, u, xₑ=μₑ, anim=true, name="car")
