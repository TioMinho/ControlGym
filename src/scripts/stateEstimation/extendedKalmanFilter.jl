# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats
using ControlSystems, Plots

# Configurations
theme(:dark)
pyplot(leg=false)

import Base: *
*(v::Any, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Any, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))

# ===================

# ==== Functions ====
# (XE,m,S) = EXTENDEDKALMANFILTER(Y,U,X0,T) compute the filtering distributions from output (Y) and
#	input (U) signals over time (T), considering prior distribution XO.
function extendedKalmanFilter(sys, y, u, t, x0, Q, R)
	# Auxiliary variables
	(f,~,A,C,Δt,Nx,Ny,Nu) = sys

	m = zeros(Nx,length(t))
	S = zeros(Nx,Nx,length(t))

	K(Sk_,Ck) = Sk_ * Ck'*(Ck*Sk_*Ck' + R)^(-1)	# Optimal Kalman Gain

	# Updating the initial state distribution
	m[:,1]   =   x0.μ + K(x0.Σ,C(x0.μ))*(y[:,1] - C(x0.μ)*x0.μ)
	S[:,:,1] = I*x0.Σ - K(x0.Σ,C(x0.μ))*C(x0.μ)*x0.Σ

	p_x = [MvNormal(m[:,1], Symmetric(S[:,:,1]))]

	# == FILTERING LOOP ==
	for k ∈ 1:length(t)-1
		# Compute the Jacobians for this step
		Ak = A(p_x[k].μ, u[:,k])
		Ck = C(p_x[k].μ)

		# Prediction step
		m[:,k+1]   = f(p_x[k].μ, u[:,k])
		S[:,:,k+1] = Ak*S[:,:,k]*Ak' + Q

		# Update step
		m[:,k+1]   = m[:,k+1]   + K(S[:,:,k+1],Ck)*(y[:,k+1] - Ck*m[:,k+1])
		S[:,:,k+1] = S[:,:,k+1] - K(S[:,:,k+1],Ck)*Ck*S[:,:,k+1]

		# Saves the filtering distribution x ~ p(x_k | y_1, ..., y_k)
		p_x = [p_x; MvNormal(m[:,k+1], Symmetric(S[:,:,k+1]))]
	end
	# ====

	return (p_x, m, S)
end

# (Y,T,X) = SIMULATE(SYS,Q,R) 
function simulate(sys, u, t, x0, Q, R)
	# Auxiliary variables
	(f,g,~,~,Δt,Nx,Ny,Nu) = sys

	y = zeros(Ny, length(t))
	x = zeros(Nx, length(t))

	# Create the distributions for the random noise variables
	p_v = MvNormal(zeros(Nx), Q)
	p_z = MvNormal(zeros(Ny), R)

	# == SIMULATION LOOP ==
	for k ∈ 1:length(t)-1
		x[:,k+1] = f(x[:,k], u[:,k]) + rand(p_v)
		y[:,k]   = g(x[:,k]) + rand(p_z)
	end

	y[:,end] = g(x[:,end]) + rand(p_z)	# Last state emission
	# ====

	return (y,t,x)
end

function circle(θ)
	x = -(θ:0.1:(2π+θ)).+π/2
	vert = vcat([(0., 0.)], [(xi,yi) for (xi,yi) in zip(sin.(x), cos.(x))])
	return Shape(vert)
end

function arrow(v,θ)
	θ -= π/2;
	vert = vcat([0.4/(1+v).*(sin(-θ),cos(-θ)), (sin(-θ),cos(-θ)), 0.85.*(sin(-θ+0.1),cos(-θ+0.1)), (sin(-θ),cos(-θ)), 0.85.*(sin(-θ-0.1),cos(-θ-0.1)), (sin(-θ),cos(-θ))]...)
	return Shape(vert)
end,

function arrowRotation(ω, θ)
	θ += π/2;
	if(ω>0); x = -((θ+0.5π-0.25ω):0.2:(θ+0.5π+0.25ω))
			 vert = vcat([(sin(x[end]+π/8), cos(x[end]+π/8)), 0.85.*(sin(x[end]), cos(x[end])), 0.7.*(sin(x[end]+π/8), cos(x[end]+π/8))]...)
			 vert = vcat([(xi,yi).*0.85 for (xi,yi) in zip(sin.(x), cos.(x))], vert)
	else;	 x = -((θ-0.5π+0.25ω):0.2:(θ-0.5π-0.25ω))
			 vert = vcat([(sin(x[1]-π/8), cos(x[1]-π/8)), 0.85.*(sin(x[1]), cos(x[1])), 0.7.*(sin(x[1]-π/8), cos(x[1]-π/8))]...)
			 vert = vcat(vert, [(xi,yi).*0.85 for (xi,yi) in zip(sin.(x), cos.(x))])
	end
	
	vert = vcat(vert, vert[end:-1:1])
	return Shape(vert)
end


# ===================

# ==== Variables ====
# Nonlinear State and Output Equations
Nx = 3; Ny = 3; Nu = 2;
Δt = 0.1; # seconds  (Sampling time)

f(x,u) = x + [cos(x[3])Δt * u[1]
			  sin(x[3])Δt * u[1]
			  u[2]*Δt]
g(x) = x

# Linearized System (Jacobians)
A(xss,uss) = [1  0  -(uss[1]*Δt)sin(xss[3]);
	 		  0  1   (uss[1]*Δt)cos(xss[3]);
	 		  0  0         1        ]
B(xss,uss) = [(Δt)cos(xss[3])  0 ;
	 		  (Δt)sin(xss[3])  0 ;
		  		   0       	  Δt ]
C(xss) = I(Nx)

sys = (f,g,A,C,Δt,Nx,Ny,Nu)

# Input signal
t = 0:0.1:20;
u = [2 .+ 0t  -(0.5π)sin.(0.5t)+(π)sin.(t)]';

# Process and Measurement Noise covariances
Q = 0.001I(3); R = 0.01I(3);

# Initial state
x0   = zeros(3)
p_x0 = MvNormal(x0, 0.001I(3))

# ===================

# ==== Script ====
# Simulates the System
(y, t, x)        = simulate(            sys,    u, t,   x0, Q, R)
(xk, xk_μ, xk_Σ) = extendedKalmanFilter(sys, y, u, t, p_x0, Q, R)

anim = @animate for ti ∈ 1:length(t)
	scatter(y[1,1:ti], y[2,1:ti], m=(:star5, 2, stroke(0)), markeralpha=range(0,0.7,length=ti+1),
               xlim=(min(y[1,:]...)-2, max(y[1,:]...)+2), 
               ylim=(min(y[2,:]...)-2, max(y[2,:]...)+2),
               ticks=nothing, size=(16,9).*30, dpi=400)
	
	if(ti>1)
		plot!(xk_μ[1, 1:(ti-1)], xk_μ[2, 1:(ti-1)], alpha=range(0,0.7,length=ti+1))
	end

	scatter!([xk_μ[1,ti]], [xk_μ[2,ti]], alpha=0.25, marker=(arrow(u[1,ti], xk_μ[3,ti]), 20*u[1,ti], stroke(1, 0.1, :white)))
	scatter!([xk_μ[1,ti]], [xk_μ[2,ti]], alpha=0.80, marker=(circle(xk_μ[3,ti]), 15, :white))
	scatter!([xk_μ[1,ti]], [xk_μ[2,ti]], alpha=0.25, marker=(arrowRotation(u[2,ti], xk_μ[3,ti]), 25, stroke(1, 0.1, :white)))
	savefig("res/tmp/tmp_ekf$(ti).png")
end
gif(anim, "res/ekf_car.gif", fps=10)

# ===================
