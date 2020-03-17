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
# XE = KALMANFILTER(Y,U,X0,T) compute the filtering distributions from output (Y) and
#	input (U) signals over time (T), considering prior distribution XO.
function kalmanFilter(sys, y, u, t, x0, Q, R)
	# Auxiliary variables
	(A,B,C) = (sys.A, sys.B, sys.C)
	m = zeros(size(A,1), length(t))
	S = zeros(size(A)..., length(t))

	K(Sk_) = Sk_ * C'*(C*Sk_*C' + R)^(-1)

	# Updating the initial state distribution
	m[:,1]   =   x0.μ + K(x0.Σ)*(y[:,1] - C*x0.μ)
	S[:,:,1] = I*x0.Σ - K(x0.Σ)*C*x0.Σ

	x = [MvNormal(m[:,1], Symmetric(S[:,:,1]))]

	# == FILTERING LOOP ==
	for k ∈ 1:length(t)-1
		# Prediction
		m[:,k+1]   = A*x[k].μ + B*u[:,k]
		S[:,:,k+1] = A*S[:,:,k]*A' + Q

		# Update
		m[:,k+1]   =   x[k].μ + K(x[k].Σ)*(y[:,k+1] - C*x[k].μ)
		S[:,:,k+1] = I*x[k].Σ - K(x[k].Σ)*C*x[k].Σ

		# Saves the filtering distribution x ~ p(x_k | y_1, ..., y_k)
		x = [x; MvNormal(m[:,k+1], Symmetric(S[:,:,k+1]))]
	end
	# ====

	return x
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

# == Variables ==
# Operating point
xss = yss = θss = π/4; vss = 1; ωss = 0; Δt = 0.1;
x_ss = [xss; yss; θss]; u_ss = [vss; ωss];

# Linearized System (Jacobians)
A = [1  0  -(vss*Δt)sin(θss);
	 0  1   (vss*Δt)cos(θss);
	 0  0         1        ]
B = [(Δt)cos(θss)  0 ;
	 (Δt)sin(θss)  0 ;
		  0       Δt ]
C = I(3)
D = zeros(3,2)

sys = ss(A, B, C, D, Δt)

# Input signal
t = 0:0.1:20;
u = [2 .+ 0t  -(0.5π)sin.(0.5t)+(π)sin.(t)]

# ===================

# ==== Script ====
# Simulates the System
(y,t,x) = lsim(sys, u.-u_ss', t, x0=-x_ss)
xk = kalmanFilter(sys, y', u'.-u_ss, t, MvNormal(-x_ss, I(3)), I(3), I(3))

# anim = @animate for ti = 1:length(t)
# 	plot(y[1:ti,1], y[1:ti,2], alpha=range(0,0.7,length=ti+1),
# 			xlim=(min(y[:,1]...)-2, max(y[:,1]...)+2), 
# 			ylim=(min(y[:,2]...)-2, max(y[:,2]...)+2),
# 			ticks=nothing, size=(16,9).*25, dpi=400)
	
# 	scatter!([y[ti,1]], [y[ti,2]], alpha=0.90, marker=(circle(y[ti,3]), 15, :white))
# 	scatter!([y[ti,1]], [y[ti,2]], alpha=0.25, marker=(arrow(u[ti,1], y[ti,3]), 20*u[ti,1], stroke(1, 0.1, :white)))
# 	scatter!([y[ti,1]], [y[ti,2]], alpha=0.25, marker=(arrowRotation(u[ti,2], y[ti,3]), 25, stroke(1, 0.1, :white)))
# 	savefig("res/tmp/tmp_$(ti).png")
# end
# gif(anim, "res/car.gif", fps=10)


# ===================
