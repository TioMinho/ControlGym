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
function uTransform(X, f, λ; u=Nothing)
# (𝓧,𝓨) = UTRANSFORM(X,f,λ; u=Nothing)
# 	Given a distribution X ~ 𝓝(μ,Σ) and a nonlinear function f:ℝⁿ×ℝᵖ → ℝᵐ computes the
#	set of sigma-points
#		𝓧 : {𝓧⁽⁰⁾, ..., 𝓧⁽²ⁿ⁺¹⁾} and
#		𝓧⁽⁰⁾ = μ, 𝓧⁽ᵏ⁾ = μ + √(n+λ)[√Σ]ₖ, 𝓧⁽ⁿ⁺ᵏ⁾ = μ - √(n+λ)[√Σ]ₖ	(k=1,⋯,n)
#	and applies the nonlinear function to obtain the transformed set
#		𝓨 : {f(𝓧⁽ᵏ⁾,uₖ)}		(k=1,⋯,n)
#
	# Unpack the mean and variance of X
	(μ, Σ) = [X.μ, I*X.Σ]; n = size(μ,1)

	# Construct the sigma-point sets
	𝓧 = [ [μ]
		[ [μ+(√(n+λ)*√(Σ))[:,i]] for i in 1:n]...
		[ [μ-(√(n+λ)*√(Σ))[:,i]] for i in 1:n]...]

	if u==Nothing
		𝓨 = [f(𝓧⁽ᵏ⁾) for 𝓧⁽ᵏ⁾ in 𝓧]
	else
		𝓨 = [f(𝓧⁽ᵏ⁾, u) for 𝓧⁽ᵏ⁾ in 𝓧]
	end

	return (𝓧,𝓨)
end

function unscentedKalmanFilter(sys, y, u, t, x₀, Q, R; α=1, κ=1, β=0)
# (Xₑ,μ,Σ) = UNSCENTEDKALMANFILTER(SYS,Y,U,T,X₀,Q,R;α=1,κ=1)
#	Computes the filtering distributions from output (Y) and input (U) signals over time (T),
#	considering prior distribution XO.
#
	# Auxiliary variables
	(f,g,~,~,Δt,Nₓ,Nᵧ,Nᵤ) = sys

	μ = zeros(Nₓ,   length(t))		# List of means 	(μ = [μ₀,⋯,μₜ])
	Σ = zeros(Nₓ,Nₓ,length(t))		# List of variances (Σ = [Σ₀,⋯,Σₜ])

	λₓ = α^2(Nₓ+κ)-Nₓ; λᵧ = α^2(Nᵧ+κ)-Nᵧ
	Wₓ⁽ᵐ⁾ = [λₓ/(Nₓ+λₓ); ones(2Nₓ).*1/(2(Nₓ+λₓ))]; Wₓ⁽ᶜ⁾ = [λₓ/(Nₓ+λₓ)+(1-α^2+β); ones(2Nₓ).*1/(2(Nₓ+λₓ))];
	Wᵧ⁽ᵐ⁾ = [λᵧ/(Nᵧ+λᵧ); ones(2Nᵧ).*1/(2(Nᵧ+λᵧ))];    Wᵧ⁽ᶜ⁾ = [λᵧ/(Nᵧ+λᵧ)+(1-α^2+β); ones(2Nᵧ).*1/(2(Nᵧ+λᵧ))]

	# Auxiliary functions
	m(W,𝓧) 		= sum([W[i]*𝓧[i] for i in 1:length(W)])				  # Estimated mean
	S(W,𝓧,μ)		= sum([W[i]*(𝓧[i]-μ)*(𝓧[i]-μ)' for i in 1:length(W)])    # Estimated variance
	C(W,𝓧,𝓨,μₓ,μᵧ) = sum([W[i]*(𝓧[i]-μₓ)*(𝓨[i]-μᵧ)' for i in 1:length(W)])  # Estimated covariance

	K(Sₖ,Cₖ) = Cₖ*(Sₖ+R)^(-1)		# Optimal Kalman Gain

	# 1. Updating the initial state distribution
	(𝓧,𝓨) = uTransform(x₀, g, λᵧ)
	yₖ = m(Wᵧ⁽ᵐ⁾,𝓨); Sₖ = S(Wᵧ⁽ᶜ⁾,𝓨,yₖ); Cₖ = C(Wᵧ⁽ᶜ⁾,𝓧,𝓨,x₀.μ,yₖ)

	μ[:,1]   =   x₀.μ + K(Sₖ,Cₖ)*(y[:,1] - yₖ)
	Σ[:,:,1] = I*x₀.Σ - K(Sₖ,Cₖ)*(Sₖ+R)*K(Sₖ,Cₖ)'

	Xₑ = [MvNormal(μ[:,1], Symmetric(Σ[:,:,1]))]

	# == FILTERING LOOP ==
	for k ∈ 1:length(t)-1
		# 2. Prediction step
		(~,𝓧) = uTransform(Xₑ[end], f, λₓ, u=u[:,k])
		μ[:,k+1]   = m(Wₓ⁽ᵐ⁾, 𝓧)
		Σ[:,:,k+1] = S(Wₓ⁽ᵐ⁾, 𝓧, μ[:,k+1]) + Q

		Xₚ = MvNormal(μ[:,k+1], Symmetric(Σ[:,:,k+1]))	# Predictive Distribution

		# 3. Update step
		(~,𝓨) = uTransform(Xₚ, g, λᵧ)
		yₖ = m(Wᵧ⁽ᵐ⁾,𝓨); Sₖ = S(Wᵧ⁽ᶜ⁾,𝓨,yₖ); Cₖ = C(Wᵧ⁽ᶜ⁾,𝓧,𝓨,x₀.μ,yₖ)

		μ[:,k+1]   = μ[:,k+1]   + K(Sₖ,Cₖ)*(y[:,k+1] - yₖ)
		Σ[:,:,k+1] = Σ[:,:,k+1] - K(Sₖ,Cₖ)*(Sₖ+R)*K(Sₖ,Cₖ)'

		Xᵤ = MvNormal(μ[:,k+1], Symmetric(Σ[:,:,k+1]))	# Filtering Distribution

		# Saves the filtering distribution Xᵤ ~ p(xₖ|y₁,⋯,yₖ) in the stack
		Xₑ = [Xₑ; Xᵤ]
	end
	# ====

	return (Xₑ, μ, Σ)
end

# (Y,T,X) = SIMULATE(SYS,Q,R)
function simulate(sys, u, t, x₀, Q, R)
	# Auxiliary variables
	(f,g,~,~,Δt,Nₓ,Nᵧ,Nᵤ) = sys

	x = zeros(Nₓ, length(t))
	y = zeros(Nᵧ, length(t))

	# Create the distributions for the random noise variables
	V = MvNormal(zeros(Nₓ), Q)
	Z = MvNormal(zeros(Nᵧ), R)

	# == SIMULATION LOOP ==
	for k ∈ 1:length(t)-1
		x[:,k+1] = f(x[:,k], u[:,k]) + rand(V)
		y[:,k]   = g(x[:,k]) + rand(Z)
	end

	y[:,end] = g(x[:,end]) + rand(Z)	# Last state emission
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
(xk, xk_μ, xk_Σ) = unscentedKalmanFilter(sys, y, u, t, p_x0, Q, R)

ti = length(t)
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

# anim = @animate for ti ∈ 1:length(t)
# 	scatter(y[1,1:ti], y[2,1:ti], m=(:star5, 2, stroke(0)), markeralpha=range(0,0.7,length=ti+1),
#                xlim=(min(y[1,:]...)-2, max(y[1,:]...)+2),
#                ylim=(min(y[2,:]...)-2, max(y[2,:]...)+2),
#                ticks=nothing, size=(16,9).*30, dpi=400)
#
# 	if(ti>1)
# 		plot!(xk_μ[1, 1:(ti-1)], xk_μ[2, 1:(ti-1)], alpha=range(0,0.7,length=ti+1))
# 	end
#
# 	scatter!([xk_μ[1,ti]], [xk_μ[2,ti]], alpha=0.25, marker=(arrow(u[1,ti], xk_μ[3,ti]), 20*u[1,ti], stroke(1, 0.1, :white)))
# 	scatter!([xk_μ[1,ti]], [xk_μ[2,ti]], alpha=0.80, marker=(circle(xk_μ[3,ti]), 15, :white))
# 	scatter!([xk_μ[1,ti]], [xk_μ[2,ti]], alpha=0.25, marker=(arrowRotation(u[2,ti], xk_μ[3,ti]), 25, stroke(1, 0.1, :white)))
# 	savefig("res/tmp/tmp_ekf$(ti).png")
# end
# gif(anim, "res/ekf_car.gif", fps=10)

# ===================
