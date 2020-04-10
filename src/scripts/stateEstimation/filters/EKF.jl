# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats
using ControlSystems, Plots

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

# ===================
