# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats

import Base: *
*(v::Any, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Any, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))

# ===================

# ==== Functions ====
"""	(Xₑ,μ,Σ) = KF(SYS,Y,U,T,X₀,Q,R)

Solves a state estimation problem using the Kalman Filter (KF).
Consider the stochastic linear discrete-time state-space system
		xₖ₊₁ = Axₖ + Buₖ + vₖ,		vₖ ~ 𝓝(0,Q)
		yₖ   = Cxₖ       + zₖ,		zₖ ~ 𝓝(0,R)
with prior distribution x₀ ~ 𝓝(μ₀,Σ₀).
The KF exactly solves the filtering distribution xₖ ~ p(xₖ|y₀,⋯,yₖ) = 𝓝(μₖ,Σₖ)
using a Bayesian approach:
	1) Compute the predictive distribution
		Xₚ ~ p(xₖ|y₁,⋯,yₖ₋₁) = 𝓝(μ⁻ₖ,Σ⁻ₖ) = 𝓝(Axₖ₋₁+Buₖ₋₁, A*Σₖ₋₁*Aᵀ + Q)
	2) Use Bayes' rule to compute the filtering distribution
		Xₚ ~ p(xₖ|y₁,⋯,yₖ)   = 𝓝(μₖ,Σₖ)   = 𝓝(μ⁻ₖ+Kₖ(yₖ-Cμ⁻ₖ), Σ⁻ₖ+Kₖ(CΣ⁻ₖCᵀ+R)Kₖᵀ)
	   with Kₖ = Σ⁻ₖ Cᵀ(CᵀΣ⁻ₖCᵀ+R)⁻¹, the optimal Kalman estimator.
"""
function KF(sys, y, u, t, x₀, Q, R)
	# Auxiliary variables
	(~,~,A,B,C,Δt,Nₓ,Nᵧ,Nᵤ) = sys
	t = t[1]:Δt:t[end]

	μ = zeros(Nₓ,     length(t))
	Σ = zeros(Nₓ, Nₓ, length(t))

	# Auxiliary functions
	K(S⁻ₖ) = S⁻ₖ * C'*(C*S⁻ₖ*C' + R)^(-1)	# Optimal Kalman Gain

	# == KALMAN FILTER ==
	# 1. UPDATING THE INITIAL STATE DISTRIBUTION
	μₖ =   x₀.μ + K(x₀.Σ)*(y[:,1] - C*x₀.μ)
	Σₖ = I*x₀.Σ - K(x₀.Σ)*(C*x₀.Σ*C'+R)*K(x₀.Σ)'

	# Creates the stack of filtering distributions Xᵤ ~ p(xₖ|y₁,⋯,yₖ)
	#  and saves the initial mean and covariance.
	Xₑ = [MvNormal(μₖ, Symmetric(Σₖ))];	μ[:,1] = μₖ; Σ[:,:,1] = Σₖ

	for k ∈ 1:length(t)-1
		# 2. PREDICTION STEP
		μ⁻ₖ = A*μₖ + B*u[:,k]
		Σ⁻ₖ = A*Σₖ*A' + Q

		Xₚ = MvNormal(μ⁻ₖ, Symmetric(Σ⁻ₖ))	# Predictive Distribution Xₚ ~ p(xₖ|y₁,⋯,yₖ₋₁)

		# 3. UPDATING STEP
		μₖ = μ⁻ₖ + K(Σ⁻ₖ)*(y[:,k+1] - C*μ⁻ₖ)
		Σₖ = Σ⁻ₖ - K(Σ⁻ₖ)*(C*Σ⁻ₖ*C'+R)*K(Σ⁻ₖ)'

		Xᵤ = MvNormal(μₖ, Symmetric(Σₖ))	# Filtering Distribution Xᵤ ~ p(xₖ|y₁,⋯,yₖ)

		# Saves the filtering distribution Xᵤ ~ p(xₖ|y₁,⋯,yₖ)
		#  and saves the current mean and covariance.
		Xₑ = [Xₑ; Xᵤ]; μ[:,k+1] = μₖ; Σ[:,:,k+1] = Σₖ
	end
	# ====

	return (Xₑ, μ, Σ)
end

# ===================
