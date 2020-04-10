# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats

import Base: *
*(v::Any, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Any, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))

# ===================

# ==== Functions ====
function EKF(sys, y, u, t, x₀, Q, R)
# (Xₑ,μ,Σ) = EKF(SYS,Y,U,T,X₀,Q,R)
#	Solves a state estimation problem using the Extended Kalman Filter (EKF).
#	Consider the stochastic nonlinear discrete-time state-space system
#			xₖ₊₁ = f(xₖ,uₖ) + vₖ,		vₖ ~ 𝓝(0,Q)
#			yₖ   = g(xₖ)    + zₖ,		zₖ ~ 𝓝(0,R)
#	with prior distribution X₀ ~ 𝓝(μ₀,Σ₀).
#	The EKF approximates the filtering distribution xₖ ~ p(xₖ|y₀,⋯,yₖ) ≈ 𝓝(μₖ,Σₖ) by first
#	computing, for each time-step, a linearization of the functions f:ℝⁿ×ℝᵖ→ℝⁿ and g:ℝⁿ→ℝᵐ,
#			xₖ₊₁ = Aₖx + Bₖu + vₖ
#			yₖ   = Cₖx       + zₖ
#	with Aₖ = ∂f/∂x|₍ₓₖ,ᵤₖ₎, Bₖ = ∂f/∂u|₍ₓₖ,ᵤₖ₎ and Cₖ = ∂g/∂x|₍ₓₖ₎.
#	The approximation is then computed using using a Bayesian approach:
#		1) Compute the predictive distribution
#			Xₚ ~ p(xₖ|y₁,⋯,yₖ₋₁) ≈ 𝓝(μ⁻ₖ,Σ⁻ₖ) = 𝓝(Aₖxₖ₋₁+Bₖuₖ₋₁, Aₖ*Σₖ₋₁*Aₖᵀ + Q)
#		2) Use Bayes' rule to compute the filtering distribution
#			Xₚ ~ p(xₖ|y₁,⋯,yₖ)   ≈ 𝓝(μₖ,Σₖ)   = 𝓝(μ⁻ₖ+Kₖ(yₖ-Cₖμ⁻ₖ), Σ⁻ₖ+Kₖ(CₖΣ⁻ₖCₖᵀ+R)Kₖᵀ)
#		   with Kₖ = Σ⁻ₖ Cₖᵀ(CₖᵀΣ⁻ₖCₖᵀ+R)⁻¹, the optimal Kalman estimator.
#
	# Auxiliary variables
	(f,g,A,~,C,Δt,Nₓ,Nᵧ,Nᵤ) = sys
	t = t[1]:Δt:t[end]

	μ = zeros(Nₓ,   length(t))		# List of means 	(μ = [μ₀,⋯,μₜ])
	Σ = zeros(Nₓ,Nₓ,length(t))		# List of variances (Σ = [Σ₀,⋯,Σₜ])

	# Auxiliary functions
	K(S⁻ₖ,Cₖ) = S⁻ₖ * Cₖ'*(Cₖ*S⁻ₖ*Cₖ' + R)^(-1) 	# Optimal Kalman Gain

	# == EKF FILTER ==
	# 1. UPDATING THE INITIAL STATE DISTRIBUTION
	Cₖ = C(x₀.μ)					# Jacobian Cₖ = ∂g/∂x|₍ₓₖ₎

	μₖ =   x₀.μ + K(x₀.Σ,Cₖ)*(y[:,1] - Cₖ*x₀.μ)
	Σₖ = I*x₀.Σ - K(x₀.Σ,Cₖ)*(Cₖ*x₀.Σ*Cₖ' + R)*K(x₀.Σ,Cₖ)'

	# Creates the stack of filtering distributions Xᵤ ~ p(xₖ|y₁,⋯,yₖ)
	#  and saves the initial mean and covariance.
	Xₑ = [MvNormal(μₖ, Symmetric(Σₖ))];	μ[:,1] = μₖ; Σ[:,:,1] = Σₖ

	for k ∈ 1:length(t)-1
		# 2. PREDICTION STEP
		Aₖ = A(μₖ, u[:,k]);		# Jacobian Aₖ = ∂f/∂x|₍ₓₖ,ᵤₖ₎

		μ⁻ₖ = f(μₖ, u[:,k])
		Σ⁻ₖ = Aₖ*Σₖ*Aₖ' + Q

		Xₚ = MvNormal(μ⁻ₖ, Symmetric(Σ⁻ₖ))	# Predictive Distribution Xₚ ~ p(xₖ|y₁,⋯,yₖ₋₁)

		# 3. UPDATING STEP
		Cₖ = C(μ⁻ₖ)				# Jacobian Cₖ = ∂g/∂x|₍ₓₖ₎

		μₖ = μ⁻ₖ + K(Σ⁻ₖ,Cₖ)*(y[:,k+1] - g(μ⁻ₖ))
		Σₖ = Σ⁻ₖ - K(Σ⁻ₖ,Cₖ)*(Cₖ*Σ⁻ₖ*Cₖ' + R)*K(Σ⁻ₖ,Cₖ)'

		Xᵤ = MvNormal(μₖ, Symmetric(Σₖ))	# Filtering Distribution Xᵤ ~ p(xₖ|y₁,⋯,yₖ)

		# Saves the filtering distribution Xᵤ ~ p(xₖ|y₁,⋯,yₖ)
		#  and saves the current mean and covariance.
		Xₑ = [Xₑ; Xᵤ]; μ[:,k+1] = μₖ; Σ[:,:,k+1] = Σₖ
	end
	# ====

	return (Xₑ, μ, Σ)
end

# ===================
