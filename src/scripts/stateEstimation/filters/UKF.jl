# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats

import Base: *
*(v::Any, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
*(v::Any, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))
*(v::Array{Float64,2}, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))

# ===================

# ==== Functions ====
function UKF(sys, y, u, t, x₀, Q, R; α=1, κ=1, β=0)
# (Xₑ,μ,Σ) = UKF(SYS,Y,U,T,X₀,Q,R;α=1,κ=1,β=0)
#	Solves a state estimation problem using the Unscented Kalman Filter (UKF).
#	Consider the stochastic nonlinear discrete-time state-space system
#			xₖ₊₁ = f(xₖ,uₖ) + vₖ,		vₖ ~ 𝓝(0,Q)
#			yₖ   = g(xₖ)    + zₖ,		zₖ ~ 𝓝(0,R)
#	with prior distribution x₀ ~ 𝓝(μ₀,Σ₀).
#	The UKF approximates the filtering distribution xₖ ~ p(xₖ|y₀,⋯,yₖ) ≈ 𝓝(μₖ,Σₖ) by computing
#	a set of sigma-points (𝓧ₖ,𝓨ₖ) and then use the unscented transformation method to estimate
#	the mean and variance of this approximation.
#
	# Auxiliary variables
	(f,g,~,~,~,Δt,Nₓ,Nᵧ,Nᵤ) = sys
	t = t[1]:Δt:t[end]

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

	# == UKF FILTER ==
	# 1. UPDATES THE INITIAL STATE DISTRIBUTION
	(𝓧,𝓨) = uTransform(x₀, g, λᵧ)
	yₖ = m(Wᵧ⁽ᵐ⁾,𝓨); Sₖ = S(Wᵧ⁽ᶜ⁾,𝓨,yₖ); Cₖ = C(Wᵧ⁽ᶜ⁾,𝓧,𝓨,x₀.μ,yₖ)

	μₖ =   x₀.μ + K(Sₖ,Cₖ)*(y[:,1] - yₖ)
	Σₖ = I*x₀.Σ - K(Sₖ,Cₖ)*(Sₖ+R)*K(Sₖ,Cₖ)'

	# Creates the stack of filtering distributions Xᵤ ~ p(xₖ|y₁,⋯,yₖ)
	#  and saves the initial mean and covariance.
	Xₑ = [MvNormal(μₖ, Symmetric(Σₖ))];	μ[:,1] = μₖ; Σ[:,:,1] = Σₖ

	for k ∈ 1:length(t)-1
		# 2. PREDICTION STEP
		(~,𝓧) = uTransform(Xₑ[k], f, λₓ, u=u[:,k])

		μ⁻ₖ = m(Wₓ⁽ᵐ⁾, 𝓧)
		Σ⁻ₖ = S(Wₓ⁽ᶜ⁾, 𝓧, μ⁻ₖ) + Q

		Xₚ = MvNormal(μ⁻ₖ, Symmetric(Σ⁻ₖ))	# Predictive Distribution Xₚ ~ p(xₖ|y₁,⋯,yₖ₋₁)

		# 3. UPDATING STEP
		(~,𝓨) = uTransform(Xₚ, g, λᵧ)
		yₖ = m(Wᵧ⁽ᵐ⁾,𝓨); Sₖ = S(Wᵧ⁽ᶜ⁾,𝓨,yₖ); Cₖ = C(Wᵧ⁽ᶜ⁾,𝓧,𝓨,μ⁻ₖ,yₖ)

		μₖ = μ⁻ₖ + K(Sₖ,Cₖ)*(y[:,k+1] - yₖ)
		Σₖ = Σ⁻ₖ - K(Sₖ,Cₖ)*(Sₖ+R)*K(Sₖ,Cₖ)'

		Xᵤ = MvNormal(μₖ, Symmetric(Σₖ))	# Filtering Distribution Xᵤ ~ p(xₖ|y₁,⋯,yₖ)

		# Saves the filtering distribution Xᵤ ~ p(xₖ|y₁,⋯,yₖ)
		#  and saves the current mean and covariance.
		Xₑ = [Xₑ; Xᵤ]; μ[:,k+1] = μₖ; Σ[:,:,k+1] = Σₖ
	end

	# ====
	return (Xₑ, μ, Σ)
end

function uTransform(X, f, λ; u=nothing)
# (𝓧,𝓨) = UTRANSFORM(X,f,λ; u=nothing)
# 	Given a distribution X ~ 𝓝(μ,Σ) and a nonlinear function f:ℝⁿ×ℝᵖ → ℝᵐ computes the
#	set of sigma-points
#		𝓧 : {𝓧⁽⁰⁾, ..., 𝓧⁽²ⁿ⁺¹⁾} and
#		𝓧⁽⁰⁾ = μ, 𝓧⁽ᵏ⁾ = μ + √(n+λ)[√Σ]ₖ, 𝓧⁽ⁿ⁺ᵏ⁾ = μ - √(n+λ)[√Σ]ₖ	(k=1,⋯,n)
#	and applies the nonlinear function to obtain the transformed set
#		𝓨 : {f(𝓧⁽ᵏ⁾,uₖ)}		(k=1,⋯,n)
#
	# Unpack the mean and variance of X
	(μ, Σ) = [X.μ, I*X.Σ]; n = size(μ,1)

	# Construct the initial set of sigma-points
	𝓧 = [ [μ]
		[ [μ+(√(n+λ)*√Σ)[:,i]] for i in 1:n]...
		[ [μ-(√(n+λ)*√Σ)[:,i]] for i in 1:n]... ]

	# Propagates the sigma-points through the nonlinear transformations
	if u==nothing;	𝓨 = [f(𝓧⁽ᵏ⁾) 	  for 𝓧⁽ᵏ⁾ in 𝓧]
	else			𝓨 = [f(𝓧⁽ᵏ⁾, u) for 𝓧⁽ᵏ⁾ in 𝓧]
	end

	#
	return (𝓧,𝓨)
end

# ===================
