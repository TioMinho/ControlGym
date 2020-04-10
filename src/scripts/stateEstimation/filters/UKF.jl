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

# ===================
