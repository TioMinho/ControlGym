# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats

# import Base: *
# *(v::Any, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
# *(v::Array{Float64,2}, Σ::PDMats.PDiagMat{Float64,Array{Float64,1}}) = v*(Σ*I(size(Σ,1)))
# *(v::Any, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))
# *(v::Array{Float64,2}, Σ::PDMat{Float64,Array{Float64,2}}) = v*(Σ*I(size(Σ,1)))

# ===================

# ==== Functions ====
"""	W = BayesRegression(X, t; ϕ=ϕ_poly(1), m₀=nothing, S₀=nothing)

The Bayesian Linear Regression method starts by assuming a zero-mean isotropic Gaussian
prior distribution for the weights,
	w⁽⁰⁾ ~ 𝓝(0, α⁻¹I)
and then updating the distribution using the Bayes' rule
	p(w⁽ᵏ⁾|t) = p(t|w⁽ᵏ⁻¹⁾)p(w⁽ᵏ⁻¹⁾) / p(t)
			  = 𝓝(mₖ, Sₖ)
with
	mₖ₊₁ = { βS₁Φ't				if k = 0		 (Sₖ₊₁)⁻¹ = { αI+βΦ'Φ		if k = 0
	 	   { Sₖ₊₁(Sₖ⁻¹mₖ+βΦ't) 	otherwise					{ Sₖ⁻¹+βΦ'Φ 	otherwise

"""
function BayesRegression(X, t; ϕ=ϕ_poly(1), m₀=nothing, S₀=nothing, α=1, β=1)
	# Initializes w0 ~ N(0,α⁻¹I) if they are not provided
	if(m₀ == nothing || S₀ == nothing)
		m₀ = zeros(size(ϕ(X[1,:]), 1))
		S₀ = (1/α)I(size(ϕ(X[1,:]), 1))
	end

	# Adds the prior to the Distribution iterations
	W = [MvNormal(m₀, S₀)]

	# == Bayesian Regression Updates ==
	for k = 1:size(X,1)
		Φ = ϕ(X[k,:])' # Computes the Design Matrix

		# Updates the Variance and Mean of distribution w⁽ᵏ⁾ ~ 𝓝(mₖ, Sₖ)
		Sₖ⁻¹ = inv(W[k].Σ) + β*Φ'Φ; 	Sₖ = inv(Sₖ⁻¹)
		mₖ = Sₖ*(inv(W[k].Σ)*W[k].μ + β*Φ't[k,:])

		# Adds the current update to the list of distributions
		W = [W; MvNormal(mₖ, Symmetric(Sₖ))]
	end
	# ==

	return W
end

# ===================
