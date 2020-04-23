# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats

# ===================

# ==== Functions ====
"""	W = BayesRegression(X, t, ϕ=ϕ_poly(1); m₀=nothing, S₀=nothing)

The Bayesian Linear Regression method starts by assuming a zero-mean isotropic Gaussian
prior distribution for the weights,
	w⁽⁰⁾ ~ 𝓝(0, α⁻¹I)
and then updating the distribution using the Bayes' rule
	p(w⁽ᵏ⁾|t) = p(t|w⁽ᵏ⁻¹⁾)p(w⁽ᵏ⁻¹⁾) / p(t)
			  = 𝓝(mₖ, Sₖ)
with
	mₖ₊₁ = { βS₁Φ't				if k = 0		 Sₖ₊₁ = { αI+βΦ'Φ			if k = 0
	 	   { Sₖ₊₁(Sₖ⁻¹mₖ+βΦ't) 	otherwise				{ Sₖ⁻¹+βΦ'Φ 		otherwise

"""
function BayesRegression(X, t, ϕ=ϕ_poly(1); m0=nothing, S0=nothing)
	# Initializes w0 ~ N(0,1) if they are not provided
	if(m0 == nothing || S0 == nothing)
		m0 = zeros(size(ϕ(X[1,:]), 1))
		S0 = ones(size(ϕ(X[1,:]), 1))
	end

	# Adds the prior to the Distribution iterations
	w = [MvNormal(m0, S0)]
	β = 1 	# Noise parameter (known)

	# Uptade the mean and variance of the weight distribution
	for n = 1:size(X,1)
		Sn_inv = inv(w[n].Σ) + β*ϕ(X[n,:])*ϕ(X[n,:])'; Sn = (inv(Sn_inv))
		mn = Sn*(inv(w[n].Σ)*w[n].μ + β*ϕ(X[n,:])*t[n,:])

		w = [w; MvNormal(mn, Symmetric(Sn))]
	end

	return w
end

# ===================
