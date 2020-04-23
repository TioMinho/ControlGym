# ==== Libraries ====
using LinearAlgebra

# ===================

# ==== Functions ====
"""	ϕ = ϕ_poly(M)

Creates a polynominal basis function of order M such that, for x ∈ ℝⁿ
	ϕ(x) ∈ ℝ⁽¹⁺ᴺᴹ⁾ = [ 1  x₁  (x₁)² ⋯ (x₁)ᴹ ⋯ xₙ  (xₙ)² ⋯ (xₙ)ᴹ ]ᵀ
"""
function ϕ_poly(M)
	return ϕ(x) = hcat( ones(size(x,1), 1), [x.^j for j in 1:M]... )'
end

"""	ϕ = ϕ_gaus(μ,σ)

Creates a gaussian basis function with mean μ ∈ ℝᵐ and variance σ such that, for x ∈ ℝⁿ
	ϕ(x) ∈ ℝ⁽¹⁺ᴺᴹ⁾ = [ 1  ϕ₁(x₁) ⋯ ϕ₁(xₙ) ⋯ ϕₘ(x₁) ⋯ ϕₘ(xₙ) ]ᵀ
from which
	ϕᵢ(x) = exp[(x-μᵢ)²/2σ]
"""
function ϕ_gauss(μ, σ)
	return ϕ(x) = hcat( ones(size(x,1), 1), exp.(-(x.-μ).^2 ./ 2σ.^2) )'
end

"""	ϕ = ϕ_sigm(μ, σ)

Creates a sigmoid basis function with position μ ∈ ℝᵐ and scale σ such that, for x ∈ ℝⁿ
	ϕ(x) ∈ ℝ⁽¹⁺ᴺᴹ⁾ = [ 1  σ((x₁-μ₁)/s) ⋯ σ((xₙ-μ₁)/s) ⋯ σ((x₁-μₘ)/s) ⋯ σ((xₙ-μₘ)/s) ]ᵀ
from which
	σ(a) = 1/(1+exp[-a])
"""
function ϕ_sigm(μ, σ)
	return ϕ(x) = hcat( ones(size(x,1), 1), 1 ./ (1 .+ exp.(-(x-μ)./σ)) )'
end
