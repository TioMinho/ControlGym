# ==== Libraries ====
using LinearAlgebra

# ===================

# ==== Functions ====
"""	W = OLS(x, T; ϕ=ϕ_poly(1))

Solves the Ordinary Least-Squares (OLS) problem to obtain normal equations
    W = (XᵀX)⁻¹XᵀT
for the k-classes classification model yⱼ = wⱼ₀ + wⱼᵀx (j ∈ [1,k]) using
the matrices
    X = [ 1  x₁₁  ⋯  x₁ₙ ]		T = [ t₁₁  ⋯  t₁ₖ ]
        [ ⋮  ⋮  ⋱  ⋮  ]			 [  ⋮  ⋱   ⋮ ]
        [ 1  xₘ₁  ⋯  xₘₙ ]			[ t₁₁  ⋯  t₁ₙ ]
The OLS problem consists of minimizing the error function
    E(W) = 1/2 Tr{(XW-T)ᵀ(XW-T)}
"""
function OLS(x, T, ϕ=ϕ_poly(1))
	X = ϕ(x)'
	return pinv(X)*T
end

# ===================
