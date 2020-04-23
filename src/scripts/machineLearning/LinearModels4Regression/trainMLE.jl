# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats

# ===================

# ==== Functions ====
"""	w = OLS(X, t, ϕ=ϕ_poly(1))

Solves the Ordinary Least-Squares (OLS) problem to obtain normal equations
    w = (ΦᵀΦ)⁻¹Φᵀt
using the Design Matrix
    Φ = [ ϕ₀(x₁)  ϕ₁(x₁)  ⋯  ϕₘ₋₁(x₁) ]
        [   ⋮      ⋮    ⋱      ⋮   ]
        [ ϕ₀(xₙ)  ϕ₁(xₙ)  ⋯  ϕₘ₋₁(xₙ) ]
The OLS problem consists of minimizing the error function
    E(w) = 1/2 Σ(tₖ - wᵀϕ(xₖ))²
arising from the maximum likelihood approach for the regression problem.
"""
function OLS(X, t, ϕ=ϕ_poly(1))
    return pinv(ϕ(X))*t     # Φ = ϕ(X) -> The Design Matrix
end

"""	w = ROLS(X, t, λ, ϕ=ϕ_poly(1))

Solves the Regularized Ordinary Least-Squares (OLS) problem to obtain normal equations
    w = (λI + ΦᵀΦ)⁻¹Φᵀt
using the Design Matrix
    Φ = [ ϕ₀(x₁)  ϕ₁(x₁)  ⋯  ϕₘ₋₁(x₁) ]
        [   ⋮      ⋮    ⋱      ⋮   ]
        [ ϕ₀(xₙ)  ϕ₁(xₙ)  ⋯  ϕₘ₋₁(xₙ) ]
The OLS problem consists of minimizing the L2 regularized error function
    E(w) = (1/2) Σ(tₖ - wᵀϕ(xₖ))² + (λ/2) wᵀw
arising from the maximum likelihood approach for the regression problem.
"""
function ROLS(X, t, λ, ϕ=ϕ_poly(1))
	Φ = ϕ(X)						# Stores the Design Matrix
    return inv(λI + Φ'Φ)*Φ't
end


"""	W = LMS(X, t, ϕ=ϕ_poly(1))

Solves the Least-Mean-Squares (LMS) problem by sequential stochastic gradient descent
    w⁽ᵏ⁺¹⁾  = w⁽ᵏ⁾ - η∇Eₙ
            = w⁽ᵏ⁾ + η(tₙ - [w⁽ᵏ⁾]ᵀϕ(xₙ))ϕ(xₙ)
which iteratively minimizes the error function
    E(w) = 1/2 Σ(tₙ - wᵀϕ(xₙ))²
arising from the maximum likelihood approach for the regression problem.
The function returns the entire history of weights W = [w⁽⁰⁾ ⋯ w⁽ᴷ⁾]
"""
function LMS(X, t, ϕ=ϕ_poly(1); η=0.1, ϵ=1e-6, T=1e3, verbose=false)
	# Declare auxiliary variables for the optimization
	Nₓ = size(ϕ(X),1); Nₘ = size(t,1)
	k = 0; W = randn(Nₓ,1);

	# === Optimization Loop ===
	if(verbose); println("k\t|\tw_k\t|\tError\t|\t||w_k - w_(k+1)||")
				 println("_________________________________________________________________")
				 println("0\t|\t$(w0)\t|\t$(F(x0))\t|\t---")
	end
	for k ∈ 1:T
		# Chooses a random datapoint
		n = rand(1:Nₘ)
		xₙ = X[:,n]; tₙ = t[n]

		# Updates the parameters
		w_k = w[:,end] + η*(tₙ - w[:,end]'ϕ(xₙ))*ϕ(xₙ)
		#-

		# Calculates the step size
		p_size = norm(x_k - X[:,end])

		# Prints the optimization step
		if(verbose); println("$(@sprintf("%d",k))\t|\t$(x_k)\t|\t$(F(x_k))\t|\t$(p_size)")
		end

		# Checks for termination conditions
		if(p_size < ϵ || k > T)
			println("! OPTIMIZATION FINISHED !")
			break
		end

		# Increases the parameters vector
		W = [W w_k]

	end

	return W
end

# ===================
