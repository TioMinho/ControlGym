# ==== Libraries ====
using LinearAlgebra, StatsBase, Random

# ===================

# ==== Functions ====
"""	X = newtonMethod(F, X0, η=1, ∇F=Nothing, ∇∇F=Nothing, ϵ=1e-6, T=1e3)

	Given an objective function F: R^n -> R and initial point x0 ∈ R^n,
	solves the optimization problem
				min_(x) F(x)
	by computing the recursive updates
		x_(k+1) = x_k + η (-∇²F(x_k))⁻¹ ∇F(x_k)	(x_0 = x0).
	The optimization stops when |x_(k+1) - x_k| < ϵ or when k > T.

	The output X = [x_0, ..., x_T] consists in the series of updates.
"""
function newtonMethod(F, x0; η=1, ∇F=Nothing, ∇∇F=Nothing, ϵ=1e-6, T=1e3, verbose=false)
	# Declare auxiliary variables for the optimization
	k = 0;
	X = x0;

	# === Optimization Loop ===
	if(verbose); println("k\t|\tx_k\t|\tF(x)\t|\t||x_k - x_(k+1)||")
				 println("_________________________________________________________________")
				 println("0\t|\t$(x0)\t|\t$(F(x0))\t|\t---")
	end
	for k ∈ 1:T
		# Updates the parameters
		x_k = X[:,end] - η * ∇∇F(X[:,end])^(-1) * ∇F(X[:,end])

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
		X = [X x_k]

	end

	return X
end

# ===================