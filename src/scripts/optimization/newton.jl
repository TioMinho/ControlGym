# ==== Libraries ====
using LinearAlgebra, StatsBase, Random, Plots
using Printf

# Configurations
theme(:dark)
pyplot(leg=false)

# Aliases
meshgrid(X,Y) = (first.(collect(Iterators.product(X, Y))), last.(collect(Iterators.product(X, Y))))

# ===================

# ==== Functions ====
# X = newtonMethod(F, X0, η=1, ∇F=Nothing, ∇∇F=Nothing, ϵ=1e-6, T=1e3) 
#	given an objective function F: R^n -> R and initial point x0 ∈ R^n,
#	solves the optimization problem
#				min_(x) F(x)
#	by computing the recursive updates
#		x_(k+1) = x_k + η (-∇²F(x_k))⁻¹ ∇F(x_k)	(x_0 = x0).
#	The optimization stops when |x_(k+1) - x_k| < ϵ or when k > T.
#
#	The output X = [x_0, ..., x_T] consists in the series of updates.
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

# ==== Variables ====
F(x)   = x'x
∇F(x)  = 2x
∇∇F(x) = 2

x0 = [-1.5; 1.5]

# ===================

# ==== Script ====
X = newtonMethod(F, x0, ∇F=∇F, ∇∇F=∇∇F, η=0.5, verbose=true)

# 2. Plot the optimization steps
# Generates the error surface
(xx1, xx2) = meshgrid(-2:0.1:2, -2:0.1:2)
xx = [[x1,x2] for (x1,x2) in zip(xx1[:], xx2[:])]
Fx = reshape(F.(xx), size(xx1))

anim = @animate for ti ∈ 1:size(X,2)
	contour(xx1, xx2, Fx,
			xlim=(min(xx1...), max(xx1...)), 
        	ylim=(min(xx2...), max(xx2...)),
        	size=(16,10).*30, dpi=200, grid=false)
	
	plot!(X[1,1:ti], X[2,1:ti], l=(1, :blue), m=(:star5, :white, 6, stroke(0)))
	savefig("res/tmp/tmp_newton_$(ti).png")
end
gif(anim, "res/newton.gif", fps=10)

# ===================
