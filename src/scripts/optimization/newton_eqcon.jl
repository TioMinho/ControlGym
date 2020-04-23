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
"""	(X, λ) = newtonMethod_KKT(F, C, X0, η=1, ∇F=Nothing, ∇∇F=Nothing, ∇C=Nothing, ∇∇C=Nothing, ϵ=1e-6, T=1e3)

Given an objective function F: R^n -> R, equality contraints C: R^n -> R^m (m < n)
and initial point x0 ∈ R^n, solves the optimization problem
			min_(x)    F(x)
			s.t.	C(x) = 0
by optimizing the Lagrangian L: R^(n+m) -> R
	L(x,λ) = F(x) - λ'C(x) = F(x) - Σ(λ_i C_i(x)) .
The Lagragian is optimized by computing the recursive updates
	| x_(k+1) | = | x_k + η p |	    (x_0 = x0).
	| λ_(k+1) |	  |  λ_(k+1)  |
with p ∈ R^n and λ_(k+1) ∈ R^m being the solutions of the Karush-Kuhn-Tucker (KKT) system
	| ∇²_x L  ∇C' | |   -p    | = | ∇F |  .
	|   ∇C     0  | | λ_(k+1) |   | C  |

The optimization stops when |x_(k+1) - x_k| < ϵ or when k > T.

The outputs X = [x_0, ..., x_T] and λ = [λ_0, ..., λ_T] consists in the series of updates.
"""
function newtonMethod_KKT(F, c, x0; η=1, ∇F=Nothing, ∇∇F=Nothing, ∇c=Nothing, ∇∇c=Nothing, ϵ=1e-6, T=1e3, verbose=false)
	# Declare auxiliary variables for the optimization
	k = 0; n = length(x0); m = length(c(x0));
	X = x0; λ = randn(m)

	∇∇L(x,λ) = ∇∇F(x) .- (∇∇c(x))'λ

	# === Optimization Loop ===
	if(verbose); println("k\t|\tx_k\t|\tF(x)\t|\t||x_k - x_(k+1)||")
				 println("_________________________________________________________________")
				 println("0\t|\t$(x0)\t|\t$(F(x0))\t|\t---")
	end
	for k ∈ 1:T
		# Solves the Karush-Kuhn-Tucker (KKT) system
		KKT   = [∇∇L(X[:,end], λ[:,end]) -∇c(X[:,end])'
			          ∇c(X[:,end])             0      ];
		KKT = Matrix(KKT)

		KKT_s = KKT^(-1) * -[∇F(X[:,end]); c(X[:,end])]

		# Calculate the optimization step
		p_k = KKT_s[1:n];

		# Updates the parameter and lagrange multipliers values
		x_k = X[:,end] + η*p_k
		λ_k = KKT_s[(n+1):end]

		# Calculates the step size
		p_size = norm(p_k)

		# Checks for termination conditions
		if(p_size < ϵ || k > T)
			println("! OPTIMIZATION FINISHED !")
			break
		end

		# Prints the optimization step
		# if(verbose); println("$(@sprintf("%d",k)) | $(x_k) | $(F(x_k)) | $(p_size)")
		# end

		# Increases the parameters and lagrange multipliers vector
		X = [X x_k]; λ = [λ λ_k]

	end

	return (X,λ)
end

# ===================

# ==== Variables ====
F(x)  	= x[1]^2 + x[2]^2 + log(x[1]*x[2])
∇F(x)	= [ 2x[1]+x[1]^-1
			2x[2]+x[2]^-1]
∇∇F(x)	= [ 2-x[1]^-2  	 0
			  0		  2-x[2]^-2]

c(x)  = x[1]*x[2]-1
∇c(x) = [x[1]
		 x[2]]'
∇∇c(x) = I(2)

x0 = [0.5; 1]

# ===================

# ==== Script ====
(X,λ) = newtonMethod_KKT(F, c, x0, η=1, verbose=true,
							∇F=∇F, ∇∇F=∇∇F, ∇c=∇c, ∇∇c=∇∇c)

# 2. Plot the optimization steps
# Generates the error surface
(xx1, xx2) = meshgrid(0.01:0.1:5, 0.01:0.1:5)
xx = [[x1,x2] for (x1,x2) in zip(xx1[:], xx2[:])]

Fx = reshape(F.(xx), size(xx1))
cx = reshape(c.(xx), size(xx1))
FA = ([map(x->x[i], cx) for i in 1:length(cx[1])])

p = contour(xx1, xx2, Fx,
		xlim=(min(xx1...), max(xx1...)),
    	ylim=(min(xx2...), max(xx2...)),
    	size=(16,10).*60, dpi=200, grid=false)

for i in 1:length(FA)
	contour!(xx1, xx2, FA[i], levels=0, c=RGBA(0.2,0.8,0.2,1))
end

display(p)

plot!(X[1,:], X[2,:], l=(1, :blue), m=(:star5, :white, 6, stroke(0)))

# anim = @animate for ti ∈ 1:size(X,2)
# 	contour(xx1, xx2, Fx,
# 			xlim=(min(xx1...), max(xx1...)),
#         	ylim=(min(xx2...), max(xx2...)),
#         	size=(16,10).*30, dpi=200, grid=false)

# 	contourf!(xx1, xx2, cx, RGBA(1,0,0,0.3))

# 	plot!(X[1,1:ti], X[2,1:ti], l=(1, :blue), m=(:star5, :white, 6, stroke(0)))
# 	savefig("res/tmp/tmp_newtoneq_$(ti).png")
# end
# gif(anim, "res/newtoneq.gif", fps=10)

# ===================
