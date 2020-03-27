# ==== Libraries ====
using LinearAlgebra, StatsBase, Random, Plots, SparseArrays
using Printf

# Configurations
theme(:dark)
pyplot(leg=false)

# Aliases
meshgrid(X,Y) = (first.(collect(Iterators.product(X, Y))), last.(collect(Iterators.product(X, Y))))

# ===================

# ==== Functions ====
function SQP_solver(G, H, ∇C, C_LU, X_LU, x0;S0=Nothing, ϵ=1e-6, T=1e3, verbose=false)
# (X,λ,ν) = SQP_SOLVER(G, H, C_LU, X_LU, x0;S0=Nothing, ϵ=1e-6, T=1e3, verbose=false) 
#	given a nonlinear objective function F: R^n -> R, nonlinear contraints C: R^n -> R^m, 
#	and initial point x0 ∈ R^n, solves the optimization problem
#				min_(x)		  F(x) 
#				s.t.	C_L ≤ C(x) ≤ C_U
#						x_L ≤   x  ≤ x_U
#	by using a Sequential Quadratic Programming approach.
#	The current solution is optimized by minimizing a QP subproblem by considering a 
#	quadratic approximation of the Lagrangian function and a linear approximation of
#	the contraints. The equivalent optimization is:
#				min_(p)	L(p) ≈ G'p + 1/2 p'Hp
#				s.t.	C_L - c(x) ≤ ∇C*p ≤ C_U - c(x)
#						   x_L - x ≤  p  ≤ x_U - x
#	Each step updates the solution as
#		| x_(k+1) |   | x_k |	  |   p  |   (x_0 = x0).
#		| λ_(k+1) |	= | λ_k | + η | λ'-λ |
#		| ν_(k+1) |	  | ν_k |     | ν'-ν |
#		| s_(k+1) |	  | s_k |     | s'-s |
#	with p,ν' ∈ R^n and λ',s' ∈ R^m, being the solutions of the equivalent
#	QP Karush-Kuhn-Tucker (KKT) system.
#
#	The outputs are X = [x_0, ..., x_T], and multipliers λ = [λ_0, ..., λ_T] 
#	and ν = [ν_0, ..., ν_T]. 
	return Nothing

end	
# ===================

# ==== Variables ====
F(x)  	= x[1]^2 + x[2]^2 + log(x[1]*x[2])
∇F(x)	= [ 2x[1]+x[1]^-1 
			2x[2]+x[2]^-1]
∇∇F(x)	= [ 2-x[1]^-2  	 0	   
			  0		  2-x[2]^-2]

c(x)  = x[1]*x[2]
∇c(x) = [x[1]
		 x[2]]
c_LU = [1 Inf]
x_LU = [0 10
		0 10]

x0 = [0.5; 2]

# ===================

# ==== Script ====
# (X,S,γ,λ) = QP_solver(∇F, ∇∇F, [A a], [B b], x0, [2], verbose=true)

# 2. Plot the optimization steps
# Generates the error surface
(xx1, xx2) = meshgrid(0.01:0.01:2.5, 0.01:0.01:2.5)
xx = [[x1,x2] for (x1,x2) in zip(xx1[:], xx2[:])]

Fx = reshape(F.(xx), size(xx1))
if(size(c(x0),2)>0)
	  F0 = ([reshape(c.(xx), size(xx1)) for i in 1:size(c(x0),1)])
else; F0 = []
end

p = contour(xx1, xx2, Fx,
		xlim=(min(xx1...), max(xx1...)), 
    	ylim=(min(xx2...), max(xx2...)),
    	size=(16,10).*50, dpi=200, grid=false)

for i in 1:length(F0)
	contourf!(xx1, xx2, F0[i].-c_LU[i,1], levels=[0 1], c=[RGBA(0,0,0,0.6), RGBA(0,0,0,0)])
	contour!( xx1, xx2, F0[i].-c_LU[i,1], levels=0, c=RGBA(0.4,0.4,0.4,1))

	contourf!(xx1, xx2, c_LU[i,2].-F0[i], levels=[0 1], c=[RGBA(0,0,0,0.6), RGBA(0,0,0,0)])
	contour!( xx1, xx2, c_LU[i,2].-F0[i], levels=0, c=RGBA(0.4,0.4,0.4,1))
end

display(p)

# plot!(X[1,:], X[2,:], l=(1, :blue), m=(:star5, :white, 6, stroke(0)))

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
