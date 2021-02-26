# ==== Libraries ====
using LinearAlgebra, StatsBase, Random, SparseArrays

# ===================

# ==== Functions ====
"""	(X,λ) = SQP_SOLVER(∇∇L, ∇F, A, x0; S0=Nothing, ϵ=1e-6, T=1e3, verbose=false)

Given a nonlinear objective function F: R^n -> R, nonlinear contraints C: R^n -> R^m,
and initial point x0 ∈ R^n, solves the optimization problem
			min_(x)	F(x)
			s.t.	C(x) = 0
by using a Sequential Quadratic Programming approach.
The current solution is optimized by minimizing a QP subproblem by considering a
quadratic approximation of the Lagrangian function and a linear approximation of
the contraints. The equivalent optimization is:
			min_(p)	L(p) ≈ (∇F_k)'p + 1/2 p'(∇²_x L_k)p
			s.t.	A_k p + c_k = 0
Each step updates the solution as
	| x_(k+1) | = | x_k + p |	(x_0 = x0).
	| λ_(k+1) |	  |	λ_(k+1) |
with p ∈ R^n and λ_(k+1) ∈ R^m, being the solutions of the equivalent
Karush-Kuhn-Tucker (KKT) system:
	| ∇²_x L_k   -A_k' | |   p     | = - | ∇F_k |  .
	|    A_k       0   | | λ_(k+1) |     |  c_k |

The outputs are X = [x_0, ..., x_T], and multipliers λ = [λ_0, ..., λ_T].
"""
function SQP_solver(∇∇L, ∇F, A, x0; S0=Nothing, ϵ=1e-6, T=1e3, verbose=false)
	return Nothing
end

# ===================