# ==== Libraries ====
using LinearAlgebra, StatsBase, Random, SparseArrays

# ===================

# ==== Functions ====
"""	(X,S,γ,λ) = QP_SOLVER(G, H, Aa, Bb, X0, S0; ϵ=1e-6, T=1e3)

Given a quadratic objective function F: R^n -> R, equality contraints A: R^n -> R^m,
inequality contraints B: R^n -> R^p, initial point x0 ∈ R^n and initial active set S0 ⊆ [1,p],
solves the optimization problem
			min_(x)	F(x) = G'x + 1/2 x'Hx
			s.t.	Ax = a
					Bx ≥ b
by using an Active Set Method.
The current solution is optimized by computing the recursive updates
	| x_(k+1) | = | x_k + η p |	    (x_0 = x0).
	| γ_(k+1) |	  |  γ_(k+1)  |
	| λ_(k+1) |	  |  λ_(k+1)  |
with p ∈ R^n, γ_(k+1) ∈ R^m and λ_(k+1) ∈ R^(|S|) being the solutions of the Karush-Kuhn-Tucker (KKT) system
	| ∇²_x F   A'  _B' | |   -p    | = | ∇F |  .
	|    A     0    0  | | γ_(k+1) |   | a  |
	|   _B     0    0  | | λ_(k+1) |   | b  |
where _B = [Bi] for i ∈ S_k, the current active set.

The outputs are X = [x_0, ..., x_T], final active set S_T, multipliers λ = [λ_0, ..., λ_T]
and γ = [γ_0, ..., γ_T].
"""
function QP_solver(G, H, A_a, B_b, x0, S0; ϵ=1e-6, T=1e3, verbose=false)
	# Declare auxiliary variables for the optimization
	k = 0; n = length(x0); (m,mx) = size(A_a); (p,px) = size(B_b)

	if(mx > 0); A = A_a[:,1:n]; a = A_a[:,end]
				γ_k = rand(m)
	else;       A = Array{Float64,2}(reshape([],0,n)); a = []
				γ_k = 0; m = 0;
	end

	if(px > 0); B = B_b[:,1:n]; b = B_b[:,end]
				λ_k = rand(p)
	else;       B = Array{Float64,2}(reshape([],0,n)); b = []'
				λ_k = 0; p = 0;
	end

	X = x0; S = [S0]; γ = [γ_k]; λ = [λ_k]

	# === Optimization Loop ===
	if(verbose); println("k\t|\tx_k\t|\tF(x)\t|\t| S")
				 println("_________________________________________________________________")
				 println("0\t|\t$(x0)\t|\t$(F(x0))\t|\t$(S[end])")
	end
	for k ∈ 1:T
		# Gets the active inequality constraints
		(_B,_b,_B!,_b!) = activeContraints(B,b,S[end])
		_p  = length(S[end])

		# Solves the Karush-Kuhn-Tucker (KKT) system
		KKT   = [	H		[A' _B']
			     [A; _B] 	0*I(m+_p)]
		KKT = Matrix(KKT)

		KKT_s = KKT^-1 * 1*[ G.*X[:,end]
				  			  A*X[:,end] .-  a
				 			 _B*X[:,end] .- _b]

		# Calculate the optimization step
		p_k = -KKT_s[1:n];

		# Updates the parameter and lagrange multipliers values
		x_k = X[:,end] + p_k
		γ_k = 1*KKT_s[(n+1):(n+m)]
		λ_k = 1*KKT_s[(n+m+1):end]

		# Changes the active set and (possibly) takes a restricted step
		terminate = false
		if(sum(B*x_k .< b) > 0)
			S_k = union(S[end], setdiff(findall(B*x_k .< b), S[end]))

			η_aux = -(_B!*X[:,end].-_b!)./(p_k'_B!')'
			η = min(η_aux[findall(η_aux .> 0)]...)
			x_k = X[:,end] + η*p_k
		elseif length(λ_k) > 0 && sum(λ_k .< 0) > 0
			S_k = S[end][setdiff(1:end, argmin(λ_k))]
		else
			S_k = S[end]
			terminate = true
			println("! OPTIMIZATION FINISHED !")
		end

		# Prints the optimization step
		if(verbose); println("$(@sprintf("%d",k)) | $(x_k) | $(F(x_k)) | $(S_k)")
		end

		# Increases the parameters, active set and lagrange multipliers vector
		X = [X x_k]; S = [S [S_k]]; γ = [γ [γ_k]];  λ = [λ [λ_k]]
		if(terminate); break; end
	end

	return (X,S,γ,λ)
end

"""	(B,b, B!,b!) = activeContraints(B, b, S)

Given linear equality contraints A: R^n -> R^m, inequality contraints
B: R^n -> R^p and a set of index for active contraints, separate the
function into the active contraints [B, b] and inactive contraints [B!, b!]
"""
function activeContraints(B, b, S)
	# Auxiliary variables
	n = size(B,2)

	# Separate the contraints in the linear map
	if(length(S) > 0)
		_B  = B[S,:]; 				  _b  = b[S]
		_B! = B[setdiff(1:end, S),:]; _b! = b[setdiff(1:end, S)]
	else
		_B  = Array{Float64,2}(reshape([],0,n)); _b  = []
		_B! = B; 			   					 _b! = b
	end

	return (_B,_b, _B!, _b!)
end

# ===================