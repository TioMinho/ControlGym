# == Libraries ==
using LinearAlgebra, ControlSystems, Plots, MAT

# Configurations
theme(:dark)
gr(leg=false, reuse=false)

# Aliases
import Base: √
√(X::Array{Float64,2}) = Array{Float64,2}(sqrt.(X))
√(X::Array{Float64,1}) = Array{Float64,1}(sqrt.(X))
∑(X) = sum(X)

# == ==

# == Functions ==
# == ==

# == Script ==
vars = matread("data/models/CDplayer.mat")

# 1. Computation of the Empirical Gramians -----------------
# Definition of the matrices and system
A = Array{Float64,2}(vars["A"]) # [-0.75 1; -0.3 -0.75] 
B = Array{Float64,2}(vars["B"]) # [2; 1] 
C = Array{Float64,2}(vars["C"]) # [1 2] 
D = zeros(size(C,1), size(B,2))

sys     = ss(A,B,C,D)
sys_adj = ss(A',C',B',D')	  	 # Adjoint System

Δt = 1;
Dsys     = c2d(sys,     Δt)[1] 
Dsys_adj = c2d(sys_adj, Δt)[1]  # Adjoint System

# Approximate the Gramians with   W_c ≈ C C^T   and   W_o ≈ O^T O 
# (Let us call C = U and O = V, in this case)
m = 100.;		# Window size of the data to be generated 
(~,~,Ux) = impulse(Dsys, 0:Δt:(2m)+1); 
(~,t,Vx) = impulse(Dsys_adj, 0:Δt:(2m)+1)

U = hcat([Ux[j,:,:]*√Δt for j in 2:length(t)]...)
V = hcat([Vx[j,:,:]*√Δt for j in 2:length(t)]...)'

EW_c = (U*U');	# (Notice that this a Cholesky factorization of EW_c)
EW_o = (V'V);	

# 2. Balanced Proper Orthogonal Decomposition (BPOD) ---------------------
# Computes the Hankel Matrix and its SVD Decomposition
H = V*U;

(K, Σs, ~) = svd(H); 
Σ  = diagm(Σs)

# Computes the Transformation Matrices 
T =  U*K*Σ^(-1/2);
S = V'*K*Σ^(-1/2); S = S';

# Visualizes the Hankel Singular Values
Σs = Σs[1:size(A,2)] 		# Selects only the non-zero Hankel Singular Values
bar(cumsum(Σs)/sum(Σs), l=(1, :white), f=(0, :white))

# # Partition the matrices T=[Ψ  Tt] and S=[Φ; St]
bd = bodeplot(sys, linecolor=:white, plotphase=false, label="Sys")
colors = [:orange, :green, :pink, :blue]
for (i, N_) in enumerate([1 2])
	(Ψ, Tt) = (T[:,1:N_], T[:,(N_+1):end]);
	(Φ, St) = (S[1:N_,:], S[(N_+1):end,:]);

	# Similarity Transformation
	A_ = Φ*A*Ψ; B_ = Φ*B; C_ = C*Ψ; D_ = D;

	sys_ = ss(A_,B_,C_,D_)

	bodeplot!(sys_, linecolor=colors[i], plotphase=false, label="BT$(N_)")
end

plot(bd)

# # == ==

