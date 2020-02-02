# == Libraries ==
using LinearAlgebra, ControlSystems, Plots, MAT

# Configurations
theme(:dark)
gr(leg=false, reuse=false)

# Aliases
import Base: √
√(X::Array{Float64,2}) = Array{Float64,2}( sqrt.( X .* (abs.(X).>100eps()) ) )
√(X::Array{Float64,1}) = Array{Float64,1}( sqrt.( X .* (abs.(X).>100eps()) ) )
∑(X) = sum(X)

# == ==

# == Functions ==
# == ==

# == Script ==
vars = matread("data/models/CDplayer.mat")

# 1. Computation of the Empirical Gramians -----------------
# Definition of the matrices and system
A = Array(vars["A"])
B = Array(vars["B"])
C = Array(vars["C"])
D = zeros(size(C,1), size(B,2))

# Continuous-time system
sys     = ss(A,B,C,D)
sys_adj = ss(A',C',B',D')   # Adjoint System

# Discrete-time system
Δt = 0.01;
Dsys     = c2d(sys,     Δt)[1] 
Dsys_adj = ss(Dsys.A',Dsys.C',Dsys.B',Dsys.D', Δt)   # Adjoint System

# Approximate the Gramians with   W_c ≈ C C^T   and   W_o ≈ O^T O 
# (Let us call C = U and O = V, in this case)
m = 10.;		# Window size of the data to be generated 
(~,~,Ux) = impulse(Dsys,     0:Δt:(5m)+1); 
(~,t,Vx) = impulse(Dsys_adj, 0:Δt:(5m)+1); Vx = Vx*Δt;

U = hcat([Ux[j,:,:]*√Δt for j in 2:length(t)]...)
V = hcat([Vx[j,:,:]*√Δt for j in 2:length(t)]...)'

EW_c = (U*U');	# (Notice that this a Cholesky factorization of EW_c)
EW_o = (V'V);	

# 2. Balanced Proper Orthogonal Decomposition (BPOD) ---------------------
# Computes the Hankel Matrix and its SVD Decomposition
H = V*U;

(K, Σs, Kt) = svd(H); 
Σ  = diagm(Σs)

# Computes the Transformation Matrices 
T = U*Kt*Σ^(-1/2);
S = V'*K*Σ^(-1/2); S = S';

# Visualizes the Hankel Singular Values
Σs = Σs[1:size(A,2)] 		# Selects only the non-zero Hankel Singular Values
bar(cumsum(Σs)/sum(Σs), l=(1, :white), f=(0, :white))

# # Partition the matrices T=[Ψ  Tt] and S=[Φ; St]
bd = bodeplot(sys, linecolor=:white, plotphase=false, label="Sys")
colors = [:orange, :green, :pink, :blue]
for (i, N_) in enumerate([1 5 20 40])
	(Ψ, Tt) = (T[:,1:N_], T[:,(N_+1):end]);
	(Φ, St) = (S[1:N_,:], S[(N_+1):end,:]);

	# Similarity Transformation
	A_ = Φ*Dsys.A*Ψ; B_ = Φ*Dsys.B; C_ = Dsys.C*Ψ; D_ = Dsys.D;

	Dsys_ = ss(A_,B_,C_,D_,Δt)

	bodeplot!(Dsys_, linecolor=colors[i], plotphase=false, label="PBOD$(N_)")
end

plot(bd, xlim=[1e0, 1e5])

# # == ==

