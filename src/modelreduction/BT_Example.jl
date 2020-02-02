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
# PLOTGRAMIANS([WW]...)
#	Displays a plot showing the ellipsoids corresponding
#	to the Gramian matrices given in [WW]
function plotGramians(T,WW...)
	# Unit-circle
	θ = 0:.01:2*π;
	x_c = cos.(θ); y_c = sin.(θ);
	
	p = plot(x_c, y_c, l=(0.5, :white, :dash))

	# Gramian ellipses
	CIRC = [x_c y_c];
	for (i,W) in enumerate(WW)
		if(i==length(WW)); 	ELLIP = T^-1*√(W)*T*CIRC'
		else; 				ELLIP = √(W)*CIRC'
		end

		plot!(ELLIP[1,:], ELLIP[2,:], l=(1), f=(0, 0.75))
	end
	
	# Displays the plot
	display(p)
end
# == ==

# == Script ==
vars = matread("data/models/CDplayer.mat")

# 1. Balanced Transformation -----------------
# Definition of the matrices and system
A = Array{Float64,2}(vars["A"])
B = Array{Float64,2}(vars["B"])
C = Array{Float64,2}(vars["C"])
D = zeros(size(C,1), size(B,2))

sys = ss(A,B,C,D)

# Calculation of the gramians
Wc = gram(sys, :c)
Wo = gram(sys, :o)

# Computing the Balanced Model Transformation matrix
U = cholesky(Hermitian(Wc)).U';
(K, Σ2, Kt) = svd(U'*Wo*U);
Σ = diagm(√(Σ2));

T = U * K * Σ^(-0.5);
S = Σ^0.5 * K' * U^-1;

# Balanced Gramians
Wc_ = S * Wc * S';
Wo_ = T' * Wo * T;

# Visualization
if(size(Wc, 1) == 2)
	plotGramians(T, [Wc, Wo, Wc_]...)
end

# 2. Balanced Truncation ---------------------
# Visualizes the Hankel Singular Values
bar(cumsum(Σ2)/sum(Σ2), l=(1, :white), f=(0, :white))

# Partition the matrices T=[Ψ  Tt] and S=[Φ; St]
bd = bodeplot(sys, linecolor=:white, plotphase=false, label="Sys")
colors = [:orange, :green, :pink, :blue]
for (i, N_) in enumerate([5 10 40])
	(Ψ, Tt) = (T[:,1:N_], T[:,(N_+1):end]);
	(Φ, St) = (S[1:N_,:], S[(N_+1):end,:]);

	# Similarity Transformation
	A_ = Φ*A*Ψ; B_ = Φ*B; C_ = C*Ψ; D_ = D;

	sys_ = ss(A_,B_,C_,D_)

	bodeplot!(sys_, linecolor=colors[i], plotphase=false, label="BT$(N_)")
end

plot(bd, xlim=[1e0, 1e5])

# == ==

