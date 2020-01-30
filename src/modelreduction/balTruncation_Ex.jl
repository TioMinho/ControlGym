# == Libraries ==
using LinearAlgebra, ControlSystems, Plots

# Configurations
theme(:dark)
pyplot(leg=false, reuse=false)

# Aliases
import Base: √
√(X::Array{Float64,2}) = Array{Float64,2}(sqrt.(X))
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
		if(i==length(WW))
			ELLIP = T^-1*√(W)*T*CIRC'
		else
			ELLIP = √(W)*CIRC'
		end

		plot!(ELLIP[1,:], ELLIP[2,:], l=(1), f=(0, 0.75))
	end
	
	# Displays the plot
	display(p)
end
# == ==

# == Script ==
close("all")

# 1. Balanced Transformation -----------------
# Definition of the matrices and system
A = [-0.75  1.00; 
	 -0.30 -0.75]
B = [2; 1]
C = [1  2]
D = 0

sys = ss(A,B,C,D)

# Calculation of the gramians
Wc = gram(sys, :c)
Wo = gram(sys, :o)

# Computing the Balanced Model Transformation matrix
(Σ2, Tu) = eigen(Wc*Wo)

Σs = (Tu^-1*Wc*(Tu')^-1)^(1/4) * (Tu'*Wo*Tu)^(-1/4)
T = Tu*Σs; 
S = T^-1;

# Balanced Gramians
Wc_ = S * Wc * S';
Wo_ = T' * Wo * T;

# Visualization
if(size(Wc, 1) == 2)
	plotGramians(T, [Wc, Wo, Wc_]...)
end

# 2. Balanced Truncation ---------------------
# Permute the columns and visualizes the Hankel Singular Values
idx = sortperm(Σ2, rev=true); Σ2 = Σ2[idx]; T = T[:,idx]

HSV_c = cumsum(Σ2)/sum(Σ2)
bar(HSV_c, l=(1, :white), f=(0, :white))

# Partition the matrices T=[Ψ  Tt] and S=[Φ; St]
N_ = 1
(Ψ, Tt) = (T[:,1:N_], T[:,(N_+1):end]);
(Φ, St) = (S[1:N_,:], S[(N_+1):end,:]);

# Similarity Transformation
A_ = Φ*A*Ψ; B_ = Φ*B; C_ = C*Ψ; D_ = D;

sysb = ss(A_,B_,C_,D_)


# == ==

