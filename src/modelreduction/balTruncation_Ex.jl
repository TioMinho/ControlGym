# == Libraries ==
using LinearAlgebra, ControlSystems, Plots

# Configurations
theme(:dark)
pyplot(leg=false)

# Aliases
import Base: √
√(X::Array{Float64,2}) = Array{Float64,2}(sqrt.(X))

# == ==

# == Functions ==
# PLOTGRAMIANS([WW]...)
#	Displays a plot showing the ellipsoids corresponding
#	to the Gramian matrices given in [WW]
function plotGramians(T,WW...)
	# Unit-circle
	θ = 0:.01:2*π;
	x_c = cos.(θ); y_c = sin.(θ);
	
	p = plot(x_c, y_c, l=(1, :white))

	# Gramian ellipses
	CIRC = [x_c y_c];
	for (i,W) in enumerate(WW)
		if(i==length(WW))
			ELLIP = T^-1*√(W)*T*CIRC'
		else
			ELLIP = √(W)*CIRC'
		end

		plot!(ELLIP[1,:], ELLIP[2,:], l=(1), f=(0, 0.5))
	end
	
	# Displays the plot
	plot(p)
end
# == ==

# == Script ==
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
T = Tu*Σs

# Balanced Gramians
Wc_ = T^-1 * Wc * (T')^-1;
Wo_ = T' * Wo * T;

# == Visualization
plotGramians(T, [Wc, Wo, Wc_]...)

# == ==

