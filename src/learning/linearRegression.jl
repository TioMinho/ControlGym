# == Libraries ==
using LinearAlgebra, Distributions, StatsBase, Random
using Plots

# Configurations
theme(:dark)
pyplot(leg=false)

# == ==

# == Functions ==
# The linear regression model
y(x, w, ϕ=ϕ_poly(1)) = w'ϕ(x)

# The basis functions
function ϕ_poly(M)
	return ϕ(x) = hcat( ones(size(x,1), 1), [x.^j for j in 1:M]... )'
end

function ϕ_gauss(μ, σ)
	return ϕ(x) = hcat( ones(size(x,1), 1), exp.(-(x.-μ).^2 ./ 2σ.^2) )'
end

function ϕ_sigm(μ, σ)
	return ϕ(x) = hcat( ones(size(x,1), 1), 1 ./ (1 .+ exp.(-(x-μ)./σ)) )'
end

# Analytical solution for w (using Maximum Likelihood Estimation)
function train(X, t, ϕ=ϕ_poly(1))
	# Creates the Design Matrix
	Φ = ϕ(X)'

	# Returns the Maximum Likelihood Estimator of w
	return pinv(Φ)*t
end

function train_bayes(X, t, ϕ=ϕ_poly(1), m0=zeros(size(X,2)+1), S0=ones(size(X,2)+1))
	# Adds the prior to the Distribution iterations
	w = [MvNormal(m0, S0)]
	β = 1 	# Noise parameter (known)

	# Uptade the mean and variance of the weight distribution
	for n = 1:size(X,1)
		Sn_inv = inv(w[n].Σ) + β*ϕ(X[n,:])*ϕ(X[n,:])'; Sn = (inv(Sn_inv))
		mn = Sn*(inv(w[n].Σ)*w[n].μ + β*ϕ(X[n,:])*t[n,:])

		w = [w; MvNormal(mn, Symmetric(Sn))]
	end

	return w
end
# == ==

# == Script ==
Random.seed!(11)
x = sample(-π:0.02:π, 35, replace=false)
t = sin.(x) + randn(size(x,1), 1)*0.2

# 1. Deterministic Linear Regression by MLE
# anim = @animate for i = 1:30
# 	xl = range(x[1], x[end], length=1000)

# 	scatter(x, t, m=(5, :lightblue, 0.75, stroke(0)))
# 	for j = 1:i-1
# 		ϕ = ϕ_poly(j); w = train(x,t,ϕ);
# 		plot!(xl, y(xl, w, ϕ)', l=(1, :pink, j/i*0.2))
# 	end

# 	ϕ = ϕ_poly(i); w = train(x,t,ϕ);
# 	plot!(xl, y(xl, w, ϕ)', l=(1, :pink))
# end
# gif(anim, "res/linreg_01.gif", fps=10)

# 2. Bayesian Linear Regression by Iterative Updating