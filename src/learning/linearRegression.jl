# == Libraries ==
using LinearAlgebra, StatsBase, Random
using Plots

# == Functions ==
# The linear regression model
y(x, w, ϕ=ϕ_poly(1)) = w'ϕ(x)

# The basis functions
function ϕ_poly(M)
	return ϕ(x) = hcat( ones(size(x,1), 1), [x.^j for j in 1:M]... )'
end

function ϕ_gauss(μ, σ)
	return ϕ(x) = hcat( ones(size(x,2), 1), exp.(-(x.-μ).^2 ./ 2σ.^2) )
end

function ϕ_sigm(μ, σ)
	return ϕ(x) = hcat( ones(size(x,2), 1), 1 ./ (1 .+ exp.(-(x-μ)./σ)) )
end

# Analytical solution for w (using Maximum Likelihood Estimation)
function train(X, t, ϕ=ϕ_poly(1))
	# Creates the Design Matrix
	Φ = ϕ(X)'

	# Returns the Maximum Likelihood Estimator of w
	return pinv(Φ)*t
end

# == Script ==
Random.seed!(11)
x = sample(-π:0.02:π, 35, replace=false)
t = sin.(x) + randn(size(x,1), 1)*0.2

pyplot(leg=false)
p = plot(x, t, t=:scatter)
for i in 1:50
	plot!(x, y(x, train(x, t, ϕ_poly(i)), ϕ_poly(i))', t=:line)
end

display(p)