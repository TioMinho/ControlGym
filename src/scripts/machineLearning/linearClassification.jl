# == Libraries ==
using LinearAlgebra, Distributions, StatsBase, Random
using CSV, DataFrames
using Plots

folders = (@__DIR__).*["/LinearModels4Classification", "/utils"]
for folder in folders;	for file in readdir(folder)
	include(folder*"/"*file)
end; end

# Configurations
theme(:dark)
pyplot(leg=false)

# =====

# == Variables ==
# The Data and auxiliary variables
# data   = CSV.read((@__DIR__)*"/data/noisySin.csv")
# (x, t) = [data.x, data.t]
#
# Nₓ = size(x, 2); Nₜ = size(t, 2); M = size(x, 1)

# The choosen basis function
ϕ = ϕ_poly(3)

# =====

# == Script ==
# 1. Learns the weigths for the linear regression model
# wₒ = OLS(x, t, ϕ)						# MLE Approach
# W  = BayesRegression(data.x, data.t, ϕ)	# Bayesian approach

# 2. Plots the regression results
𝓒₁ = rand(MvNormal([-1; -1], 0.3I(2)), 100)
𝓒₀ = rand(MvNormal([1; 1], 0.45I(2)), 100)
allC = [𝓒₁ 𝓒₀];
data = DataFrame(x₁=allC[1,:], x₂=allC[2,:], t=[ones(100);zeros(100)])

plot_classification(data, 1, ϕ, method="bayes")
