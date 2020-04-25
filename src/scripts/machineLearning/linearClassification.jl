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
data   = CSV.read((@__DIR__)*"/data/3classLinear.csv")
Nₓ = size(data, 2)-1; Nₖ = length(unique(data.t)); M = size(data, 1)

(x₁, x₂, t) = [data.x₁, data.x₂, data.t]
X = [x₁ x₂]; (T, tᵤ) = oneHotEncoding(t)

# The choosen basis function
ϕ = ϕ_poly(1)

# =====

# == Script ==
# 1. Learns the weigths for the linear classification models
Wₒ = OLS(X, T, ϕ)						# Least-Squares Approach

# 2. Plots the regression results
 plot_classification(data, Wₒ, ϕ, method="OLS")
