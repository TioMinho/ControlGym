# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats
using ControlSystems

# ===================

# ==== Functions ====
# (Y,T,X) = SIMULATE(SYS,Q,R)
function sim(sys, u, t, x₀, Q, R)
	# Auxiliary variables
	(f,g,~,~,~,Δt,Nₓ,Nᵧ,Nᵤ) = sys

	x = zeros(Nₓ, length(t))
	y = zeros(Nᵧ, length(t))

	# Create the distributions for the random noise variables
	V = MvNormal(zeros(Nₓ), Q)
	Z = MvNormal(zeros(Nᵧ), R)

	# == SIMULATION LOOP ==
	for k ∈ 1:length(t)-1
		x[:,k+1] = f(x[:,k], u[:,k]) + rand(V)
		y[:,k]   = g(x[:,k]) + rand(Z)
	end

	y[:,end] = g(x[:,end]) + rand(Z)	# Last state emission
	# ====

	return (y,t,x)
end
