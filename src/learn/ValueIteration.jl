# ==== Libraries ====
using LinearAlgebra

# ===================

# ==== Functions ====
"""	V = ValueIteration(T, R, doms; it=50)
"""
function ValueIteration(T, R, doms, γ=0.1; it=50)
	# Allocates the action-dependent (V) and optimal (Vˣ) matrices
	V = zeros(size(doms[1])..., size(doms[2])..., it)
	Vˣ = zeros(size(doms[1])..., it)

	for k in 2:it

		for sₖ in doms[1]
		for aₖ in doms[2]
			V[sₖ...,aₖ,k] = sum([T([sₖ...],aₖ,[sₖ₊₁...])*(R([sₖ...],aₖ,[sₖ₊₁...])+γ*Vˣ[sₖ₊₁...,k-1]) for sₖ₊₁ in doms[1]])
		end
			Vˣ[sₖ...,k] = max(V[sₖ...,:,k]...)
		end

	end

	return (V, Vˣ)
end

# ===================
