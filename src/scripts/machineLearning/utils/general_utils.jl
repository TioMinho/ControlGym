# ==== Libraries ====
using LinearAlgebra

# ===================

# ==== Functions ====
"""	(T,Tᵤ) = oneHotEncoding(t)
"""
function oneHotEncoding(t)
	# Computes the (sorted) unique values of the target vector
	tᵤ = sort(unique(t)); tᵤ_dict = Dict(collect(zip(tᵤ, 1:size(tᵤ,1))))

	# Creates and fill the 1-of-K binary coding scheme
	T = zeros(length(t), length(tᵤ))
	for r in 1:size(T,1)
		T[r,tᵤ_dict[t[r]]] = 1
	end

	return (T,tᵤ)
end
