# ==== Libraries ====
using LinearAlgebra

# ===================

# ==== Functions ====
"""	a
"""
function QLearning(T, R, doms; α=0.2, γ=0.95, episodes=50, it=50)
	# Allocates the action-dependent (V) and optimal (Vˣ) matrices
    Q  = zeros(size(doms[1])..., size(doms[2])..., it, episodes)
    Qₖ = zeros(size(doms[1])..., size(doms[2])...); 
    Qˣ = zeros(size(doms[1])...);

    x = zeros(Int8, 2,it,episodes); 
    u = zeros(Int8, 1,it,episodes); 
    r = zeros(Int8, 1,it,episodes); 

    for e in 1:episodes
        x[:,1,e] = sample([[1; 1], [1; 11], [7, 11]]); Q[:,:,:,1,e] = Qₖ;

        ϵ = min(1, 1-log10((e+1)/25))

        for k in 1:it-1
            sₖ = x[:,k,e];
            aₖ = sample(doms[2])
            
            if rand() >= ϵ; aₖ = findmax(Qₖ[sₖ...,:])[2];
            end

            sₖ₊₁ = Int.(T(sₖ,aₖ))

            Qₖ[sₖ...,aₖ] = Qₖ[sₖ...,aₖ] + α*((R([sₖ...],aₖ,[sₖ₊₁...])+γ*Qˣ[sₖ₊₁...]) - Qₖ[sₖ...,aₖ]);
            Qˣ[sₖ...]   = maximum(Qₖ[sₖ...,:]);

            #
            x[:,k+1,e] = sₖ₊₁; u[1,k,e] = aₖ; r[1,k,e] = R([sₖ...],aₖ,[sₖ₊₁...]);
            Q[:,:,:,k+1,e] = Qₖ;
        end

	end

	return (Q, x, u, r)
end

# ===================
