# == Libraries ==
using LinearAlgebra, Distributions, StatsBase, Random
using PaddedViews

folders = (@__DIR__).*["/MDP", "/utils"]
for folder in folders;	for file in readdir(folder)
	include(folder*"/"*file)
end; end

# =====

# == Functions ==
function T(sₖ,aₖ,sₖ₊₁; mask=zeros(5,5))
	aₖ_dir =  round.([-sin(π/2*aₖ); -cos(π/2*aₖ)])
	aₖ_err1 = round.([+cos(π/2*aₖ); -sin(π/2*aₖ)])
	aₖ_err2 = round.([-cos(π/2*aₖ); +sin(π/2*aₖ)])

	if 	   (sₖ₊₁ == sₖ+aₖ_dir);  return 0.9*mask[sₖ₊₁...]
	elseif (sₖ₊₁ == sₖ+aₖ_err1); return 0.05*mask[sₖ₊₁...]
	elseif (sₖ₊₁ == sₖ+aₖ_err2); return 0.05*mask[sₖ₊₁...]
	end

	return 0;
end

function T_a(sₖ,aₖ; mask=zeros(5,5))
	aₖ_dir =  Int.(round.([-sin(π/2*aₖ); -cos(π/2*aₖ)]))
	aₖ_err1 = Int.(round.([+cos(π/2*aₖ); -sin(π/2*aₖ)]))
	aₖ_err2 = Int.(round.([-cos(π/2*aₖ); +sin(π/2*aₖ)]))

	if	mask[(sₖ+aₖ_dir)...] == 1;  return sₖ+aₖ_dir;
	end

	return sₖ;
end

function R(sₖ,aₖ,sₖ₊₁; mask=zeros(5,5))
	aₖ_dir =  round.([-sin(π/2*aₖ); -cos(π/2*aₖ)])
	aₖ_err1 = round.([+cos(π/2*aₖ); -sin(π/2*aₖ)])
	aₖ_err2 = round.([-cos(π/2*aₖ); +sin(π/2*aₖ)])

	if 	   (sₖ₊₁ == sₖ+aₖ_dir);  return mask[sₖ₊₁...]
	elseif (sₖ₊₁ == sₖ+aₖ_err1); return mask[sₖ₊₁...]
	elseif (sₖ₊₁ == sₖ+aₖ_err2); return mask[sₖ₊₁...]
	end

	return 0;
end

# =====

# == Variables ==
# Definition of the GridWorld and its dimensions
grid = """.xx......x.
		  .xx....xxx.
		  ...........
		  ..x......x.
		  .xx...x..x.
		  ......x....
		  xxx.xox.xx."""

Nₓ, Nᵧ = [length(split(grid, "\n")) length(split(grid, "\n")[1])]
maskT = PaddedView(0, reshape([Int(p=='.')+Int(p=='o') for p in replace(grid,r"\n" => "")], Nᵧ, Nₓ)', (0:Nₓ+1, 0:Nᵧ+1))
maskR = PaddedView(-10, reshape(10*[Int(p=='o')-Int(p=='x') for p in replace(grid,r"\n" => "")], Nᵧ, Nₓ)', (0:Nₓ+1, 0:Nᵧ+1))

# The State (𝓢) and Action (𝓐) domains
𝓢, 𝓐 = [Iterators.product(1:Nₓ, 1:Nᵧ)|>collect, 1:4|>collect]

# Alias for the Transition and Reward functions given the GridWorld
T(sₖ,aₖ,sₖ₊₁) 	= T(sₖ,aₖ,sₖ₊₁, mask=maskT)
T_a(sₖ,aₖ) 		= T_a(sₖ,aₖ, mask=maskT)
R(sₖ,aₖ,sₖ₊₁) 	= R(sₖ,aₖ,sₖ₊₁, mask=maskR)

# =====

# == Script ==
(V, Vˣ) = ValueIteration(T, R, [𝓢, 𝓐], 0.95, it=200)
# (Q, x, u, r) = QLearning(T_a, R, [𝓢, 𝓐], episodes=201, it=100); data = vcat([x; u; r])
plotGridWorld(grid, V, qlearn=false)
