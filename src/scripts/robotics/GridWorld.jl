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

	if 	   (sₖ₊₁ == sₖ+aₖ_dir);  return 0.8*mask[sₖ₊₁...]
	elseif (sₖ₊₁ == sₖ+aₖ_err1); return 0.1*mask[sₖ₊₁...]
	elseif (sₖ₊₁ == sₖ+aₖ_err2); return 0.1*mask[sₖ₊₁...]
	end

	return 0;
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
grid = """......
		  ..x...
		  .xx...
		  ......
		  xxx.xo"""

Nₓ, Nᵧ = [length(split(grid, "\n")) length(split(grid, "\n")[1])]
maskT = PaddedView(0, ones(Nᵧ, Nₓ)', (0:Nₓ+1, 0:Nᵧ+1))
maskR = PaddedView(-1, reshape([Int(p=='o')-Int(p=='x') for p in replace(grid,r"\n" => "")], Nᵧ, Nₓ)', (0:Nₓ+1, 0:Nᵧ+1))

# The State (𝓢) and Action (𝓐) domains
𝓢, 𝓐 = [Iterators.product(1:Nₓ, 1:Nᵧ)|>collect, 1:4|>collect]

# Alias for the Transition and Reward functions given the GridWorld
T(sₖ,aₖ,sₖ₊₁) = T(sₖ,aₖ,sₖ₊₁, mask=maskT)
R(sₖ,aₖ,sₖ₊₁) = R(sₖ,aₖ,sₖ₊₁, mask=maskR)

# =====

# == Script ==
(V, Vˣ) = ValueIteration(T, R, [𝓢, 𝓐], 0.9, it=50)
plotGridWorld(grid, V)
