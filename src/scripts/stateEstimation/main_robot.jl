# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats
using ControlSystems, Plots

folders = (@__DIR__).*["/filters", "/utils"]
for folder in folders;	for file in readdir(folder)
	include(folder*"/"*file)
end; end

# ==== Variables ====
# Nonlinear State and Output Equations
Nₓ = 3; Nᵧ = 3; Nᵤ = 2;
Δt = 0.1; # seconds  (Sampling time)

f(x,u) = x + [cos(x[3])Δt * u[1]
			  sin(x[3])Δt * u[1]
			  u[2]*Δt]
g(x) = x

# Linearized System (Jacobians)
A(xₛₛ,uₛₛ) = [1  0  -(uₛₛ[1]*Δt)sin(xₛₛ[3]);
	 		  0  1   (uₛₛ[1]*Δt)cos(xₛₛ[3]);
	 		  0  0         1        ]
B(xₛₛ,uₛₛ) = [(Δt)cos(xₛₛ[3])  0 ;
	 		  (Δt)sin(xₛₛ[3])  0 ;
		  		   0       	  Δt ]
C(xₛₛ) = I(Nₓ)

sys = (f,g,A,B,C,Δt,Nₓ,Nᵧ,Nᵤ)

# Input signal
t = 0:Δt:20;
u = [2 .+ 0t  -(0.5π)sin.(0.5t)+(π)sin.(t)]';

# Process and Measurement Noise covariances
Q = 0.001I(3); R = 0.01I(3);

# Initial state
x₀ = zeros(3)
X₀ = MvNormal(x₀, 0.001I(3))

# ===================

# ==== Script ====
# Simulates the System
(y, t, x)    = sim(sys,    u, t, x₀, Q, R)
(Xₑ, μₑ, Σₑ) = UKF(sys, y, u, t, X₀, Q, R)

tᵢ = length(t)
scatter(y[1,1:tᵢ], y[2,1:tᵢ], m=(:star5, 2, stroke(0)), markeralpha=range(0,0.7,length=tᵢ+1),
		   xlim=(min(y[1,:]...)-2, max(y[1,:]...)+2),
		   ylim=(min(y[2,:]...)-2, max(y[2,:]...)+2),
		   ticks=nothing, size=(16,9).*30, dpi=300)

if(tᵢ>1)
	plot!(μₑ[1, 1:(tᵢ-1)], μₑ[2, 1:(tᵢ-1)], alpha=range(0,0.7,length=tᵢ+1))
end

scatter!([μₑ[1,tᵢ]], [μₑ[2,tᵢ]], alpha=0.25, marker=(arrow(u[1,tᵢ], μₑ[3,tᵢ]), 20*u[1,tᵢ], stroke(1, 0.1, :white)))
scatter!([μₑ[1,tᵢ]], [μₑ[2,tᵢ]], alpha=0.80, marker=(circle(μₑ[3,tᵢ]), 15, :white))
scatter!([μₑ[1,tᵢ]], [μₑ[2,tᵢ]], alpha=0.25, marker=(arrowRotation(u[2,tᵢ], μₑ[3,tᵢ]), 25, stroke(1, 0.1, :white)))

# anim = @animate for tᵢ ∈ 1:length(t)
# 	scatter(y[1,1:tᵢ], y[2,1:tᵢ], m=(:star5, 2, stroke(0)), markeralpha=range(0,0.7,length=tᵢ+1),
#                xlim=(min(y[1,:]...)-2, max(y[1,:]...)+2),
#                ylim=(min(y[2,:]...)-2, max(y[2,:]...)+2),
#                ticks=nothing, size=(16,9).*30, dpi=400)
#
# 	if(tᵢ>1)
# 		plot!(μₑ[1, 1:(tᵢ-1)], μₑ[2, 1:(tᵢ-1)], alpha=range(0,0.7,length=tᵢ+1))
# 	end
#
# 	scatter!([μₑ[1,tᵢ]], [μₑ[2,tᵢ]], alpha=0.25, marker=(arrow(u[1,tᵢ], μₑ[3,tᵢ]), 20*u[1,tᵢ], stroke(1, 0.1, :white)))
# 	scatter!([μₑ[1,tᵢ]], [μₑ[2,tᵢ]], alpha=0.80, marker=(circle(μₑ[3,tᵢ]), 15, :white))
# 	scatter!([μₑ[1,tᵢ]], [μₑ[2,tᵢ]], alpha=0.25, marker=(arrowRotation(u[2,tᵢ], μₑ[3,tᵢ]), 25, stroke(1, 0.1, :white)))
# 	savefig("res/tmp/tmp_ekf$(ti).png")
# end
# gif(anim, "res/ekf_car.gif", fps=10)
