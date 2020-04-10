# ==== Libraries ====
using LinearAlgebra, Distributions, StatsBase, Random, PDMats
using ControlSystems

# ===================

# ==== Functions ====
# (Y,T,X) = SIM(SYS,U,T,X₀;Q=nothing,R=nothing,MODE="nonlinear")
# 	Simulates the discrete-time state-space system given by the model SYS.
#	Given a input sequence U = [u₁,⋯,uₜ] for a time-span T = [t₀, tₜ], computes the simulation
#		MODE == "nonlinear"			MODE =="linear"
#		xₖ₊₁ = f(xₖ, uₖ)			xₖ₊₁ = Axₖ + Buₖ
#		yₖ	 = g(xₖ)				yₖ	 = Cxₖ
#	with initial state x₀ given by the arguments.
#	If Q or R are different than "nothing", the function simulates the stochastic system
#		MODE == "nonlinear"			MODE =="linear"
#		xₖ₊₁ = f(xₖ, uₖ) + vₖ		xₖ₊₁ = Axₖ + Buₖ + vₖ
#		yₖ	 = g(xₖ)	 + zₖ		yₖ	 = Cxₖ		 + zₖ
#	with Vₖ ~ 𝓝(0,Q), Zₖ ~ 𝓝(0,R), and initial state x₀ given by the arguments.
#
function sim(sys, u, t, x₀; Q=nothing, R=nothing, mode="nonlinear")
	# Auxiliary variables
	(f,g,A,B,C,Δt,Nₓ,Nᵧ,Nᵤ) = sys
	t = t[1]:Δt:t[end]
	mode = lowercase(mode)

	x = zeros(Nₓ, length(t))	# List of state-vectors x = [x₁,⋯,xₜ]
	y = zeros(Nᵧ, length(t))	# List of output-vectors y = [y₁,⋯,yₜ]

	# Create the distributions for the random noise variables
	if Q != nothing; Vₖ = MvNormal(zeros(Nₓ), Q)	# Vₖ ~ 𝓝(0,Q)
	else 			 Vₖ = [0]
	end

	if R != nothing; Zₖ = MvNormal(zeros(Nᵧ), R)	# Zₖ ~ 𝓝(0,R)
	else 			 Zₖ = [0]
	end

	# == SIMULATION LOOP ==
	# Initial state
	x[:,1] = x₀ + rand(Vₖ)

	# Intermediate state transitions and emissions
	for k ∈ 1:length(t)-1
		if mode == "nonlinear"
			x[:,k+1] = f(x[:,k], u[:,k]) + rand(Vₖ)
			y[:,k]   = g(x[:,k]) 		 + rand(Zₖ)
		elseif mode == "linear"
			x[:,k+1] = A*x[:,k] + B*u[:,k]  + rand(Vₖ)
			y[:,k]   = C*x[:,k] 			+ rand(Zₖ)
		else
			println("!!! Illegal parameter (MODE) !!!")
		end
	end

	# Last state emission
	if mode == "nonlinear";  y[:,end] = g(x[:,end]) + rand(Zₖ)
	elseif mode == "linear"; y[:,end] = C*x[:,end]  + rand(Zₖ)
	end

	# ====

	return (y,t,x)
end
