# ==== Libraries ====
using Plots

# Configurations
theme(:dark)
pyplot(leg=false)

# ===================

# ==== Functions ====
"""	𝓥 = DRAW_BASE(θ)
Defines a set of vertices 𝓥 = {(x,y) | ∀θ ∈ [0,π] & x = sin(θ) & x = cos(θ)} to
draw a circle representing the base of the robot.
"""
function draw_base(θ)
	x = -(θ:0.1:(2π+θ)).+π/2
	vert = vcat([(0., 0.)], [(xi,yi) for (xi,yi) in zip(sin.(x), cos.(x))])
	return Shape(vert)
end

"""	𝓥 = DRAW_VELOCITY(v,θ)
Defines a set of vertices 𝓥 = {(x,y)} to draw an arrow pointing from the front of the base.
The length of the arrow is given by the magnitude of the velocity v.
"""
function draw_velocity(v,θ)
	θ -= π/2;
	vert = vcat([0.4/(1+v).*(sin(-θ),cos(-θ)), (sin(-θ),cos(-θ)), 0.85.*(sin(-θ+0.1),cos(-θ+0.1)), (sin(-θ),cos(-θ)), 0.85.*(sin(-θ-0.1),cos(-θ-0.1)), (sin(-θ),cos(-θ))]...)
	return Shape(vert)
end

"""	𝓥 = DRAW_ROTATION(ω,θ)
Defines a set of vertices 𝓥 = {(x,y)} to draw a curvy arrow around the base of the robot.
The length and direction of the arrow is given by the magnitude of the angular velocity ω.
"""
function draw_rotation(ω, θ)
	θ += π/2;
	if ω > 0; x = -((θ+0.5π-0.25ω):0.2:(θ+0.5π+0.25ω))
			  vert = vcat([(sin(x[end]+π/8), cos(x[end]+π/8)), 0.85.*(sin(x[end]), cos(x[end])), 0.7.*(sin(x[end]+π/8), cos(x[end]+π/8))]...)
			  vert = vcat([(xi,yi).*0.85 for (xi,yi) in zip(sin.(x), cos.(x))], vert)
	else;	  x = -((θ-0.5π+0.25ω):0.2:(θ-0.5π-0.25ω))
			  vert = vcat([(sin(x[1]-π/8), cos(x[1]-π/8)), 0.85.*(sin(x[1]), cos(x[1])), 0.7.*(sin(x[1]-π/8), cos(x[1]-π/8))]...)
			  vert = vcat(vert, [(xi,yi).*0.85 for (xi,yi) in zip(sin.(x), cos.(x))])
	end

	vert = vcat(vert, vert[end:-1:1])
	return Shape(vert)
end

"""	P = PLOT_TRAJECTORY(T,X,Y,U;Xₑ=nothing,anim=false,name="car")
Plot the time (T=[t₀,tₜ]) trajectory of a system described by a sequence of state-vectors X=[x₁,⋯,xₜ],
a sequence of output-vectors Y=[y₁,⋯,yₜ] and a sequence of input-vectors U=[u₁,⋯,uₜ].
If Xₑ ≠ nothing, such that Xₑ=[xₑ₁,⋯,xₑₜ] is the sequence of estimated state-vectors, the function plots
the estimated state trajectory in a solid line and the real state trajectory in a near-transparent dashed-line.
"""
function plot_trajectory(t, x, y, u; xₑ=nothing, anim=false, name="car")
	# Initial checks
	if anim == false; t₀ = length(t)
	else; 			  t₀ = 1
	end

	# Plot the robot
	vid = @animate for tᵢ ∈ t₀:length(t)
		scatter(y[1,1:tᵢ], y[2,1:tᵢ], m=(:star5, 3, stroke(0)), markeralpha=range(0,0.7,length=tᵢ+1),
	               xlim=(min(y[1,:]...)-2, max(y[1,:]...)+2),
	               ylim=(min(y[2,:]...)-2, max(y[2,:]...)+2),
	               ticks=nothing, size=(16,9).*30, dpi=400)

		if tᵢ > 1
			if xₑ == nothing
				plot!(x[1, 1:(tᵢ-1)], x[2, 1:(tᵢ-1)], alpha=range(0,0.8,length=tᵢ+1))
			else
				plot!(xₑ[1, 1:(tᵢ-1)], xₑ[2, 1:(tᵢ-1)], alpha=range(0,0.8,length=tᵢ+1))
				plot!(x[1, 1:(tᵢ-1)], x[2, 1:(tᵢ-1)], l=(1,:dash), alpha=range(0,0.3,length=tᵢ+1))
			end
		end

		scatter!([xₑ[1,tᵢ]], [xₑ[2,tᵢ]], alpha=0.25, marker=(arrow(u[1,tᵢ], xₑ[3,tᵢ]), 20*u[1,tᵢ], stroke(1, 0.1, :white)))
		scatter!([xₑ[1,tᵢ]], [xₑ[2,tᵢ]], alpha=0.80, marker=(circle(xₑ[3,tᵢ]), 15, :white))
		scatter!([xₑ[1,tᵢ]], [xₑ[2,tᵢ]], alpha=0.25, marker=(arrowRotation(u[2,tᵢ], xₑ[3,tᵢ]), 25, stroke(1, 0.1, :white)))
		savefig("res/tmp/tmp_$(name)$(tᵢ).png")
	end

	# Saves the figure or the animation
	if anim == false; savefig("res/$(name).pdf")
	else; 			  gif(vid, "res/$(name).gif", fps=10)
	end

end

# ===================
