# ==== Libraries ====
using Plots

# Configurations
theme(:dark)
pyplot(leg=false)

# ===================

# ==== Functions ====
function circle(θ)
	x = -(θ:0.1:(2π+θ)).+π/2
	vert = vcat([(0., 0.)], [(xi,yi) for (xi,yi) in zip(sin.(x), cos.(x))])
	return Shape(vert)
end

function arrow(v,θ)
	θ -= π/2;
	vert = vcat([0.4/(1+v).*(sin(-θ),cos(-θ)), (sin(-θ),cos(-θ)), 0.85.*(sin(-θ+0.1),cos(-θ+0.1)), (sin(-θ),cos(-θ)), 0.85.*(sin(-θ-0.1),cos(-θ-0.1)), (sin(-θ),cos(-θ))]...)
	return Shape(vert)
end,

function arrowRotation(ω, θ)
	θ += π/2;
	if(ω>0); x = -((θ+0.5π-0.25ω):0.2:(θ+0.5π+0.25ω))
			 vert = vcat([(sin(x[end]+π/8), cos(x[end]+π/8)), 0.85.*(sin(x[end]), cos(x[end])), 0.7.*(sin(x[end]+π/8), cos(x[end]+π/8))]...)
			 vert = vcat([(xi,yi).*0.85 for (xi,yi) in zip(sin.(x), cos.(x))], vert)
	else;	 x = -((θ-0.5π+0.25ω):0.2:(θ-0.5π-0.25ω))
			 vert = vcat([(sin(x[1]-π/8), cos(x[1]-π/8)), 0.85.*(sin(x[1]), cos(x[1])), 0.7.*(sin(x[1]-π/8), cos(x[1]-π/8))]...)
			 vert = vcat(vert, [(xi,yi).*0.85 for (xi,yi) in zip(sin.(x), cos.(x))])
	end

	vert = vcat(vert, vert[end:-1:1])
	return Shape(vert)
end

# ===================
