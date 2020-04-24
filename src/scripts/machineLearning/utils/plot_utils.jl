# ==== Libraries ====
using Plots

# Configurations
theme(:dark)
pyplot(leg=false)

# ===================

# ==== Functions ====
"""	P = PLOT_REGRESSION(𝓓, w, ϕ; method="MLE")

Plots the data and the estimates of the regression model given its training method.

If method=="MLE":
	-> Plots the data as scatter-points 𝓓 = {x,t} and estimates the target value for the
		correspondent interval using the linear model yₑ = w'ϕ(x)
If method=="BAYES":
	-> Plots the data as scatter-points 𝓓 = {x,t}, the target value estimates for the
		correspondent interval using mean of predictive distribution yₑ ~ 𝓝(μₙ,σₙI),
		with μₙ = wₙ.μ'x and σₙ = ϕ(x)'(wₙ.Σ)ϕ(x), and also plots the area for
		interval [μₙ-σ²ₙ, μₙ+σ²ₙ] containing ≈95% of the probability mass of yₑ
"""
function plot_regression(𝓓, w, ϕ; method="MLE")
	# Define some auxiliary variables and perform some pre-processing
	(x,t) = [𝓓.x, 𝓓.t]; n = size(x,1)
	x_rng = [min(x...), max(x...)];	t_rng = [min(t...), max(t...)]

	method = lowercase(method)

	# Plots the datapoints
	p = scatter(x, t, m=(4, 0.5, stroke(0)), lab="Data")

	# Plots the regression model
	xₑ = range(1.1*min(x...), 1.1*max(x...), length=1000)
	if method=="mle"
		yₑ = w'ϕ(xₑ)
		plot!(xₑ, yₑ', l=(1), lab=titlecase(method))

	elseif method=="bayes"
		μ = w[n].μ'ϕ(xₑ); σ = sqrt.(diag(ϕ(xₑ)'*w[n].Σ*ϕ(xₑ)));

		plot!(xₑ, μ', l=(1), lab=titlecase(method))
		plot!(xₑ, μ'+2σ, f=(μ'-2σ, 0.15), l=0)


	end

	# Adjust some details of the plot
	plot(p, xlab="Attribute - x", ylab="Target - t")
	xlims!(1.1*x_rng...); ylims!(t_rng...)

	# ---
	return p
end

"""	P = PLOT_CLASSIFICATION(𝓓, w, ϕ; method="MLE")

Plots the data and the decision boundary of the classification model given its training method.

If method=="MLE":
If method=="BAYES":
"""
function plot_classification(𝓓, w, ϕ; method="MLE")
	# Define some auxiliary variables and perform some pre-processing
	(x₁,x₂,t) = [𝓓.x₁, 𝓓.x₂, 𝓓.t]; n = size(t,1)
	x₁_rng = [min(x₁...), max(x₁...)];	x₂_rng = [min(x₂...), max(x₂...)]

	method = lowercase(method)

	# Plots the datapoints
	𝓒₀ = (t.==0); 𝓒₁ = (t.==1)
	p = scatter(x₁[𝓒₁], x₂[𝓒₁], m=(4, :lightgreen, 0.5, stroke(0)), lab="Data (C₁)")
		scatter!(x₁[𝓒₀], x₂[𝓒₀], m=(4, :red, 0.5, stroke(0)), lab="Data (C₀)")

	# Plots the regression model
	# xₑ = range(1.1*min(x...), 1.1*max(x...), length=1000)
	# if method=="mle"
	# 	yₑ = w'ϕ(xₑ)
	# 	plot!(xₑ, yₑ', l=(1), lab=titlecase(method))
	#
	# elseif method=="bayes"
	# 	μ = w[n].μ'ϕ(xₑ); σ = sqrt.(diag(ϕ(xₑ)'*w[n].Σ*ϕ(xₑ)));
	#
	# 	plot!(xₑ, μ', l=(1), lab=titlecase(method))
	# 	plot!(xₑ, μ'+2σ, f=(μ'-2σ, 0.15), l=0)
	#
	#
	# end

	# Adjust some details of the plot
	plot(p, xlab="Attribute - x₁", ylab="Attribute - x₂", leg=true)
	xlims!(1.1*x₁_rng...); ylims!(1.1*x₂_rng...)

	# ---
	return p
end

# ===================
