# ==== Libraries ====
using Luxor, ColorBrewer
using StatsBase

# Global Variables
cmapDV = palette("RdYlGn", 11)

# ===================

# ==== Functions ====
"""	PLOTGRIDWORLD(grid, V)
"""
function plotGridWorld(grid, V; filename="gridWorld", anim=true, qlearn=true, data=nothing)
	gridRows = split(grid,"\n")
	gridFull = replace(grid, r"\n" => "")
	nrows = length(gridRows)
	ncols = length(gridRows[1])


	if anim == true
		if qlearn == true
			for e in [1; 50:50:size(V,5)], k in 1:size(V,4)
				plotTiles_robot(gridFull, V[:,:,:,k,e], data, e, (nrows, ncols), "tmp/"*filename*"_e$(e)_$k", epoch=k)
			end
		else
			for k in 1:size(V,4)
				plotTiles(gridFull, V[:,:,:,k], (nrows, ncols), "tmp/"*filename*"_"*string(k), epoch=k)
			end
		end
	else
		plotTiles(gridFull, V, (nrows, ncols), filename)
	end

end

function plotTiles_robot(grid, V, data, epi, dims, filename; epoch=nothing)
	nrows, ncols = dims
	V_norm = replace((V .- min(V...)) ./ (max(V...) .- min(V...)), NaN=>0)

	Drawing(80*ncols, 90*nrows, "res/"*filename*".png")
	background("white")
	origin()

	tiles = Tiler(80*ncols, 80*nrows, nrows, ncols, margin=0)
	for (pos, n) in tiles
		@layer begin
			translate(pos)
			if grid[n] == 'x'
				sethue((22, 22, 22)./255)
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :fill)

			elseif grid[n] == 'o'
				sethue(cmapDV[end])
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :fill)

				sethue((242, 242, 242)./255)
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :stroke)
				box(Point(0,0), tiles.tilewidth/1.15, tiles.tileheight/1.15, :stroke)

				sethue("white"); fontsize(18); fontface("Roboto")
				text("END", Point(-1, 6), halign=:center, valign=:center)

			elseif grid[n] == '.'
				Gₓ, Gᵧ = Int.([floor(n/ncols)-(n%ncols==0)+1 n%ncols+ncols*(n%ncols==0)])

				for k = 1:size(V,3)
					sethue(cmapDV[Int.(ceil(V_norm[Gₓ,Gᵧ,k]/0.11))+1])

					poly([Point(0, 0),
							Point((+sin(π/2*k)-cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)+cos(π/2*k))*tiles.tileheight/2),
							Point((-sin(π/2*k)-cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)-cos(π/2*k))*tiles.tileheight/2)],
							:fill, close=true)

					# fontface("Roboto"); sethue("black");
					# textcentered(string(round(V_norm[Gₓ,Gᵧ,k], digits=2)),
					# 			 Point(-cos(-π/2*k)*tiles.tilewidth/3.25,
					# 			 		sin(-π/2*k)*tiles.tileheight/3+(cos(-π/2*k)^2)*4+(sin(-π/2*k)^2)*4))
				end

				sethue((42, 42, 42)./255)
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :stroke)

				if Gₓ == data[1,epoch,epi] && Gᵧ == data[2,epoch,epi]
					sethue("black");
					direction = data[3,epoch,epi]
					if 		direction == 1; arrow(Point(  0,-15), Point(  0,-35), linewidth=3)
					elseif 	direction == 2; arrow(Point( 19,  0), Point( 39,  0), linewidth=3)
					elseif 	direction == 3; arrow(Point(  0, 15), Point(  0, 35), linewidth=3)
					elseif 	direction == 4; arrow(Point(-19,  0), Point(-39,  0), linewidth=3)
					end

					Luxor.scale(0.3, 0.3)
					placeimage(readpng("res/robot.png"), Point(0,0), centered=true)
				end
			end
		end
	end

	if epoch ≠ nothing
		sethue("black"); fontsize(18); fontface("Latin Modern Roman")
		text("Epoch = $epoch", Point(-40*ncols+10, -40*nrows-10), halign=:left, valign=:center)
		text("Episode #$epi", Point(0*ncols+10, -40*nrows-10), halign=:center, valign=:center)
		text("Reward = $(sum(data[4,1:epoch,epi])) ($(data[4,epoch,epi]))", Point(+40*ncols-10, -40*nrows-10), halign=:right, valign=:center)
	end

	finish()
end

"""	PLOTTILES(grid, V)
"""
function plotTiles(grid, V, dims, filename; epoch=nothing)
	nrows, ncols = dims
	V_norm = replace((V .- min(V...)) ./ (max(V...) .- min(V...)), NaN=>0)

	Drawing(80*ncols, 90*nrows, "res/"*filename*".png")
	background("white")
	origin()

	tiles = Tiler(80*ncols, 80*nrows, nrows, ncols, margin=0)
	for (pos, n) in tiles
		@layer begin
			translate(pos)
			if grid[n] == 'x'
				sethue((22, 22, 22)./255)
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :fill)

			elseif grid[n] == 'o'
				sethue(cmapDV[end])
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :fill)

				sethue((242, 242, 242)./255)
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :stroke)
				box(Point(0,0), tiles.tilewidth/1.15, tiles.tileheight/1.15, :stroke)

				sethue("white"); fontsize(18); fontface("Roboto")
				text("END", Point(-1, 6), halign=:center, valign=:center)

			elseif grid[n] == '.'
				Gₓ, Gᵧ = Int.([floor(n/ncols)-(n%ncols==0)+1 n%ncols+ncols*(n%ncols==0)])

				for k = 1:size(V,3)
					sethue(cmapDV[Int.(ceil(V_norm[Gₓ,Gᵧ,k]/0.11))+1])

					poly([Point(0, 0),
							Point((+sin(π/2*k)-cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)+cos(π/2*k))*tiles.tileheight/2),
							Point((-sin(π/2*k)-cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)-cos(π/2*k))*tiles.tileheight/2)],
							:fill, close=true)

					# fontface("Roboto"); sethue("black");
					# textcentered(string(round(V_norm[Gₓ,Gᵧ,k], digits=2)),
					# 			 Point(-cos(-π/2*k)*tiles.tilewidth/3.25,
					# 			 		sin(-π/2*k)*tiles.tileheight/3+(cos(-π/2*k)^2)*4+(sin(-π/2*k)^2)*4))
				end

				sethue("black");
				direction = findmax(V_norm[Gₓ,Gᵧ,:])[2]
				if 		direction == 1; arrow(Point(  0, 15), Point(  0,-15), linewidth=3)
				elseif 	direction == 2; arrow(Point(-12,  0), Point( 12,  0), linewidth=3)
				elseif 	direction == 3; arrow(Point(  0,-15), Point(  0, 15), linewidth=3)
				elseif 	direction == 4; arrow(Point( 12,  0), Point(-12,  0), linewidth=3)
				end

				sethue((42, 42, 42)./255)
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :stroke)
			end
		end
	end

	if epoch ≠ nothing
		sethue("black"); fontsize(18); fontface("Latin Modern Roman")
		text("Value Iteration #$epoch", Point(0, -40*nrows-10), halign=:center, valign=:center)
		# text("K = $epoch", Point(-40*ncols+10, -40*nrows-10), halign=:left, valign=:center)
	end

	finish()
end

# ===================
