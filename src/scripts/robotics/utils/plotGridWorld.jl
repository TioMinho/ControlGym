# ==== Libraries ====
using Luxor, ColorBrewer
using StatsBase

# Global Variables
cmapDV = palette("RdYlGn", 11)

# ===================

# ==== Functions ====
"""	PLOTGRIDWORLD(grid, V)
"""
function plotGridWorld(grid, V; filename="gridWorld", anim=true)
	gridRows = split(grid,"\n")
	gridFull = replace(grid, r"\n" => "")
	nrows = length(gridRows)
	ncols = length(gridRows[1])


	if anim == true
		for k in 1:size(V,4)
			plotTiles(gridFull, V[:,:,:,k], (nrows, ncols), "tmp/"*filename*"_"*string(k))
		end
	else
		plotTiles(gridFull, V, (nrows, ncols), filename)
	end

end

"""	PLOTTILES(grid, V)
"""
function plotTiles(grid, V, dims, filename)
	nrows, ncols = dims
	V_norm = replace((V .- min(V...)) ./ (max(V...) .- min(V...)), NaN=>0)

	Drawing(80*ncols, 80*nrows, "res/"*filename*".png")
	background("black")
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
				for k = 1:size(V,3)
					Gₓ, Gᵧ = Int.([floor(n/ncols)-(n%ncols==0)+1 n%ncols+ncols*(n%ncols==0)])
					sethue(cmapDV[Int.(ceil(V_norm[Gₓ,Gᵧ,k]/0.11))+1])

					poly([Point(0, 0),
							Point((+sin(π/2*k)-cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)+cos(π/2*k))*tiles.tileheight/2),
							Point((-sin(π/2*k)-cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)-cos(π/2*k))*tiles.tileheight/2)],
							:fill, close=true)

					fontface("Roboto"); sethue("black");
					textcentered(string(round(V_norm[Gₓ,Gᵧ,k], digits=2)),
								 Point(-cos(-π/2*k)*tiles.tilewidth/3.25,
								 		sin(-π/2*k)*tiles.tileheight/3+(cos(-π/2*k)^2)*4+(sin(-π/2*k)^2)*4))
				end

				sethue((42, 42, 42)./255)
				box(Point(0,0), tiles.tilewidth, tiles.tileheight, :stroke)
			end
		end
	end

	finish()
end

# ===================
