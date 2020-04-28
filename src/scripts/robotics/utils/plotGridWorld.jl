# ==== Libraries ====
using Luxor, ColorBrewer

# Global Variables
cmapDV = palette("RdYlGn", 11)

# Configurations
# ===================

# ==== Functions ====
# Grid text
grid = """......
		  ..x...
		  .xx...
		  ......
		  xxx.xo"""

gridRows = split(grid,"\n")
gridFull = replace(grid, r"\n" => "")
nrows = length(gridRows)
ncols = length(gridRows[1])

Drawing(80*ncols, 80*nrows, "res/gridWorld.png")
background("black")
origin()

tiles = Tiler(80*ncols, 80*nrows, nrows, ncols, margin=0)
for (pos, n) in tiles
	@layer begin
		translate(pos)
		if gridFull[n] == 'x'
			sethue((22, 22, 22)./255)
			box(Point(0,0), tiles.tilewidth, tiles.tileheight, :fill)

		elseif gridFull[n] == 'o'
			sethue(cmapDV[end])
			box(Point(0,0), tiles.tilewidth, tiles.tileheight, :fill)

			sethue((242, 242, 242)./255)
			box(Point(0,0), tiles.tilewidth, tiles.tileheight, :stroke)
			box(Point(0,0), tiles.tilewidth/1.15, tiles.tileheight/1.15, :stroke)

			sethue("white"); fontsize(18); fontface("Roboto")
			text("END", Point(-1, 6), halign=:center, valign=:center)

		elseif gridFull[n] == '.'
			for k = 0:3
				id = rand(1:11)
				sethue(cmapDV[id])
				poly([Point(0, 0),
						Point((+sin(π/2*k)+cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)+cos(π/2*k))*tiles.tileheight/2),
						Point((-sin(π/2*k)+cos(π/2*k))*tiles.tilewidth/2, (-sin(π/2*k)-cos(π/2*k))*tiles.tileheight/2)],
						:fill, close=true)

				fontface("Roboto"); sethue("black");
				textcentered(string(round(rand(), digits=2)),
							 Point(cos(π/2*k)*tiles.tilewidth/3.25,
							 		-sin(π/2*k)*tiles.tileheight/3+(cos(π/2*k)^2)*4+(sin(π/2*k)^2)*4))
			end

			sethue((42, 42, 42)./255)
			box(Point(0,0), tiles.tilewidth, tiles.tileheight, :stroke)
		end
	end

end

finish()

# ===================
