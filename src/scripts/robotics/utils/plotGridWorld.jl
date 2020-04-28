# ==== Libraries ====
using Luxor, ColorBrewer

# Global Variables
width, height = (800, 800)
cx, cy = (width, height) ./ 2
cmapDV = palette("RdYlGn", 11)

# Configurations
Drawing(width, height, "res/gridWorld.png")
background("black")
origin()

# ===================

# ==== Functions ====
tiles = Tiler(width, height, 10, 10, margin=0)
for (pos, n) in tiles
	id = rand(1:11)
	sethue(cmapDV[id])
	box(pos, tiles.tilewidth, tiles.tileheight,
		:fill)

	sethue((42, 42, 42)./255)
	box(pos, tiles.tilewidth, tiles.tileheight,
		:stroke)

	sethue("white")
	textcentered(string(n), pos + Point(0, 5))
end

# sethue("white"); rect(Point(0, 0), width, 100, :fill)

finish()
preview()

# ===================
