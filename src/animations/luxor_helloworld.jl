using Luxor

width, height = (800, 600)
cx, cy = (width, height) ./ 2
Drawing(width, height, "res/hello-world.png")

background("black")
origin()

# Floor
sethue("white"); rect(Point(-cx, cy-100), width, 100, :fill)

# Cart
sethue((185,103,255)./255); box(Point(0.0, (cy-100)-25), 75, 50, :fill)
translate(); sethue((255,251,150)./255); box(Point(0.0, (cy-100)-25+50), 10, 100, :fill)

finish()
preview()