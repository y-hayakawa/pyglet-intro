import pyglet

window = pyglet.window.Window(width=400,height=300)
pyglet.gl.glClearColor(0.3, 0.3, 0.5, 1.0)

@window.event
def on_draw():
    window.clear()

pyglet.app.run()
