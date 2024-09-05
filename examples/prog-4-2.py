import pyglet
from pyglet import shapes, text
from pyglet.math import Mat4,Vec3
import math

# ウィンドウの生成
# config = pyglet.gl.Config(sample_buffers = 1,samples = 4,doubel_buffer=True)
window = pyglet.window.Window(width=1280,height=720,resizable=True)
# pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)
pyglet.gl.glClearColor(0.3, 0.3, 0.5, 1.0)

# バッチの生成
batch = pyglet.graphics.Batch()

# Sierpinskiカーペットの生成
triangles = [ ]
def sierpinski(level,x,y):
    ell = (1/2)**(level-1)
    if level<=7:
        sierpinski(level+1,x-(ell/4), y-ell*math.sqrt(3)/12)
        sierpinski(level+1,x+(ell/4), y-ell*math.sqrt(3)/12)
        sierpinski(level+1,x, y+ell*math.sqrt(3)/6)
    else:
        vertices = [x-ell*1/2, y-ell*math.sqrt(3)/6, x+ell*1/2, y - ell*math.sqrt(3)/6, x, y+ell*math.sqrt(3)/3]
        triangle = shapes.Triangle(*vertices, color=(255, 200, 40), batch=batch)
        triangles.append(triangle)

# リサイズ用イベントハンドラーの定義
@window.event
def on_resize(width,height):
    window.viewport = (0, 0, width, height)
    ratio = width/height
    window.projection = Mat4.orthogonal_projection(-1*ratio, 1*ratio, -1, 1, -100, 100)    
    return pyglet.event.EVENT_HANDLED

zoom = 1.0
pos_dx = 0
pos_dy = 0

# 描画用イベントハンドラーの定義
@window.event
def on_draw():
    window.clear()
    window.view = Mat4.from_translation(Vec3(pos_dx, pos_dy, 0)) @ Mat4.from_scale(Vec3(zoom,zoom,1.0))
    batch.draw()

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global pos_dx, pos_dy
    pos_dx += 2*dx/window.height
    pos_dy += 2*dy/window.height
    
@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global zoom
    zoom *= math.exp(scroll_y/100)

sierpinski(0,0,0)

# イベントループ開始
pyglet.app.run()
