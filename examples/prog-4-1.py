import pyglet
from pyglet import shapes, text
from pyglet.math import Mat4,Vec3

# ウィンドウの生成
window = pyglet.window.Window(width=1280,height=720,resizable=True)
pyglet.gl.glClearColor(0.3, 0.3, 0.5, 1.0)

# バッチの生成
batch = pyglet.graphics.Batch()

# 図形オブジェクトの生成
image = pyglet.image.load('moon.png')
sprite = pyglet.sprite.Sprite(img=image, x=150, y=130,  batch=batch)

label = text.Label('Pygletの２次元図形', font_name='Arial', font_size=36, color=(10,200,250,255), \
                   x=-400, y=200, anchor_x='left', anchor_y='center', batch=batch)

bezier = shapes.BezierCurve((-450,100),(100,-100),(0,-300),(200,-500),(400,200),thickness=5, \
                            color=(100,200,200,200),batch=batch)

circle = shapes.Circle(-300, -100, 100, color=(255, 200, 40), batch=batch)

rectangle = shapes.Rectangle(-150, -200, 100, 250, color=(120, 200, 150), batch=batch)

box = shapes.Box(x=100, y=-150, width=200, height=200, thickness=3, color=(200, 200, 255), batch=batch)

vlist = [(100,-200),(0,-100),(0,0),(100,100),(300,100),(400,0),(400,-100),(300,-200)]
polygon = shapes.Polygon(*vlist, color=(200,10,210, 100), batch=batch)

star = shapes.Star(x=410, y=240, outer_radius=30, inner_radius=5, num_spikes=8, color=(240, 240, 255), \
                   batch=batch)



# リサイズ用イベントハンドラーの定義
@window.event
def on_resize(width,height):
    window.viewport = (0, 0, width, height)
    ratio = height/width
    window.projection = Mat4.orthogonal_projection(-500, 500, -500*ratio, 500*ratio, -100, 100)    
    return pyglet.event.EVENT_HANDLED

rot_angle = 0.0

# 描画用イベントハンドラーの定義
@window.event
def on_draw():
    window.clear()
    window.view = Mat4.from_rotation(rot_angle, Vec3(0,0,1))
    batch.draw()

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global rot_angle
    rot_angle += scroll_y * 3.1415/60

@window.event
def on_mouse_press(x,y,button,modifier):
    x -= image.width/2
    y -= image.height/2
    x2 = (x-window.width/2)*1000/window.width
    y2 = (y-window.height/2)*1000/window.width
    sprite.x = x2
    sprite.y = y2

# イベントループ開始
pyglet.app.run()

