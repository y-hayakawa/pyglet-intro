import numpy as np
import math
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3, Vec4
    
window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Jupiter')

batch = pyglet.graphics.Batch()

time=0
def update(dt):
    global time
    time += dt*0.3

@window.event
def on_draw():
    window.clear()
    rot_mat = Mat4.from_rotation(time,Vec3(0,1,0)) 
    model_jupiter.matrix = Mat4.from_translation(Vec3(0,0,0)) @ rot_mat
    batch.draw()

@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.1, z_far=255, fov=60)
    return pyglet.event.EVENT_HANDLED

def setup():
    glClearColor(0.3, 0.3, 0.5, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    on_resize(*window.size)
    
# OpenGLの初期設定
setup()

# OBJファイルの読み込み
model_jupiter = pyglet.resource.model("jupiter.obj", batch=batch)

# 視点を設定
window.view = Mat4.look_at(position=Vec3(0,0,3), target=Vec3(0,0,0), up=Vec3(0,1,0))

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()
