import numpy as np
import math
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.graphics.shader import Shader, ShaderProgram
import h5py

vertex_source0 = """#version 330 core
    in vec3 position;
    in vec4 colors;
    in float frames;

    out vec3 vertex_position;
    out vec4 vertex_colors;
    flat out int vertex_frames;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

    void main()
    {
        mat4 modelview = window.view * model;
        vec4 pos = modelview * vec4(position, 1.0);
        gl_Position = window.projection * pos;
        gl_PointSize = 3.0 ;
        vertex_position = pos.xyz;
        vertex_colors = colors;
        vertex_frames = int(frames);
    }
"""

fragment_source0 = """#version 330 core
    in vec3 vertex_position;
    in vec4 vertex_colors;
    flat in int vertex_frames;

    out vec4 final_colors;

    uniform int frame;

    void main()
    {
       vec2 pos = gl_PointCoord * 2.0 - 1.0; 
       float r = 1.0 - dot(pos, pos); 
       if (r < 0.0) discard;
       float alpha ;
       if (vertex_frames > frame) alpha = 0.05 ;
       else alpha = 0.98 * exp(-float(frame - vertex_frames)*0.05 ) + 0.02 ; 
       final_colors = vec4(vertex_colors.rgb, alpha) ;
    }
"""

vertex_source1 = """#version 330 core
    in vec3 position;
    in vec4 colors;

    out vec4 vertex_colors;
    out vec3 vertex_position;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

    void main()
    {
        mat4 modelview = window.view * model;
        vec4 pos = modelview * vec4(position, 1.0);
        gl_Position = window.projection * pos;
        vertex_position = pos.xyz;
        vertex_colors = colors;
    }
"""

fragment_source1 = """#version 330 core
    in vec4 vertex_colors;
    in vec3 vertex_position;
    out vec4 final_colors;

    void main()
    {
        final_colors = vertex_colors ;
    }
"""

window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Flock of Starling')

# シェーダーのコンパイル
batch0 = pyglet.graphics.Batch()
vert_shader0 = Shader(vertex_source0, 'vertex')
frag_shader0 = Shader(fragment_source0, 'fragment')
shader0 = ShaderProgram(vert_shader0, frag_shader0)

batch1 = pyglet.graphics.Batch()
vert_shader1 = Shader(vertex_source1, 'vertex')
frag_shader1 = Shader(fragment_source1, 'fragment')
shader1 = ShaderProgram(vert_shader1, frag_shader1)

# 初期化
glClearColor(0.1, 0.1, 0.2, 1.0)
glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 
glEnable(GL_BLEND)

@window.event
def on_draw():
    window.clear()
    shader0['model']=Mat4()
    shader1['model']=Mat4()
    batch0.draw()
    batch1.draw()    

fov = 60
@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    ratio = width/height
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.01, z_far=1000, fov=fov)
    return pyglet.event.EVENT_HANDLED

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global fov
    fov -= scroll_y
    if fov < 1:
        fov = 1
    elif fov>120:
        fov = 120
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.01, z_far=1000, fov=fov)
    return pyglet.event.EVENT_HANDLED

@window.event
def on_mouse_drag(x,y,dx,dy,buttons,modifiers):
    global view_matrix
    rotz = (x-window.width/2)/window.width * fov * dy * 0.0001
    roty = -dx*fov*0.0001
    rotx = dy*fov*0.0001 * np.exp(-8*(window.width/2-x)**2 / window.width**2)
    view_matrix = Mat4.from_rotation(rotz, Vec3(0, 0, 1)) @ view_matrix    
    view_matrix = Mat4.from_rotation(roty, Vec3(0, 1, 0)) @ view_matrix
    view_matrix = Mat4.from_rotation(rotx, Vec3(1, 0, 0)) @ view_matrix
    window.view = view_matrix
    return pyglet.event.EVENT_HANDLED

@window.event
def on_key_press(symbol, modifiers):
    global view_matrix    
    if symbol == pyglet.window.key.R or symbol == pyglet.window.key.HOME:
        view_matrix = Mat4.look_at(position=Vec3(0,50,-50), target=Vec3(0,0,-50), up=Vec3(0,0,1))        
        window.view = view_matrix
    elif symbol == pyglet.window.key.UP:
        view_matrix = Mat4.from_translation(Vec3(0,0,2)) @ view_matrix
        window.view = view_matrix        
    elif symbol == pyglet.window.key.DOWN:
        view_matrix = Mat4.from_translation(Vec3(0,0,-2)) @ view_matrix
        window.view = view_matrix
    elif symbol == pyglet.window.key.RIGHT:
        view_matrix = Mat4.from_translation(Vec3(-2,0,0)) @ view_matrix
        window.view = view_matrix        
    elif symbol == pyglet.window.key.LEFT:
        view_matrix = Mat4.from_translation(Vec3(2,0,0)) @ view_matrix
        window.view = view_matrix
        
    return pyglet.event.EVENT_HANDLED

def gen_grid(x0,x1,y0,y1,z0,z1,step,shader,batch):
    vertices = []
    nx = len(np.arange(x0,x1,step))
    ny = len(np.arange(y0,y1,step))
    nz = len(np.arange(z0,z1,step))
             
    for x in np.arange(x0,x1,step):
        for y in np.arange(y0,y1,step):
            for z in np.arange(z0,z1,step):
                vertices.extend([x, y, z])                
            
    indices = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):            
                p0 = i*ny*nz + j*nz + k
                if i<nx-1:
                    p1 = (i+1)*ny*nz + j*nz + k
                    indices.extend([p0, p1])
                if j<ny-1:
                    p2 = i*ny*nz + (j+1)*nz + k
                    indices.extend([p0, p2])
                if k<nz-1:
                    p3 = i*ny*nz + j*nz + (k+1)
                    indices.extend([p0, p3])                

    vertex_list = shader.vertex_list_indexed(len(vertices)//3, GL_LINES, indices, batch=batch,
                                             position=('f', vertices),
                                             colors =('f', [0.2,0.2,0.5,1]*(len(vertices)//3)))
    return vertex_list

frame_cur = 0
frame_max = 0
frame_min = 0
def update(dt):
    global frame_cur
    shader0['frame'] = int(frame_cur)
    frame_cur +=1 
    if frame_cur > frame_max:
        frame_cur = frame_min

# HDF5データ読み込み

filename = "starling-20141026s2f8.h5"
with h5py.File(filename, 'r') as h5f:
    data = {}
    for key in h5f.keys():
        data[key] = h5f[key][:]

# 頂点リスト生成
vertices = np.column_stack((data['x'], data['z'], -data['y'])).ravel()
nvertex = len(vertices)//3

frames = data['frame']
        
vertex_list = shader0.vertex_list(len(vertices)//3, GL_POINTS, batch=batch0)
vertex_list.position[:] = vertices 
vertex_list.colors[:] =  [1,1,1,1] * nvertex
vertex_list.frames[:] = frames

vertex_grid = gen_grid(-30,40, 0,30, -100,0, 10, shader1, batch1)

frame_max = max(frames)
frame_min = min(frames)
frame_cur = frame_min

view_matrix = Mat4.look_at(position=Vec3(0,50,-50), target=Vec3(0,0,-50), up=Vec3(0,0,1))
window.view = view_matrix

on_resize(*window.size)

shader0['frame'] = int(frame_cur)

pyglet.clock.schedule_interval(update,1/30)
pyglet.app.run()

