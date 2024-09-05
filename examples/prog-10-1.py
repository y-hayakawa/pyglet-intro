import numpy as np
import math
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.graphics.shader import Shader, ShaderProgram
import pandas as pd

# 星の表示用シェーダー
vertex_source_0 = """#version 330 core
    in vec3 position;
    in vec4 colors;
    in float absmags;

    out vec4 vertex_colors;
    out vec3 vertex_position;
    out float vertex_brightness;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model;

#define BFACT 4000

    void main()
    {
        mat4 modelview = window.view * model;
        vec4 pos = modelview * vec4(position, 1.0);
        gl_Position = window.projection * pos;
        float dist = distance(vec3(0.0, 0.0, 0.0), pos.xyz) ;
        float brightness = exp(-absmags*0.921) / (dist*dist) * BFACT ;
        gl_PointSize = log(brightness+1.0)+2.0 ;
        if (brightness>1.0) brightness=1.0 ;
        vertex_position = pos.xyz;
        vertex_colors = colors;
        vertex_brightness = brightness;
    }
"""

fragment_source_0 = """#version 330 core
    in vec4 vertex_colors;
    in vec3 vertex_position;
    in float vertex_brightness;
    out vec4 final_colors;

    void main()
    {
       vec2 pos = gl_PointCoord * 2.0 - 1.0; 
       float r = 1.0 - dot(pos, pos); 
       if (r < 0.0) discard; 
       final_colors = vertex_colors * vertex_brightness ;
    }
"""

# ワイヤーフレーム表示用シェーダー
vertex_source_1 = """#version 330 core
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

fragment_source_1 = """#version 330 core
    in vec4 vertex_colors;
    in vec3 vertex_position;
    out vec4 final_colors;

    void main()
    {
        final_colors = vertex_colors ;
    }
"""

window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Star map')

# シェーダーのコンパイル
batch0 = pyglet.graphics.Batch()
vert_shader_0 = Shader(vertex_source_0, 'vertex')
frag_shader_0 = Shader(fragment_source_0, 'fragment')
shader0 = ShaderProgram(vert_shader_0, frag_shader_0)

batch1 = pyglet.graphics.Batch()
vert_shader_1 = Shader(vertex_source_1, 'vertex')
frag_shader_1 = Shader(fragment_source_1, 'fragment')
shader1 = ShaderProgram(vert_shader_1, frag_shader_1)

@window.event
def on_draw():
    window.clear()
    shader0['model']=Mat4()
    shader1['model']=Mat4() # 赤道座標グリッド
    batch1.draw()    
    batch0.draw()

fov = 60
@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    ratio = width/height
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.01, z_far=100000, fov=fov)
    return pyglet.event.EVENT_HANDLED

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global fov
    fov += scroll_y
    if fov < 1:
        fov = 1
    elif fov>90:
        fov = 90
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.01, z_far=10000, fov=fov)
    return pyglet.event.EVENT_HANDLED

@window.event
def on_mouse_drag(x,y,dx,dy,buttons,modifiers):
    global view_matrix
    view_matrix = Mat4.from_rotation(-dx*0.005, Vec3(0, 1, 0))  @ view_matrix
    view_matrix = Mat4.from_rotation(dy*0.005, Vec3(1, 0, 0)) @ view_matrix
    window.view = view_matrix
    return pyglet.event.EVENT_HANDLED

@window.event
def on_key_press(symbol, modifiers):
    global view_matrix    
    if symbol == pyglet.window.key.R:
        view_matrix = Mat4.look_at(position=Vec3(0,0,0), target=Vec3(100000,0,0), up=Vec3(0,1,0))
        window.view = view_matrix
    elif symbol == pyglet.window.key.U:
        d = view_matrix.row(2)
        view_matrix = Mat4.look_at(position=Vec3(0,0,0), target=Vec3(-d[0]*100000,-d[1]*100000,-d[2]*100000), up=Vec3(0,1,0))
        window.view = view_matrix
    return pyglet.event.EVENT_HANDLED

def setup():
    glClearColor(0.01, 0.01, 0.05, 1.0)
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
    on_resize(*window.size)

###
# this part was taken from the answer by DocLeonard in
# https://stackoverflow.com/questions/21977786/star-b-v-color-index-to-apparent-rgb-color
redco = [ 1.62098281e-82, -5.03110845e-77, 6.66758278e-72, -4.71441850e-67, 1.66429493e-62, -1.50701672e-59, -2.42533006e-53,
          8.42586475e-49, 7.94816523e-45, -1.68655179e-39, 7.25404556e-35, -1.85559350e-30, 3.23793430e-26, -4.00670131e-22,
          3.53445102e-18, -2.19200432e-14, 9.27939743e-11, -2.56131914e-07,  4.29917840e-04, -3.88866019e-01, 3.97307766e+02]
greenco = [ 1.21775217e-82, -3.79265302e-77, 5.04300808e-72, -3.57741292e-67, 1.26763387e-62, -1.28724846e-59, -1.84618419e-53,
            6.43113038e-49, 6.05135293e-45, -1.28642374e-39, 5.52273817e-35, -1.40682723e-30, 2.43659251e-26, -2.97762151e-22,
            2.57295370e-18, -1.54137817e-14, 6.14141996e-11, -1.50922703e-07,  1.90667190e-04, -1.23973583e-02,-1.33464366e+01]
blueco = [ 2.17374683e-82, -6.82574350e-77, 9.17262316e-72, -6.60390151e-67, 2.40324203e-62, -5.77694976e-59, -3.42234361e-53,
           1.26662864e-48, 8.75794575e-45, -2.45089758e-39, 1.10698770e-34, -2.95752654e-30, 5.41656027e-26, -7.10396545e-22,
           6.74083578e-18, -4.59335728e-14, 2.20051751e-10, -7.14068799e-07,  1.46622559e-03, -1.60740964e+00, 6.85200095e+02]

redco = np.poly1d(redco)
greenco = np.poly1d(greenco)
blueco = np.poly1d(blueco)

def temp2rgb(temp):
    red = redco(temp)
    green = greenco(temp)
    blue = blueco(temp)
    if red > 255:
        red = 255
    elif red < 0:
        red = 0
    if green > 255:
        green = 255
    elif green < 0:
        green = 0
    if blue > 255:
        blue = 255
    elif blue < 0:
        blue = 0

    return (red/255, green/255, blue/255)
###

def bv2rgb(bv):
    t = 4600*(1/(0.92*bv + 1.7) + 1/(0.92*bv+0.62))
    r,g,b = temp2rgb(t)
    return (r,g,b)


def gen_stars(filename, shader, batch):
    print("reading",filename)
    df = pd.read_csv(filename)
    print("generating vertecies...")
    vertices = []
    absmags = []
    colors = []
    for index, row in df.iterrows():
        vertices.extend([row['x'],row['z'],-row['y']])
        absmags.append(row['absmag'])
        if row['ci']:
            r,g,b = bv2rgb(row['ci'])
            colors.extend([r,g,b,1.0])
        else:
            colors.extend([0,0,0,0])
        
    vertex_list = shader.vertex_list(len(vertices)//3, GL_POINTS, batch=batch)
    vertex_list.position[:] = vertices 
    vertex_list.absmags[:] = absmags
    vertex_list.colors[:] = colors

    return vertex_list

def gen_sphere(radius,stacks,slices,shader, batch):
    vertices = []
    for i in range(stacks + 1):
        phi = math.pi / 2 - i * math.pi / stacks
        y = radius * math.sin(phi) 
        r = radius * math.cos(phi)
        for j in range(slices + 1):
            theta = j * 2 * math.pi / slices
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            vertices.extend([x, y, z])

    indices = []
    for i in range(stacks):
        for j in range(slices):
            p1 = i * (slices+1) + j
            p2 = p1 + (slices+1)
            indices.extend([p1, p2])
            indices.extend([p1, p1+1])            

    vertex_list = shader.vertex_list_indexed(len(vertices)//3, GL_LINES, indices, batch=batch,
                                             position=('f', vertices),
                                             colors =('f', [0.15,0.15,0.2,1]*(len(vertices)//3)))
    return vertex_list

# OpenGLの初期設定
setup()

# CSV
filename = "hygdata_v41.csv"
vertex_list0 = gen_stars(filename,shader0,batch0)
vertex_list1 = gen_sphere(100,12,24,shader1,batch1)

# 視点を設定
view_matrix = Mat4.look_at(position=Vec3(0,0,0), target=Vec3(100000,0,0), up=Vec3(0,1,0))
window.view = view_matrix

pyglet.app.run()
