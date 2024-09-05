import numpy as np
import math
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.graphics.shader import Shader, ShaderProgram

# シェーダー
# Vertex shader
vertex_source = """#version 330 core
    in vec3 position;
    in vec3 normals;
    in vec4 colors;

    out vec4 vertex_colors;
    out vec3 vertex_normals;
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
        mat3 normal_matrix = transpose(inverse(mat3(modelview)));
        vertex_position = pos.xyz;
        vertex_colors = colors;
        vertex_normals = normal_matrix * normals;
    }
"""

fragment_source = """#version 330 core
    in vec4 vertex_colors;
    in vec3 vertex_normals;
    in vec3 vertex_position;
    out vec4 final_colors;

    uniform vec3 light_position;

    void main()
    {
        vec3 normal = normalize(vertex_normals);
        vec3 light_dir = normalize(light_position - vertex_position);
        float diff = max(dot(normal, light_dir), 0.0);
        final_colors = vertex_colors * diff * 1.2;
    }
"""

window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Tetrahedron')

batch = pyglet.graphics.Batch()
vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
shader = ShaderProgram(vert_shader, frag_shader)

@window.event
def on_draw():
    window.clear()
    shader['model']=Mat4()
    shader['light_position']=Vec3(-10,0,20)
    batch.draw()

@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    ratio = width/height
    window.projection = Mat4.orthogonal_projection(-2*ratio, 2*ratio, -2, 2, -100, 100)
    return pyglet.event.EVENT_HANDLED

def setup():
    glClearColor(0.3, 0.3, 0.5, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    on_resize(*window.size)

def tetrahedron(shader, batch):
    vpos = [np.array((+1,0,-1/math.sqrt(2))), np.array((-1,0,-1/math.sqrt(2))),
            np.array((0,1,1/math.sqrt(2))), np.array((0,-1,1/math.sqrt(2)))]
    indices = [(0,1,2),(0,3,1),(0,2,3),(1,3,2)]    
    normals = []
    vertices = []

    for n in range(4):
        i = indices[n][0]
        j = indices[n][1]
        k = indices[n][2]
        vertices.extend([vpos[i],vpos[j],vpos[k]])
        u = vpos[j] - vpos[i]
        v = vpos[k] - vpos[i]
        n = np.cross(u,v)
        n = n/np.linalg.norm(n)
        normals.extend([n,n,n])

    vertices = np.concatenate(vertices).tolist()
    normals = np.concatenate(normals).tolist()
        
    vertex_list = shader.vertex_list(len(vertices)//3, GL_TRIANGLES, batch=batch)
    vertex_list.position[:] = vertices 
    vertex_list.normals[:] = normals
    vertex_list.colors[:] = (1.0,0.0,0.0,1.0) * (len(vertices)//3)

    return vertex_list

# OpenGLの初期設定
setup()

# 正四面体モデルを生成
th_vertex_list = tetrahedron(shader,batch)

# 視点を設定
window.view = Mat4.look_at(position=Vec3(0,0,5), target=Vec3(0,0,0), up=Vec3(0,1,0))

pyglet.app.run()
