import numpy as np
import math
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.graphics.shader import Shader, ShaderProgram

# シェーダー
# Vertex shader
vertex_source = """#version 330 core
    layout(location=0) in vec3 position;
    layout(location=1) in vec3 normals;
    layout(location=2) in vec4 colors;

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

vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
shader = ShaderProgram(vert_shader, frag_shader)

# 正四面体の頂点座標、法線ベクトル、色を計算
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
colors = (1.0,0.0,0.0,1.0) * (len(vertices)//3)

# 頂点座標(position)用のVBOを作成
vbo0 = GLuint()
glGenBuffers(1,vbo0)
glBindBuffer(GL_ARRAY_BUFFER, vbo0)
vertices = np.array(vertices, dtype='float32')
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, GL_DYNAMIC_DRAW)

# 法線ベクトル(normals)用のVBOを作成
vbo1 = GLuint()
glGenBuffers(1,vbo1)
glBindBuffer(GL_ARRAY_BUFFER, vbo1)
normals = np.array(normals, dtype='float32')
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals.ctypes.data, GL_DYNAMIC_DRAW)

# 色情報(colors)用のVBOを作成
vbo2 = GLuint()
glGenBuffers(1,vbo2)
glBindBuffer(GL_ARRAY_BUFFER, vbo2)
colors = np.array(colors, dtype='float32')
glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors.ctypes.data, GL_DYNAMIC_DRAW)

# VAOを作成
vao = GLuint()
glGenVertexArrays(1, vao)
glBindVertexArray(vao)

glBindBuffer(GL_ARRAY_BUFFER, vbo0)
glEnableVertexAttribArray(0)  # layout(location=0) in vec3 position; に対応
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0 , 0)

glBindBuffer(GL_ARRAY_BUFFER, vbo1)
glEnableVertexAttribArray(1)  # layout(location=1) in vec3 normals; に対応
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0)

glBindBuffer(GL_ARRAY_BUFFER, vbo2)
glEnableVertexAttribArray(2)  # layout(location=2) in vec4 colors; に対応
glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0)


time=0
def update(dt):
    global time
    time += dt

@window.event
def on_draw():
    window.clear()
    shader.use()  # 使用するシェーダーを選択
    shader['model']=Mat4.from_rotation(time,Vec3(1,0,0)) @ Mat4.from_rotation(time/3,Vec3(0,1,0))
    shader['light_position']=Vec3(-5,5,20)
    # VAOに基づいて描画（GL_TRIANGLESを配列の0番目から）
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices)//3)
    glBindVertexArray(0)

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

# 視点を設定
window.view = Mat4.look_at(position=Vec3(0,0,3), target=Vec3(0,0,0), up=Vec3(0,1,0))

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()
