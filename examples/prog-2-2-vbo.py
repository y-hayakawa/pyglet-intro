import pyglet
from pyglet.gl import *
from pyglet.math import Mat4
import numpy as np

vertex_source = """#version 330 core
    layout(location=0) in vec2 position;
    layout(location=1) in vec4 colors;
    out vec4 vertex_colors;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    uniform mat4 model ;

    void main()
    {
        gl_Position = window.projection * window.view * model * vec4(position, 0.0, 1.0) ;
        vertex_colors = colors;
    }
"""

fragment_source = """#version 330 core
    in vec4 vertex_colors;
    out vec4 final_colors;

    void main()
    {
        final_colors = vertex_colors;
    }
"""


window = pyglet.window.Window(width=400,height=300)
glClearColor(0.3, 0.3, 0.5, 1.0)

# シェーダーをコンパイル
vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
shader = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)

# 頂点座標(position)用のVBOを作成
vbo0 = GLuint()
glGenBuffers(1,vbo0)
glBindBuffer(GL_ARRAY_BUFFER, vbo0)
vertices = np.array([-0.9, -0.9, 0.9, -0.9, 0.0, 0.9], dtype='float32')
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, GL_DYNAMIC_DRAW)

# 色情報(colors)用のVBOを作成
vbo1 = GLuint()
glGenBuffers(1,vbo1)
glBindBuffer(GL_ARRAY_BUFFER, vbo1)
colors = np.array([1.0, 0.0, 0.0, 1.0,  0.0, 1.0, 0.0, 1.0,  0.0, 0.0, 1.0, 1.0], dtype='float32')
glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors.ctypes.data, GL_DYNAMIC_DRAW)

# VAOを作成
vao = GLuint()
glGenVertexArrays(1, vao)
glBindVertexArray(vao)

glBindBuffer(GL_ARRAY_BUFFER, vbo0)
glEnableVertexAttribArray(0)  # layout(location=0) in vec2 position; に対応
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0 , 0)

glBindBuffer(GL_ARRAY_BUFFER, vbo1)
glEnableVertexAttribArray(1)  # layout(location=1) in vec4 colors; に対応
glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0)

# 描画イベントハンドラー
@window.event
def on_draw():
    window.clear()
    shader.use() # 使用するシェダープログラムを設定
    window.viewport = (100, 100, 200, 100) # (0, 0, 400, 300)
    window.projection = Mat4()
    window.view = Mat4()
    shader['model'] = Mat4()
    # VAOに基づいて描画（GL_TRIANGLESを配列の0番目から頂点3つ分）
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glBindVertexArray(0)

# イベントループ開始
pyglet.app.run()
