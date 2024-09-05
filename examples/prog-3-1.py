import pyglet
from pyglet.gl import *
from pyglet.math import Mat4,Vec3

# シェダーの定義
vertex_source = """#version 330 core
    in vec2 position;
    in vec4 colors;
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

# ウィンドウの生成
window = pyglet.window.Window(width=400,height=300)
glClearColor(0.3, 0.3, 0.5, 1.0)

# シェダーのコンパイル
vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
shader = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
batch = pyglet.graphics.Batch()

# 頂点リストの生成
vlist = shader.vertex_list(3, GL_TRIANGLES, batch=batch)

rot_angle = 0.0

# 描画用イベントハンドラーの定義
@window.event
def on_draw():
    window.clear()
    window.viewport = (0, 0, 400, 300)
    window.projection = Mat4.orthogonal_projection(-4, 4, -3, 3, -100, 100)
    window.view = Mat4()
    shader['model'] = Mat4.from_rotation(rot_angle,Vec3(0,0,1)) @ Mat4.from_translation(Vec3(2,0,0))
    vlist.position = (-0.9, -0.9, 0.9, -0.9, 0.0, 0.9)
    vlist.colors = (1.0, 0.0, 0.0, 1.0,  0.0, 1.0, 0.0, 1.0,  0.0, 0.0, 1.0, 1.0)
    batch.draw()

# 時間毎の更新
def update(dt):
    global rot_angle
    rot_angle += 3.1415/30

# タイマー設定
pyglet.clock.schedule_interval(update, 1/30)

# イベントループ開始
pyglet.app.run()

