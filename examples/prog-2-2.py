import pyglet
from pyglet.gl import *
from pyglet.math import Mat4

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


window = pyglet.window.Window(width=400,height=300)
glClearColor(0.3, 0.3, 0.5, 1.0)

vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
shader = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
batch = pyglet.graphics.Batch()

vlist = shader.vertex_list(3, GL_TRIANGLES, batch=batch)


@window.event
def on_draw():
    window.clear()
    window.viewport = (100, 100, 200, 100) # (0, 0, 400, 300)
    
    window.projection = Mat4()
    window.view = Mat4()
    shader['model'] = Mat4()
    vlist.position = (-0.9, -0.9, 0.9, -0.9, 0.0, 0.9)
    vlist.colors = (1.0, 0.0, 0.0, 1.0,  0.0, 1.0, 0.0, 1.0,  0.0, 0.0, 1.0, 1.0)
    batch.draw()


pyglet.app.run()
