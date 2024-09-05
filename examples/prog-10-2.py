
# Vertex shader
vertex_source = """#version 330 core

in vec3 positions;
in vec4 colors;
in float radius;

out VS_OUT {
    vec3 positions;
    vec4 colors;
    float radius;
} vs_out;

void main()
  {
    vs_out.positions = positions;
    vs_out.colors = colors ;
    vs_out.radius = radius ;
  }
"""

# Geometory shader
geometry_source = """#version 330 core

layout(lines) in;
layout(triangle_strip, max_vertices = 200) out;

in VS_OUT {
    vec3 positions;
    vec4 colors;
    float radius;
} gs_in[];

out vec3 geom_pos;
out vec3 geom_normal;
out vec4 geom_color;

uniform mat4 model;

uniform WindowBlock
{
  mat4 projection;
  mat4 view;
} window;

#define N 6
#define TWO_PI 6.28318530718

void main()
{
    mat4 modelview = window.view * model ;
    mat3 normal_matrix = transpose(inverse(mat3(modelview)));
    vec4 world_position ;
    vec4 view_position ;

    vec3 pos1 = gs_in[0].positions;
    float r1 = gs_in[0].radius;
    vec4 color1 = gs_in[0].colors; 
    vec3 pos2 = gs_in[1].positions;
    float r2 = gs_in[1].radius;
    vec4 color2 = gs_in[1].colors; 
    vec3 u = normalize(pos2 - pos1) ;
    vec3 v1,v2 ;
    if (abs(u.z - 1) > 1e-6) {
       v1 = normalize(vec3(u.y, -u.x, 0)) ;
       v2 = cross(u,v1) ;
    } else {
       v1 = vec3(1,0,0) ;
       v2 = vec3(0,1,0) ;
    }

    for (int i=0; i<N; i++) {
       float angle1 = TWO_PI * i/N ;
       float angle2 = TWO_PI * (i+1)/N;
       vec3 dr1 = cos(angle1) * v1 + sin(angle1) * v2;
       vec3 dr2 = cos(angle2) * v1 + sin(angle2) * v2;
       // Triangle 1
       world_position = vec4(pos1+dr1*r1, 1.0) ;
       view_position =  modelview * world_position ;
       gl_Position = window.projection * view_position ;
       geom_pos = view_position.xyz;
       geom_normal = normal_matrix * dr1 ;
       geom_color = color1 ;
       EmitVertex();

       world_position = vec4(pos1+dr2*r1, 1.0) ;
       view_position =  modelview * world_position ;
       gl_Position = window.projection * view_position ;
       geom_pos = view_position.xyz;
       geom_normal = normal_matrix * dr2 ;
       geom_color = color1 ;
       EmitVertex();

       world_position = vec4(pos2+dr2*r2, 1.0) ;
       view_position =  modelview * world_position ;
       gl_Position = window.projection * view_position ;
       geom_pos = view_position.xyz;
       geom_normal = normal_matrix * dr2 ;
       geom_color = color2 ;
       EmitVertex();

       EndPrimitive();

       // Triangle 2
       world_position = vec4(pos1+dr1*r1, 1.0) ;
       view_position =  modelview * world_position ;
       gl_Position = window.projection * view_position ;
       geom_pos = view_position.xyz;
       geom_normal = normal_matrix * dr1 ;
       geom_color = color1 ;
       EmitVertex();

       world_position = vec4(pos2+dr2*r2, 1.0) ;
       view_position =  modelview * world_position ;
       gl_Position = window.projection * view_position ;
       geom_pos = view_position.xyz;
       geom_normal = normal_matrix * dr2 ;
       geom_color = color2 ;
       EmitVertex();

       world_position = vec4(pos2+dr1*r2, 1.0) ;
       view_position =  modelview * world_position ;
       gl_Position = window.projection * view_position ;
       geom_pos = view_position.xyz;
       geom_normal = normal_matrix * dr1 ;
       geom_color = color2 ;
       EmitVertex();

       EndPrimitive();
    }
}
"""

# Fragment shader
fragment_source = """#version 330 core

in vec3 geom_pos;
in vec3 geom_normal;
in vec4 geom_color;

out vec4 frag_color;

uniform vec3 light_position ;

void main()
{
    vec3 normal = normalize(geom_normal);
    vec3 light_dir = normalize(light_position - geom_pos);
    float diff = max(dot(normal, light_dir), 0.0);
    frag_color = geom_color * diff ;
}
"""

import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3
from pyglet.graphics.shader import Shader, ShaderProgram


window = pyglet.window.Window(width=1280, height=720, resizable=True)

    
@window.event
def on_draw():
    window.clear()
    shader['light_position'] = Vec3(50,200,200)
    batch.draw()

@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.1, z_far=255, fov=60)
    return pyglet.event.EVENT_HANDLED


def update(dt):
    global time
    time += dt
    rot_x = Mat4.from_rotation(time/3, Vec3(1, 0, 0))
    rot_y = Mat4.from_rotation(time/7, Vec3(0, 1, 0))
    rot_z = Mat4.from_rotation(time/11, Vec3(0, 0, 1))
    trans = Mat4.from_translation(Vec3(0, 0, 0))
    shader['model'] = trans @ rot_z @ rot_y @ rot_x ;


def setup():
    # One-time GL setup
    glClearColor(0.2, 0.2, 0.3, 1)
    glEnable(GL_DEPTH_TEST)
    
    on_resize(*window.size)
    # Uncomment this line for a wireframe view:
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)


setup()

time = 0.0
batch = pyglet.graphics.Batch()
group = pyglet.graphics.Group(order=0)

vert_shader = Shader(vertex_source, 'vertex')
geom_shader = Shader(geometry_source, 'geometry')
frag_shader = Shader(fragment_source, 'fragment')
shader = ShaderProgram(vert_shader, geom_shader, frag_shader)

vertices = [ ]
x = 0
y = 0
z = 0
for i in range(2000):
    vertices.extend([x,y,z])
    x += np.random.normal(loc=0,scale=1)
    y += np.random.normal(loc=0,scale=1)
    z += np.random.normal(loc=0,scale=1)    

lines = shader.vertex_list(len(vertices)//3, GL_LINE_STRIP,
                           positions=('f', vertices),
                           colors = ('f', [1.0, 0.0, 0.0, 1.0] * (len(vertices)//3)),
                           radius = ('f', [0.15]*(len(vertices)//3)),
                           batch=batch)

window.view = Mat4.look_at(position=Vec3(0,0,25), target=Vec3(0,0,0), up=Vec3(0,1,0))

pyglet.clock.schedule_interval(update, 1/60)
pyglet.app.run()
