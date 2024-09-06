import numpy as np
import math
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3, Vec4
from pyglet.graphics.shader import Shader, ShaderProgram

# シェーダー
# Vertex shader
vertex_source = """#version 330 core
    in vec3 position;
    in vec3 normals;
    in vec4 diffuse_colors;
    in vec4 ambient_colors;
    in vec4 specular_colors;
    in vec4 emission_colors;
    in float shininess;
    in vec2 tex_coords;

    out vec4 vertex_diffuse;
    out vec4 vertex_ambient;
    out vec4 vertex_specular;
    out vec4 vertex_emission;
    out float vertex_shininess;
    out vec3 vertex_normals;
    out vec3 vertex_position;
    out vec2 texture_coords;

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
        vertex_diffuse = diffuse_colors;
        vertex_ambient = ambient_colors;
        vertex_specular = specular_colors;
        vertex_emission = emission_colors;
        vertex_shininess = shininess;
        vertex_normals = normal_matrix * normals;
        texture_coords = tex_coords;
    }
"""

fragment_source = """#version 330 core
    in vec4 vertex_diffuse;
    in vec4 vertex_ambient;
    in vec4 vertex_specular;
    in vec4 vertex_emission;
    in float vertex_shininess;
    in vec3 vertex_normals;
    in vec3 vertex_position;
    in vec2 texture_coords; 
    out vec4 final_colors;

    uniform vec3 light_position;
    uniform vec4 light_color;

    uniform sampler2D texture_2d;

    void main()
    {
        vec3 normal = normalize(vertex_normals);
        vec3 light_dir = normalize(light_position - vertex_position);
        vec3 refrect_dir = normalize(-light_dir + 2 * dot(light_dir, normal) * normal);
        // vec3 refrect_dir = reflect(-light_dir, normal);
        vec3 view_dir = -normalize(vertex_position);
        float spec = pow(max(dot(view_dir, refrect_dir), 0.0), vertex_shininess);
        float diff = max(dot(normal, light_dir), 0.0);

        final_colors = vertex_ambient * light_color
                     + (texture(texture_2d, texture_coords)) * vertex_diffuse * diff
                     + vertex_specular * spec * light_color
                     + vertex_emission ;
    }
"""
    
window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Polyhedrons')

batch = pyglet.graphics.Batch()
vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
shader = ShaderProgram(vert_shader, frag_shader)            

time=0
def update(dt):
    global time
    time += dt*0.3

@window.event
def on_draw():
    window.clear()
    rot_mat = Mat4.from_rotation(time,Vec3(0,1,0)) 
    model_jupiter.matrix = Mat4.from_translation(Vec3(0,0,0)) @ rot_mat
    shader['light_position']=Vec3(-20,0,30)
    shader['light_color']=Vec4(1.0, 1.0, 1.0, 1.0)
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

def load_texed_model_from_obj_file(filename, shader, batch, has_normal=False):
    file = open(filename,"rb")    
    mesh_list = pyglet.model.codecs.obj.parse_obj_file(filename=filename, file=file)
    vertex_lists=[ ]
    groups=[ ]
    for mesh in mesh_list:
        material = mesh.material
        img = pyglet.image.load(material.texture_name)
        texture = img.get_texture()
        count = len(mesh.vertices) // 3
        group = pyglet.model.TexturedMaterialGroup(material=material, program=shader, texture=texture, order=0)
        normals=[ ]        
        if has_normal:
            normals[:] = mesh.normals
        else:
            for k in range(0,count,3):
                dx0 = mesh.vertices[(k+1)*3+0] - mesh.vertices[k*3+0]
                dy0 = mesh.vertices[(k+1)*3+1] - mesh.vertices[k*3+1]
                dz0 = mesh.vertices[(k+1)*3+2] - mesh.vertices[k*3+2]
                dx1 = mesh.vertices[(k+2)*3+0] - mesh.vertices[k*3+0]
                dy1 = mesh.vertices[(k+2)*3+1] - mesh.vertices[k*3+1]
                dz1 = mesh.vertices[(k+2)*3+2] - mesh.vertices[k*3+2]
                nx = dy0*dz1 - dz0*dy1
                ny = dz0*dx1 - dx0*dz1
                nz = dx0*dy1 - dy0*dx1
                n = math.sqrt(nx*nx + ny*ny + nz*nz)
                if n>0:
                    nx /= n
                    ny /= n
                    nz /= n
                normals.extend([nx,ny,nz,nx,ny,nz,nx,ny,nz])
                
        vertex_list = shader.vertex_list(count, GL_TRIANGLES, batch=batch, group=group, \
                                         position=('f', mesh.vertices), \
                                         normals = ('f', normals), \
                                         tex_coords = ('f', mesh.tex_coords) )
        vertex_list.diffuse_colors[:] = material.diffuse * count
        vertex_list.ambient_colors[:] = material.ambient * count
        vertex_list.specular_colors[:] = material.specular * count
        vertex_list.emission_colors[:] = material.emission * count
        vertex_list.shininess[:] = [material.shininess] * count
        vertex_lists.append(vertex_list)
        groups.append(group)
        
    return  pyglet.model.Model(vertex_lists=vertex_lists, groups=groups, batch=batch)

# OpenGLの初期設定
setup()

# マテリアルデータの読み込み
model_jupiter = load_texed_model_from_obj_file(filename="../data/jupiter.obj", shader=shader, batch=batch, has_normal=True)

# 視点を設定
window.view = Mat4.look_at(position=Vec3(0,0,3), target=Vec3(0,0,0), up=Vec3(0,1,0))

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()
