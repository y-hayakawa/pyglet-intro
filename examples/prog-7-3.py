# prog-7-3.py
# https://wagtail.cds.tohoku.ac.jp/coda/python/pyglet/pyglet-part3.html
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
    in vec3 tangents;
    in vec4 colors;
    in vec2 tex_coords;

    out vec4 vertex_colors;
    out vec3 vertex_normals;
    out vec3 vertex_tangents;
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
        vertex_colors = colors;
        vertex_normals = normal_matrix * normals;
        vertex_tangents = normal_matrix * tangents;
        texture_coords = tex_coords;
    }
"""

fragment_source = """#version 330 core
    in vec4 vertex_colors;
    in vec3 vertex_normals;
    in vec3 vertex_tangents;
    in vec3 vertex_position;
    in vec2 texture_coords;
    out vec4 final_colors;

    uniform vec3 light_position;
    uniform sampler2D texture_0;
    uniform sampler2D texture_1;

    void main()
    {
        vec3 normal = normalize(vertex_normals);
        vec3 bitangent = normalize(cross(normal, normalize(vertex_tangents)));
        mat3 tbn_mat = mat3(vertex_tangents, bitangent, normal);
        vec3 tex_normal = normalize(texture(texture_1, texture_coords).xyz - 0.5) ;
        vec3 delta_normal = tbn_mat * tex_normal;
        vec3 light_dir = normalize(light_position - vertex_position);
        float diff = max(dot(delta_normal, light_dir), 0.0);
        final_colors = (texture(texture_0, texture_coords) * vertex_colors) * diff ;
    }
"""


class TwoTexturedMaterialGroup(pyglet.model.TexturedMaterialGroup):
    def __init__(self, material: pyglet.model.Material, program: pyglet.graphics.shader.ShaderProgram, \
                 texture0: pyglet.image.Texture, texture1: pyglet.image.Texture, \
                 order: int = 0, parent: pyglet.graphics.Group | None = None):
        super().__init__(material, program, texture0, order, parent)
        self.texture0 = texture0
        self.texture1 = texture1

    def set_state(self) -> None:
        self.program['texture_0'] = 0
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(self.texture0.target, self.texture0.id)
        self.program['texture_1'] = 1
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(self.texture1.target, self.texture1.id)        
        self.program.use()
        self.program['model'] = self.matrix
        
    def __hash__(self) -> int:
        return hash((self.texture0.target, self.texture0.id, self.program, self.order, self.parent))

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.material == other.material and
                self.matrix == other.matrix and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent)



window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Tetrahedron')

batch = pyglet.graphics.Batch()
vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
shader = ShaderProgram(vert_shader, frag_shader)

brick_wall = pyglet.image.load('brick-wall.png')
texture0 = brick_wall.get_texture()

normalmap = pyglet.image.load('brick-wall-normalmap.png')
texture1 = normalmap.get_texture()

time=0
def update(dt):
    global time
    time += dt

@window.event
def on_draw():
    window.clear()
    th_model.matrix = Mat4.from_rotation(time,Vec3(1,0,0)) @ Mat4.from_rotation(time/3,Vec3(0,1,0))
    shader['light_position']=Vec3(-5,5,20)
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

def tetrahedron(shader, texture0, texture1, batch):
    vpos = [np.array((+1,0,-1/math.sqrt(2))), np.array((-1,0,-1/math.sqrt(2))),
            np.array((0,1,1/math.sqrt(2))), np.array((0,-1,1/math.sqrt(2)))]
    indices = [(0,1,2),(0,3,1),(0,2,3),(1,3,2)]    
    normals = []
    tangents = []    
    vertices = []
    tex_coords = []

    for n in range(4):
        i = indices[n][0]
        j = indices[n][1]
        k = indices[n][2]
        vertices.extend([vpos[i],vpos[j],vpos[k]])
        tangents.extend([vpos[j]-vpos[i], vpos[j]-vpos[i], vpos[j]-vpos[i]])        
        u = vpos[j] - vpos[i]
        v = vpos[k] - vpos[i]
        n = np.cross(u,v)
        n = n/np.linalg.norm(n)
        normals.extend([n,n,n])
        tex_coords.extend([0, 0, 1, 0, 0.5, math.sqrt(3)/2])

    vertices = np.concatenate(vertices).tolist()
    normals = np.concatenate(normals).tolist()
    tangents = np.concatenate(tangents).tolist()    

    diffuse = [1.0, 1.0, 1.0, 1.0]
    ambient = [1.0, 1.0, 0.0, 1.0]
    specular = [1.0, 1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 50

    material = pyglet.model.Material("custom", diffuse, ambient, specular, emission, shininess)
    group = TwoTexturedMaterialGroup(material=material, program=shader, \
                                     texture0=texture0, texture1=texture1)
        
    vertex_list = shader.vertex_list(len(vertices)//3, GL_TRIANGLES, batch=batch, group=group)
    vertex_list.position[:] = vertices 
    vertex_list.normals[:] = normals
    vertex_list.tangents[:] = tangents
    vertex_list.colors[:] = material.diffuse * (len(vertices)//3)
    vertex_list.tex_coords[:] = tex_coords

    return pyglet.model.Model(vertex_lists=[vertex_list], groups=[group], batch=batch)

# OpenGLの初期設定
setup()

# 正四面体モデルを生成
th_model = tetrahedron(shader,texture0,texture1,batch)

# 視点を設定
window.view = Mat4.look_at(position=Vec3(0,0,3), target=Vec3(0,0,0), up=Vec3(0,1,0))

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()
