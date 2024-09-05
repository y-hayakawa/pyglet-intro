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

    out vec4 vertex_diffuse;
    out vec4 vertex_ambient;
    out vec4 vertex_specular;
    out vec4 vertex_emission;
    out float vertex_shininess;
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
        vertex_diffuse = diffuse_colors;
        vertex_ambient = ambient_colors;
        vertex_specular = specular_colors;
        vertex_emission = emission_colors;
        vertex_shininess = shininess;
        vertex_normals = normal_matrix * normals;
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
    out vec4 final_colors;

    uniform vec3 light_position;
    uniform vec4 light_color;

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
                     + vertex_diffuse * diff
                     + vertex_specular * spec * light_color
                     + vertex_emission ;
    }
"""

class MyMaterialGroup(pyglet.model.BaseMaterialGroup):
    def __init__(self, material:pyglet.model.Material, program: pyglet.graphics.shader.ShaderProgram, \
                 order:int=0, parent: pyglet.graphics.Group | None = None):
        super().__init__(material, program, order, parent)
        
    def set_state(self) -> None:
        self.program.use()
        self.program['model'] = self.matrix
        
    def __hash__(self) -> int:
        return hash((self.program, self.order, self.parent))
    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.material == other.material and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent)

    
window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Tetrahedron')

batch = pyglet.graphics.Batch()
vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
shader = ShaderProgram(vert_shader, frag_shader)

time=0
def update(dt):
    global time
    time += dt*0.5

@window.event
def on_draw():
    window.clear()
    th_model.matrix = Mat4.from_rotation(time,Vec3(0,1,0)) @ Mat4.from_rotation(time/2,Vec3(0,0,1)) @ Mat4.from_rotation(time/3,Vec3(1,0,0))
    shader['light_position']=Vec3(-200,100,200)
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

def tetrahedron(shader, diffuse, ambient, specular, emission, shininess, batch):
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

    material = pyglet.model.Material("custom", diffuse, ambient, specular, emission, shininess)
    group = MyMaterialGroup(material=material, program=shader)
        
    vertex_list = shader.vertex_list(len(vertices)//3, GL_TRIANGLES, batch=batch, group=group)
    vertex_list.position[:] = vertices 
    vertex_list.normals[:] = normals
    vertex_list.diffuse_colors[:] = material.diffuse * (len(vertices)//3)
    vertex_list.ambient_colors[:] = material.ambient * (len(vertices)//3)
    vertex_list.specular_colors[:] = material.specular * (len(vertices)//3)
    vertex_list.emission_colors[:] = material.emission * (len(vertices)//3)
    vertex_list.shininess[:] = [material.shininess] * (len(vertices)//3)        

    return pyglet.model.Model(vertex_lists=[vertex_list], groups=[group], batch=batch)

# OpenGLの初期設定
setup()

# 正四面体モデルを生成
th_model = tetrahedron(shader=shader,diffuse=[0.9,0.7,0.13,1], ambient=[0.1,0.1,0.2,1], specular=[1,1,1,1], emission=[0.1,0.1,0.1,1],shininess=5,batch=batch)

# 視点を設定
window.view = Mat4.look_at(position=Vec3(0,0,3), target=Vec3(0,0,0), up=Vec3(0,1,0))

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()
