import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3, Vec4
from pyglet.graphics.shader import Shader, ShaderProgram
import numpy as np
import ctypes
from skimage import measure

vertex_source1 = '''#version 330 core
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
'''

fragment_source1 = '''#version 330 core
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
        vec3 refrect_dir = reflect(-light_dir, normal);
        vec3 view_dir = -normalize(vertex_position);
        float spec = pow(max(dot(view_dir, refrect_dir), 0.0), vertex_shininess);
        float diff = max(dot(normal, light_dir), 0.0);
        if (dot(normal, light_dir)<0) spec=0.0 ;

        final_colors = vertex_ambient * light_color
                     + vertex_diffuse * diff
                     + vertex_specular * spec * light_color
                     + vertex_emission ;
    }
'''
vertex_source2 = '''#version 330 core
   in vec3 position;

   uniform WindowBlock {
        mat4 projection;
        mat4 view;
   } window;

   uniform mat4 model ;

   void main(void)
   {
     mat4 modelview = window.view * model;
     vec4 pos = modelview * vec4(position, 1.0);
     gl_Position = window.projection * pos;
   }
'''

fragment_source2 = '''#version 330 core
   out vec4 final_colors ;
   void main(void)
   {
     final_colors = vec4(1.0, 0.0, 0.0, 1.0) ;
   }
'''

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
            self.matrix == other.matrix and
            self.program == other.program and
            self.order == other.order and
            self.parent == other.parent)

window = pyglet.window.Window(800, 600, "3D Isosurface Example")

batch1 = pyglet.graphics.Batch()
vert_shader1 = Shader(vertex_source1, 'vertex')
frag_shader1 = Shader(fragment_source1, 'fragment')
shader1 = ShaderProgram(vert_shader1, frag_shader1)

glClearColor(0.3, 0.3, 0.5, 1.0)
glEnable(GL_DEPTH_TEST)

def load_npy_and_gen_model(npy_file, iso_level, shader, batch):
    data3d = np.load(npy_file)
    vertices, faces, normals, values = measure.marching_cubes(data3d, level=iso_level)

    xsiz,ysiz,zsiz = data3d.shape
    scales = np.array([2/xsiz,2/ysiz,2/zsiz])
    
    vertices = vertices * scales[:]
    vertices = vertices.flatten() - 1    
    normals  = normals * scales[:]
    normals = normals.flatten()
    
    indices = faces.flatten()

    diffuse = (0.2,0.5,0.8,1)
    ambient = (0,0,0,1)
    specular = (1,1,1,1)
    emission = (0,0,0,1)
    shininess = 20

    material = pyglet.model.Material("custom", diffuse, ambient, specular, emission, shininess)
    group = MyMaterialGroup(material=material, program=shader)
        
    vertex_list = shader.vertex_list_indexed(len(vertices)//3, GL_TRIANGLES, indices, batch=batch, group=group)
    vertex_list.position[:] = vertices 
    vertex_list.normals[:] = normals
    vertex_list.diffuse_colors[:] = material.diffuse * (len(vertices)//3)
    vertex_list.ambient_colors[:] = material.ambient * (len(vertices)//3)
    vertex_list.specular_colors[:] = material.specular * (len(vertices)//3)
    vertex_list.emission_colors[:] = material.emission * (len(vertices)//3)
    vertex_list.shininess[:] = [material.shininess] * (len(vertices)//3)        

    return pyglet.model.Model(vertex_lists=[vertex_list], groups=[group], batch=batch)

npy_file = 'cahn–hilliard-model.npy'
iso_model = load_npy_and_gen_model(npy_file, 0.5, shader1, batch1)

# draw a cube
vert_shader2 = Shader(vertex_source2, 'vertex')
frag_shader2 = Shader(fragment_source2, 'fragment')
shader2 = ShaderProgram(vert_shader2, frag_shader2)
batch2 = pyglet.graphics.Batch()
cube = shader2.vertex_list(24, GL_LINES,
                           position=('f',(-1, -1, -1, 1, -1, -1,    -1, -1, -1, -1, 1, -1,    -1, -1, -1, -1, -1, 1,
                                          1, 1, 1, -1, 1, 1,    1, 1, 1, 1, -1, 1,    1, 1, 1, 1, 1, -1,
                                          -1, 1, 1, -1, -1, 1,    -1, 1, 1, -1, 1, -1,    1, -1, 1, -1, -1, 1,
                                          1, -1, 1, 1, -1, -1,    1, 1, -1, 1, -1, -1, 1,    1, -1, -1, 1, -1)),
                           batch=batch2)

time=0
model_mat = Mat4()
def update(dt):
    global time, model_mat
    time = time + dt
    model_mat = Mat4.from_rotation(time*0.1,Vec3(0,1,0))

@window.event
def on_draw():
    window.clear()    
    ratio = window.height/window.width
    window.viewport = (0, 0, window.width, window.height)
    window.projection = Mat4.orthogonal_projection(-2,2,-2*ratio,2*ratio,-10,20)
    shader1['light_color'] = Vec4(1,1,1,1)
    shader1['light_position'] = Vec3(10,10,10)    
    iso_model.matrix= model_mat
    shader2['model']= model_mat 
    batch1.draw()
    batch2.draw()    



window.view = Mat4.look_at(position=Vec3(0,5,10), target=Vec3(0,0,0), up=Vec3(0,1,0))        
pyglet.clock.schedule_interval(update,1/30)
pyglet.app.run()

