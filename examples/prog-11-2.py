import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3, Vec4
from pyglet.graphics.shader import Shader, ShaderProgram
import numpy as np
import ctypes

vertex_source1 = '''#version 330 core
  in vec3 position;
  in vec3 tex_coords;

  out vec3 vertex_position;
  out vec3 vertex_tex_coord;

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
     vertex_tex_coord = tex_coords.xyz ;
  }
'''

fragment_source1 = '''#version 330 core
  in vec3 vertex_tex_coord;
  out vec4 final_colors;

  uniform sampler3D texture_3d;

  void main(void)
  {
     vec4 tex_color = texture(texture_3d, vertex_tex_coord);
     float alpha = tex_color.r ;
     if (alpha < 0.01) discard;
     final_colors = vec4(tex_color.r, tex_color.r, tex_color.r, alpha*0.5);
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

def load_npy_as_3d_texture(npy_file):
    image_stack = np.load(npy_file)
    image_array = []
    for i in range(image_stack.shape[0]):
        image_data = image_stack[i] * 255
        height, width = image_data.shape
        image_data = image_data.astype(np.uint8)
        image = pyglet.image.ImageData(width, height, 'R', image_data.tobytes())
        image_array.append(image)
    texture_3d = pyglet.image.Texture3D.create_for_images(image_array, internalformat=GL_RED)
    return texture_3d

def calculate_intersection(normal, point, edge):
    r0, r1 = np.array(edge[0]), np.array(edge[1])
    direction = r1 - r0
    denom = np.dot(normal, direction)
    
    if denom != 0:
        t = (np.dot(normal, point) - np.dot(normal, r0)) / denom
        if 0 <= t <= 1:
            return r0 + t * direction
    return None

def sort_intersections(intersections, normal):
    center = np.mean(intersections, axis=0)
    dr0 = (intersections[0] - center) / np.linalg.norm(intersections[0] - center)
    dr1 = np.cross(normal, dr0) / np.linalg.norm(np.cross(normal, dr0))
    
    angles = [np.arctan2(np.dot(dr0, (p - center) / np.linalg.norm(p - center)),
                         np.dot(dr1, (p - center) / np.linalg.norm(p - center)))
              for p in intersections]
    
    sorted_indices = np.argsort(angles)
    return intersections[sorted_indices], center

def generate_triangle_vertices(sorted_intersections, center):
    vertices = []
    nisect = len(sorted_intersections)
    
    for i in range(nisect):
        next_i = (i + 1) % nisect
        vertices.extend(sorted_intersections[i].tolist() +
                        center.tolist() +
                        sorted_intersections[next_i].tolist())
    
    return vertices

def gen_triangles(normal, point):
    edges = [
        [[-1, -1, -1], [1, -1, -1]], [[-1, -1, -1], [-1, 1, -1]], [[-1, -1, -1], [-1, -1, 1]],
        [[1, 1, 1], [-1, 1, 1]], [[1, 1, 1], [1, -1, 1]], [[1, 1, 1], [1, 1, -1]],
        [[-1, 1, 1], [-1, -1, 1]], [[-1, 1, 1], [-1, 1, -1]], [[1, -1, 1], [-1, -1, 1]],
        [[1, -1, 1], [1, -1, -1]], [[1, 1, -1], [1, -1, -1]], [[1, 1, -1], [-1, 1, -1]]
    ]
    
    intersections = [calculate_intersection(normal, point, edge) for edge in edges]
    intersections = [pt for pt in intersections if pt is not None]

    if not intersections:
        return []
    
    sorted_intersections, center = sort_intersections(np.array(intersections), normal)
    vertices = generate_triangle_vertices(sorted_intersections, center)
    
    return vertices

window = pyglet.window.Window(800, 600, "3D Texture Example")

batch1 = pyglet.graphics.Batch()
vert_shader1 = Shader(vertex_source1, 'vertex')
frag_shader1 = Shader(fragment_source1, 'fragment')
shader1 = ShaderProgram(vert_shader1, frag_shader1)

npy_file = 'cahn–hilliard-model.npy'
texture = load_npy_as_3d_texture(npy_file)

glClearColor(0.3, 0.3, 0.5, 1.0)
glEnable(GL_DEPTH_TEST)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_BLEND)
glActiveTexture(GL_TEXTURE0)
glBindTexture(texture.target, texture.id)

def gen_slices(x0, y0, z0, batch):
    normal = np.array([x0, y0, z0], dtype='float')
    normal /= np.linalg.norm(normal)

    vertices = []
    tex_coords = []
    d_values = np.linspace(-np.sqrt(3), np.sqrt(3), 256)

    for d in d_values:
        point = d * normal
        triangles = gen_triangles(normal, point)
        if triangles:
            vertices.extend(triangles)
            tex_coords.extend((np.array(triangles) + 1) * 0.5)

    vertex_list = shader1.vertex_list(len(vertices) // 3, GL_TRIANGLES, batch=batch)
    vertex_list.position[:] = vertices
    vertex_list.tex_coords[:] = tex_coords
    
    return vertex_list

# camera position
x0 = 0
y0 = 10
z0 = 10
camera_pos = Vec3(x0,y0,z0)
# initial slices
vertex_list = gen_slices(x0,y0,z0,batch1)

# draw the bounding cube
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
    global vertex_list ,time, model_mat
    time = time + dt
    model_mat = Mat4.from_rotation(time*0.1,Vec3(0,1,0))
    inv_pos = model_mat.__invert__() @ Vec4(camera_pos.x, camera_pos.y, camera_pos.z, 1.0)
    # regenerate slice
    vertex_list.delete()
    vertex_list = gen_slices(inv_pos.x, inv_pos.y, inv_pos.z, batch1)    


@window.event
def on_draw():
    window.clear()    
    ratio = window.height/window.width
    window.viewport = (0, 0, window.width, window.height)
    window.projection = Mat4.orthogonal_projection(-2,2,-2*ratio,2*ratio,-10,20)    
    shader1['model']= model_mat
    shader2['model']= model_mat 
    batch1.draw()
    batch2.draw()    

window.view = Mat4.look_at(position=camera_pos, target=Vec3(0,0,0), up=Vec3(0,1,0))        
pyglet.clock.schedule_interval(update,1/30)
pyglet.app.run()

