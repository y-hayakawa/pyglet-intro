import pyglet
from pyglet.gl import *
from pyglet.math import Mat4, Vec3, Vec4
from pyglet.graphics.shader import Shader, ShaderProgram
import numpy as np
import ctypes

vertex_source1 = '''#version 330 core
   in vec3 position;
   in vec3 normal;

   out VS_OUT {
        vec3 position;
        vec3 normal;
   } vs_out;

   void main()
   {
        vs_out.position = position;
        vs_out.normal = normal;
    }
'''

geometory_source1 = '''#version 330 core
  layout(points) in;
  layout(triangle_strip, max_vertices = 32) out;

   in VS_OUT {
        vec3 position;
        vec3 normal;
   } gs_in[];

   out vec3 geom_tex_coord;

   uniform mat4 model;

   uniform WindowBlock
   {
        mat4 projection;
        mat4 view;
   } window;

    vec3 calculate_intersection(vec3 normal, vec3 point, vec3 r0, vec3 r1) {
        vec3 direction = r1 - r0;
        float denom = dot(normal, direction);
    
        if (denom != 0.0) {
            float t = (dot(normal, point) - dot(normal, r0)) / denom;
            if (t >= 0.0 && t <= 1.0) {
                return r0 + t * direction;
            }
        }
        return vec3(0.0);
    }

    void sort_intersections(inout vec3 intersections[12], int count, vec3 normal) {
        vec3 center = vec3(0.0);
        for (int i = 0; i < count; i++) {
            center += intersections[i];
        }
        center /= float(count);

        vec3 dr0 = normalize(intersections[0] - center);
        vec3 dr1 = normalize(cross(normal, dr0));

        float angles[12];
        for (int i = 0; i < count; i++) {
            vec3 p = normalize(intersections[i] - center);
            angles[i] = atan(dot(dr0, p), dot(dr1, p));
        }

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < count - 1 - i; j++) {
                if (angles[j] > angles[j + 1]) {
                    float tempAngle = angles[j];
                    vec3 tempVec = intersections[j];
                    angles[j] = angles[j + 1];
                    intersections[j] = intersections[j + 1];
                    angles[j + 1] = tempAngle;
                    intersections[j + 1] = tempVec;
                }
            }
        }
    }

    void main() {
        vec3 cube_edges[24] = vec3[24](
            vec3(-1, -1, -1), vec3(1, -1, -1),
            vec3(-1, -1, -1), vec3(-1, 1, -1),
            vec3(-1, -1, -1), vec3(-1, -1, 1),
            vec3(1, 1, 1), vec3(-1, 1, 1),
            vec3(1, 1, 1), vec3(1, -1, 1),
            vec3(1, 1, 1), vec3(1, 1, -1),
            vec3(-1, 1, 1), vec3(-1, -1, 1),
            vec3(-1, 1, 1), vec3(-1, 1, -1),
            vec3(1, -1, 1), vec3(-1, -1, 1),
            vec3(1, -1, 1), vec3(1, -1, -1),
            vec3(1, 1, -1), vec3(1, -1, -1),
            vec3(1, 1, -1), vec3(-1, 1, -1)
        );

        vec3 intersections[12];
        int nIntersections = 0;

        for (int i = 0; i < 12; i++) {
            vec3 intersection = calculate_intersection(gs_in[0].normal, gs_in[0].position, cube_edges[2 * i], cube_edges[2 * i + 1]);
            if (intersection != vec3(0.0)) {
                intersections[nIntersections++] = intersection;
            }
        }

        mat4 modelview = window.view * model;
        vec4 pos = modelview * vec4(gs_in[0].position, 1.0);

        if (nIntersections > 2) {
            sort_intersections(intersections, nIntersections, gs_in[0].normal);
            vec3 center = vec3(0.0);
            for (int i = 0; i < nIntersections; i++) {
                center += intersections[i];
            }
            center /= float(nIntersections);

            for (int i = 0; i < nIntersections; i++) {
                int next_i = (i + 1) % nIntersections;

                gl_Position = window.projection * modelview * vec4(intersections[i], 1.0);
                geom_tex_coord = (vec3(intersections[i]) + vec3(1,1,1)) / 2 ;
                EmitVertex();

                gl_Position = window.projection * modelview * vec4(center, 1.0);
                geom_tex_coord = (vec3(center) + vec3(1,1,1)) / 2 ;
                EmitVertex();

                gl_Position = window.projection * modelview * vec4(intersections[next_i], 1.0);
                geom_tex_coord = (vec3(intersections[next_i]) + vec3(1,1,1)) / 2 ;
                EmitVertex();

                EndPrimitive();
            }
        }
    }
'''

fragment_source1 = '''#version 330 core
  in vec3 geom_tex_coord;
  out vec4 final_colors;

  uniform sampler3D texture_3d;

  void main(void)
  {
        vec4 tex_color = texture(texture_3d, geom_tex_coord);
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


window = pyglet.window.Window(800, 600, "3D Texture Example")

batch1 = pyglet.graphics.Batch()
vert_shader1 = Shader(vertex_source1, 'vertex')
geom_shader1 = Shader(geometory_source1, 'geometry')
frag_shader1 = Shader(fragment_source1, 'fragment')
shader1 = ShaderProgram(vert_shader1, geom_shader1, frag_shader1)

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

    vertices = [ ]
    normals = [ ]
    d_values = np.linspace(-np.sqrt(3), np.sqrt(3), 256)

    for d in d_values:
        point = d * normal
        vertices.extend(point)
        normals.extend(normal)

    vertex_list = shader1.vertex_list(len(vertices) // 3, GL_POINTS, batch=batch)
    vertex_list.position[:] = vertices
    vertex_list.normal[:] = normals
    
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

