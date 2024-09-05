import networkx as nx
import pyglet
from pyglet.gl import *
import numpy as np
from pyglet.math import Mat4, Vec3, Vec4
import ctypes

compute_src = """#version 430 core
precision highp float;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (std430, binding = 0) buffer SSBO0 {
  float positions[];
} ;

layout (std430, binding = 1) buffer SSBO1 {
  int nedges[];
} ;

layout (std430, binding = 2) buffer SSBO2 {
  int indices[];
} ;

layout (std430, binding = 3) buffer SSBO3 {
  int neighbors[];
} ;


vec3 get_position(uint i) {
    return vec3(positions[i*3], positions[i*3+1], positions[i*3+2]) ;
}

void set_position(uint i, vec3 value) {
    positions[i*3] = value.x ;
    positions[i*3+1] = value.y ;
    positions[i*3+2] = value.z ;
}

vec3 attractive_force(vec3 pi, vec3 pj) {
    return (pj - pi) ;
}

vec3 repulsive_force(vec3 pi, vec3 pj) {
    vec3 direction = normalize(pi - pj) ;
    float dist = distance(pi,pj) ;
    float force = 1.0/(dist*dist) ;
    return force * direction ;
}

void main() {
    uint i = gl_GlobalInvocationID.x ;
    uint nedge = gl_NumWorkGroups.x;
    int j,ncnt ;
    vec3 posi = get_position(i) ;
    vec3 force = vec3(0,0,0) ;

    // attraction
    j = indices[i] ;
    ncnt = nedges[i] ;
    while (ncnt>0) {
        vec3 posj = get_position(neighbors[j]) ;
        force = force + attractive_force(posi, posj) ;
        j = j + 1 ;
        ncnt -= 1 ;
    }    

    // repulsion
    for (j=0 ; j<nedge ; j++) {
        if (j==i) continue ;
        vec3 posj = get_position(j) ;        
        float charge2 = float(nedges[i]) * float(nedges[j]) ;
        force = force + charge2 * repulsive_force(posi, posj) ;
    }

#define DT 0.002

    posi = posi + force*DT ;

    set_position(i, posi) ;
}
"""
vertex_source1 = '''#version 330 core
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

fragment_source1 = '''#version 330 core
  out vec4 final_colors ;
  void main(void)
  {
     final_colors = vec4(0.2, 0.7, 0.5, 0.75) ;
  }
'''

vertex_source2 = '''#version 330 core
in vec3 position;
in float size;
in float phi;
in float theta;
in float nodeid; // could not use int...

out vec3 vertex_position;
out float vertex_size;
out float vertex_phi;
out float vertex_theta;
out int vertex_nodeid;

void main(void) {
    vertex_position = position ;
    vertex_size = size ;
    vertex_phi = phi ;
    vertex_theta = theta ;
    vertex_nodeid = int(nodeid) ;
}
'''
geometry_source2 = '''#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 48) out;

in vec3 vertex_position[];
in float vertex_size[];
in float vertex_phi[];
in float vertex_theta[];
in int vertex_nodeid[];

out vec3 geom_position;
out vec3 geom_normal;
out vec2 geom_texcoord;
flat out int geom_nodeid;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

uniform mat4 model;

void main(void) {    
    const vec3 vlist[8] = vec3[8](
        vec3(-0.5, -0.5, -0.5),
        vec3( 0.5, -0.5, -0.5),
        vec3( 0.5,  0.5, -0.5),
        vec3(-0.5,  0.5, -0.5),
        vec3(-0.5, -0.5,  0.5),
        vec3( 0.5, -0.5,  0.5),
        vec3( 0.5,  0.5,  0.5),
        vec3(-0.5,  0.5,  0.5)
    );

    const int indices[36] = int[36](
        0, 2, 1, 0, 3, 2,  
        4, 5, 6, 4, 6, 7,  
        0, 1, 5, 0, 5, 4,  
        2, 3, 7, 2, 7, 6,  
        0, 4, 7, 0, 7, 3,  
        1, 2, 6, 1, 6, 5   
    );

     const vec2 texcoords[36] = vec2[36](
        vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 0.0), 
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0), 

        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), 
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0), 
    
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0), 
        vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 0.0), 
    
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), 
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0), 
    
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), 
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0), 
    
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0), 
        vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 0.0)  
    );

    mat3 rot_z = mat3(
        cos(vertex_phi[0]), -sin(vertex_phi[0]), 0.0,
        sin(vertex_phi[0]),  cos(vertex_phi[0]), 0.0,
        0.0, 0.0, 1.0
    );

    mat3 rot_x = mat3(
        1.0, 0.0, 0.0,
        0.0, cos(vertex_theta[0]), -sin(vertex_theta[0]),
        0.0, sin(vertex_theta[0]), cos(vertex_theta[0])
    );

    mat4 modelview = window.view * model;
    mat3 normal_matrix = transpose(inverse(mat3(modelview)));

    for (int i = 0; i < 36; i += 3) {
        vec3 p1 = rot_z * rot_x * vlist[indices[i]];
        vec3 p2 = rot_z * rot_x * vlist[indices[i+1]];
        vec3 p3 = rot_z * rot_x * vlist[indices[i+2]];
        vec3 normal = normalize(cross(p2 - p1, p3 - p1));

        vec4 pos;

        pos = modelview * vec4(p1 * vertex_size[0] + vertex_position[0], 1.0);
        gl_Position = window.projection * pos;
        geom_position = pos.xyz;
        geom_normal = normal_matrix * normal;
        geom_texcoord = texcoords[i];         
        geom_nodeid = vertex_nodeid[0];
        EmitVertex();

        pos = modelview * vec4(p2 * vertex_size[0] + vertex_position[0], 1.0);
        gl_Position = window.projection * pos;
        geom_position = pos.xyz;
        geom_normal = normal_matrix * normal;
        geom_texcoord = texcoords[i + 1];     
        geom_nodeid = vertex_nodeid[0];        
        EmitVertex();

        pos = modelview * vec4(p3 * vertex_size[0] + vertex_position[0], 1.0);
        gl_Position = window.projection * pos;
        geom_position = pos.xyz;
        geom_normal = normal_matrix * normal;
        geom_texcoord = texcoords[i + 2];     
        geom_nodeid = vertex_nodeid[0];
        EmitVertex();

        EndPrimitive();
    }
}
'''

fragment_source2 = """#version 330 core
in vec3 geom_position;
in vec3 geom_normal;
in vec2 geom_texcoord;
flat in int geom_nodeid;

out vec4 frag_color;

uniform vec3 light_position ;

uniform sampler2D texture_2d; 

vec4 get_tex_color(int id) {
    int d4 = (id / 10000) % 10;
    int d3 = (id / 1000) % 10;
    int d2 = (id / 100) % 10;
    int d1 = (id / 10) % 10;
    int d0 = id % 10;

    vec4 frag_color ;

    if (geom_texcoord.x < 2.0/10.0) {
        vec2 texcoord_d4 = vec2( float(d4)/10.0  + (geom_texcoord.x-0.0)/2, geom_texcoord.y) ;    
        frag_color = texture(texture_2d, texcoord_d4);
    } else if (geom_texcoord.x < 4.0/10.0) {
        vec2 texcoord_d3 = vec2( float(d3)/10.0  + (geom_texcoord.x-0.2)/2, geom_texcoord.y) ;    
        frag_color = texture(texture_2d, texcoord_d3);
    } else if (geom_texcoord.x < 6.0/10.0) {
        vec2 texcoord_d2 = vec2( float(d2)/10.0  + (geom_texcoord.x-0.4)/2, geom_texcoord.y) ;      
        frag_color = texture(texture_2d, texcoord_d2);
    } else if (geom_texcoord.x < 8.0/10.0) {
        vec2 texcoord_d1 = vec2( float(d1)/10.0  + (geom_texcoord.x-0.6)/2, geom_texcoord.y) ;    
        frag_color = texture(texture_2d, texcoord_d1);
    } else {
        vec2 texcoord_d0 = vec2( float(d0)/10.0  + (geom_texcoord.x-0.8)/2, geom_texcoord.y) ;    
        frag_color = texture(texture_2d, texcoord_d0);
    }
    return frag_color ;
}


void main()
{
    int nodeid = gl_PrimitiveID % 1000 + 1;
    vec4 color = vec4(1.0, 0.5, 0.5, 1.0) ;
    vec3 normal = normalize(geom_normal);
    vec3 light_dir = normalize(light_position - geom_position);
    float diff = max(dot(normal, light_dir), 0.0);
    vec4 tex_color = get_tex_color(geom_nodeid) ;
    frag_color = tex_color * color * diff ;
    // frag_color = (texture(texture_2d, geom_texcoord) * color)  ;    
}
"""

filename = "YeastL.net"
# filename = "Erdos02.net"
G = nx.read_pajek(filename).to_undirected()

# selec connected graphics, them choose the maximum one 
largest_cc = max(nx.connected_components(G), key=len)

# generate a subgraph from the largest connected graph
G2 = G.subgraph(largest_cc).copy()

print(G2)
for i,label in enumerate(G2.nodes()):
    print(i,label)

# prepare arrays to bind SSBOs
node_mapping = {node: i for i, node in enumerate(G2.nodes, start=0)}
nnode = len(node_mapping)
print('number of nodes=',nnode)

positions = np.random.normal(loc=0.0, scale=30.0, size=(nnode,3)).astype(dtype='float32')
nedges = np.zeros(shape=(nnode,), dtype='int32')
indices = np.zeros(shape=(nnode,), dtype='int32')
neighbors = [ ]

icum=0
i=0
for node, adjacencies in G2.adjacency():
    node_num = node_mapping[node]
    neighbor = [node_mapping[neighbor] for neighbor in adjacencies]
    neighbors.extend(neighbor)
    nedges[i] = len(neighbor)
    indices[i] = icum
    # print(i,nedges[i],icum)
    icum += nedges[i]    
    i += 1

neighbors = np.array(neighbors,dtype='int32')

#
#
window = pyglet.window.Window(width=1280, height=720, resizable=True)
window.set_caption('Complex Network')
glClearColor(0.05, 0.05, 0.15, 1)
glEnable(GL_DEPTH_TEST)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 
numbers = pyglet.image.load('number-texture.png') 
texture = numbers.get_texture()      
glActiveTexture(GL_TEXTURE0)            
glBindTexture(texture.target, texture.id)  

# compute shader program
program = pyglet.graphics.shader.ComputeShaderProgram(compute_src)
# shader to display edges with lines
vert_shader1 = pyglet.graphics.shader.Shader(vertex_source1, 'vertex')
frag_shader1 = pyglet.graphics.shader.Shader(fragment_source1, 'fragment')
shader1 = pyglet.graphics.shader.ShaderProgram(vert_shader1, frag_shader1)
# shader to display nodes with tetrahedron
vert_shader2 = pyglet.graphics.shader.Shader(vertex_source2, 'vertex')
geom_shader2 = pyglet.graphics.shader.Shader(geometry_source2, 'geometry')
frag_shader2 = pyglet.graphics.shader.Shader(fragment_source2, 'fragment')
shader2 = pyglet.graphics.shader.ShaderProgram(vert_shader2, geom_shader2, frag_shader2)

    

# edges
batch1 = pyglet.graphics.Batch()
# nodes
batch2 = pyglet.graphics.Batch()

# generate SSBOs (shader storate buffers)
position_data = positions.flatten()
ssbo0 = GLuint()
glGenBuffers(1,ssbo0)

nedge_data = nedges.flatten()
ssbo1 = GLuint()
glGenBuffers(1,ssbo1)

index_data = indices.flatten()
ssbo2 = GLuint()
glGenBuffers(1,ssbo2)

neighbor_data = neighbors.flatten()
ssbo3 = GLuint()
glGenBuffers(1,ssbo3)

def update_position():
  global positions, position_data
  program.use() 

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo0)
  glBufferData(GL_SHADER_STORAGE_BUFFER, position_data.nbytes, position_data.ctypes.data, GL_DYNAMIC_DRAW)
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo0)

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo1)
  glBufferData(GL_SHADER_STORAGE_BUFFER, nedge_data.nbytes, nedge_data.ctypes.data, GL_STATIC_DRAW)
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo1)

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo2)
  glBufferData(GL_SHADER_STORAGE_BUFFER, index_data.nbytes, index_data.ctypes.data, GL_STATIC_DRAW)
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo2)

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo3)
  glBufferData(GL_SHADER_STORAGE_BUFFER, neighbor_data.nbytes, neighbor_data.ctypes.data, GL_STATIC_DRAW)
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo3)

  with program:
      program.dispatch(nnode, 1, 1)

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo0)
  mapped_data_ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
  mapped_data = ctypes.cast(mapped_data_ptr, ctypes.POINTER(ctypes.c_float))
  result = np.ctypeslib.as_array(mapped_data, shape=(nnode,3))
  position_data[:] = result.flatten()[:]
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)  
  program.stop()    
  return result

def gen_vertices(nedges, indices, neighbors, pos):
  vertices = [ ]
  N = len(nedges)
  for i in range(N):
    x0 = pos[i,0]
    y0 = pos[i,1]
    z0 = pos[i,2]     
    k = indices[i]
    l = k + nedges[i]
    for n in range(k,l):
      j = neighbors[n]
      x1 = pos[j,0]
      y1 = pos[j,1]
      z1 = pos[j,2]
      vertices.extend([x0,y0,z0,x1,y1,z1])  
  return vertices   

def gen_points(pos):
  points = [ ]
  N = len(nedges)
  for i in range(N):
    x0 = pos[i,0]
    y0 = pos[i,1]
    z0 = pos[i,2]     
    points.extend([x0,y0,z0])  
  return points

time =0
def update(dt):
  global time
  res = update_position()
  vertices = gen_vertices(nedge_data, index_data, neighbor_data, res)
  vertex_list1.position[:] = vertices

  points = gen_points(res)
  vertex_list2.position[:] = points

  time += dt

fov = 60
@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global fov
    fov *= np.exp(scroll_y*0.05)
    if fov < 1:
        fov = 1
    elif fov>120:
        fov = 120
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.1, z_far=500, fov=fov)
    return pyglet.event.EVENT_HANDLED

model_mat = Mat4()
@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
  global model_mat
  model_mat = Mat4.from_rotation(dx*fov*0.0005, Vec3(0, 1, 0))  @ model_mat
  model_mat = Mat4.from_rotation(-dy*fov*0.0005, Vec3(1, 0, 0)) @ model_mat
  shader1['model'] = model_mat
  shader2['model'] = model_mat  
  return pyglet.event.EVENT_HANDLED

running=False
@window.event
def on_key_press(symbol, modifiers):
    global model_mat,positions,position_data,running
    if symbol == pyglet.window.key.R:
        model_mat = Mat4()
        shader1['model'] = model_mat
        shader2['model'] = model_mat 
    elif symbol == pyglet.window.key.I:
        positions = np.random.normal(loc=0.0, scale=30.0, size=(nnode,3)).astype(dtype='float32')
        position_data = positions.flatten()
    elif symbol == pyglet.window.key.S:
        if running:
            pyglet.clock.unschedule(update)
            running = False
        else:
            pyglet.clock.schedule_interval(update,1/30)
            running = True           
    return pyglet.event.EVENT_HANDLED

@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.1, z_far=500, fov=fov)
    return pyglet.event.EVENT_HANDLED
   
@window.event
def on_draw():
    window.clear()    
    ratio = window.height/window.width
    window.viewport = (0, 0, window.width, window.height)
    window.projection = Mat4.perspective_projection(window.aspect_ratio, z_near=0.1, z_far=500, fov=fov) 
    shader1['model']= model_mat
    shader2['model']= model_mat
    glEnable(GL_BLEND)
    batch1.draw() 
    glDisable(GL_BLEND)
    batch2.draw()

# initial plot
result = update_position()
vertices = gen_vertices(nedge_data, index_data, neighbor_data, result)
vertex_list1 = shader1.vertex_list(len(vertices)//3, GL_LINES, batch=batch1)
vertex_list1.position[:] = vertices

points = gen_points(result)
vertex_list2 = shader2.vertex_list(len(points)//3, GL_POINTS, batch=batch2)                                   
vertex_list2.position[:] = points
vertex_list2.size[:] = [0.5] * (len(points)//3)
vertex_list2.phi[:] = np.random.uniform(0.0, 2*np.pi, len(points)//3)
vertex_list2.theta[:] = np.random.uniform(0.0, 2*np.pi, len(points)//3)
vertex_list2.nodeid = [i for i in range(len(points)//3)]

shader2['light_position'] = Vec3(0,500,1000)

window.view = Mat4.look_at(position=Vec3(0,0,200), target=Vec3(0,0,0), up=Vec3(0,1,0))    

pyglet.clock.schedule_interval(update,1/30)
running=True
pyglet.app.run()    