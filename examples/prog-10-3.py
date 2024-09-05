import pyglet
from pyglet.gl import *
import numpy as np
import random

print(gl_info.get_version())
print(gl_info.get_vendor())
print(gl_info.get_renderer())
# print(gl_info.get_extensions())

compute_src = """#version 430 core
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(rgba32f,binding=0) uniform image2D image_data;
layout(rgba32f,binding=1) uniform image2D image_out;

ivec2 coord(ivec2 center, ivec2 size, int dx, int dy) {
    return ivec2((center.x + dx + size.x) % size.x, (center.y + dy + size.y) % size.y) ;
}

void main() {
    vec4 value = vec4(0, 0.05, 0.15, 1);
    ivec2 img_size = imageSize(image_data) ;
    ivec2 c_coord = ivec2(gl_GlobalInvocationID.xy);
    vec4 center = imageLoad(image_data, c_coord);    

    ivec2 r_coord = coord(c_coord, img_size, +1,  0) ;
    ivec2 l_coord = coord(c_coord, img_size, -1,  0) ;
    ivec2 t_coord = coord(c_coord, img_size,  0, +1) ;
    ivec2 b_coord = coord(c_coord, img_size,  0, -1) ;

    ivec2 tr_coord = coord(c_coord, img_size, +1, +1) ; 
    ivec2 tl_coord = coord(c_coord, img_size, -1, +1) ; 
    ivec2 br_coord = coord(c_coord, img_size, +1, -1) ; 
    ivec2 bl_coord = coord(c_coord, img_size, -1, -1) ; 

    vec4 left = imageLoad(image_data, l_coord);
    vec4 right = imageLoad(image_data, r_coord);
    vec4 top = imageLoad(image_data, t_coord);
    vec4 bottom = imageLoad(image_data, b_coord);
    vec4 top_right = imageLoad(image_data, tr_coord);
    vec4 top_left = imageLoad(image_data, tl_coord);
    vec4 bottom_right = imageLoad(image_data, br_coord);
    vec4 bottom_left = imageLoad(image_data, bl_coord);

    float sum = left.r + right.r + top.r + bottom.r + top_right.r + top_left.r + bottom_right.r + bottom_left.r ;
 
    if (sum==3) {
       value.r = 1 ;
    } else if (center.r==1 && sum==2) {
       value.r = 1 ;
    } else {
       value.r = 0 ;
    }

    imageStore(image_out, c_coord, value);
}
"""

width=1024
height=1024

window = pyglet.window.Window(width=width, height=height, resizable=False)
window.set_caption('Game of life')

init_array = np.empty((height, width, 4), dtype=np.float32)
for i in range(height):
    for j in range(width):
        init_array[i,j,0] = random.choice([0.0, 1.0])
        init_array[i,j,1] = 0
        init_array[i,j,2] = 0  
        init_array[i,j,3] = 1.0

program = pyglet.graphics.shader.ComputeShaderProgram(compute_src)

img_array = init_array.reshape((height, width, 4)) 
glActiveTexture(GL_TEXTURE0)
texture0 = pyglet.image.Texture.create(width, height, internalformat=GL_RGBA32F)
texture0.bind_image_texture(unit=0,fmt=GL_RGBA32F)
glTexImage2D(pyglet.gl.GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0,
             GL_RGBA, GL_FLOAT, img_array.ctypes.data)

texture1 = pyglet.image.Texture.create(texture0.width, texture0.height, internalformat=GL_RGBA32F)
glActiveTexture(GL_TEXTURE1)
texture1.bind_image_texture(unit=1,fmt=GL_RGBA32F)

default_shader = pyglet.graphics.get_default_blit_shader()

loop_cnt=0

label = pyglet.text.Label('GENERATION =' + str(loop_cnt),
                          font_name='Arial', color=(200, 200, 200, 128),
                          font_size=14, x=10, y=10,
                          anchor_x='left', anchor_y='bottom')

def rotate():
    global loop_cnt
    texture0.blit_into(texture1.get_image_data(),0,0,0)
    loop_cnt += 1

def update(dt):
    rotate()
    label.text = 'GENERATION =' + str(loop_cnt)

@window.event
def on_draw():   
    program.use()
    with program:
        program.dispatch(texture0.width, texture0.height, 1)
    program.stop()

    window.clear()
    default_shader.use()
    texture1.blit(0, 0)

    label.draw()


pyglet.clock.schedule_interval(update,1/30)
pyglet.app.run()
