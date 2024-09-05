from pyglet.gl import *
from ctypes import c_int, byref

max_invocations = c_int()
glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, byref(max_invocations))
print('GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS=',max_invocations.value)

x_size = c_int()
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, byref(x_size))
print('MAX_LOCAL_SIZE_X=', x_size.value)

y_size = c_int()
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, byref(y_size))
print('MAX_LOCAL_SIZE_Y=', y_size.value)

z_size = c_int()
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, byref(z_size))
print('MAX_LOCAL_SIZE_Z=', z_size.value)
