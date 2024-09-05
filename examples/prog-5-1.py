import numpy as np
import math

def tetrahedron():
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

    return (vertices,normals)

# 正四面体モデルを生成
vertices,normals = tetrahedron()

print("Vertices=",vertices)
print("Normal_Vectors=",normals)
