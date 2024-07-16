import trimesh

# Load a mesh from OBJ file
mesh = trimesh.load('/home/haoran/Projects/Rearrangement/Open6DOR/Method/assets/objaverse_final_norm/69511a7fad2f42ee8c4b0579bbc8fec6/material.obj')

# Translate mesh to its centroid
mesh.apply_translation(-mesh.centroid)

import pdb; pdb.set_trace()
# Scale the mesh (1 unit here)
scale_factor = 1.0 / mesh.bounding_box.extents.max()
mesh.apply_scale(scale_factor)

# save the new mesh to OBJ file
mesh.export('output.obj')