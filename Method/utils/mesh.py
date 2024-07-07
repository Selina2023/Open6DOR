import trimesh

# Load a mesh from OBJ file
mesh = trimesh.load('path_to_mesh.obj')

# Translate mesh to its centroid
mesh.apply_translation(-mesh.centroid)

# Scale the mesh (1 unit here)
scale_factor = 1.0 / mesh.bounding_box.extents.max()
mesh.apply_scale(scale_factor)

# save the new mesh to OBJ file
mesh.export('output.obj')