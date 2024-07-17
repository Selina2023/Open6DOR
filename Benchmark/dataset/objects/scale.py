import trimesh
import os
import json
import math


mesh_path = '/Users/selina/Desktop/projects/ObjectPlacement/assets/mesh/final_norm'
category_path = '/Users/selina/Desktop/projects/Open6DOR/Benchmark/benchmark_catalogue/category_dictionary.json'
object_path = '/Users/selina/Desktop/projects/Open6DOR/Benchmark/benchmark_catalogue/object_dictionary_complete_0702.json'
new_path = "/Users/selina/Desktop/projects/Open6DOR/Benchmark/dataset/objects/rescale"

category_dict = json.load(open(category_path, 'r'))
object_dict = json.load(open(object_path, 'r'))
for root, dirs, files in os.walk(mesh_path):
    for dir in dirs:
        try:
            obj_dir = os.path.join(root, dir)
            obj_name = dir
            if obj_name not in object_dict:
                continue
            obj_cat = object_dict[obj_name]['category']
            obj_scale = category_dict[obj_cat]['scale']
            obj_mesh = trimesh.load(os.path.join(mesh_path, dir) + '/material.obj')
            
            obj_mesh.apply_translation(-obj_mesh.centroid)
        
            if obj_mesh.bounding_box.extents.max() < 0.1:
                print(f"Object {obj_name} is too small")
                continue
            scale_factor = 0.7 * math.sqrt(obj_scale) / obj_mesh.bounding_box.extents.max()

            obj_mesh.apply_scale(scale_factor)
            if not os.path.exists(os.path.join(new_path, dir)):
                os.makedirs(os.path.join(new_path, dir), exist_ok=False)
                obj_mesh.export(os.path.join(new_path, dir) + '/material.obj')
        except:
            import pdb; pdb.set_trace()
       
        
    break

# # Load a mesh from OBJ file
# mesh = trimesh.load('/Users/selina/Desktop/projects/Open6DOR/Benchmark/dataset/objects/rescale/c61227cac7224b86b43c53ac2a2b6ec7/material.obj')

# # Translate mesh to its centroid
# mesh.apply_translation(-mesh.centroid)

# # Scale the mesh (1 unit here)
# # scale_factor = 1.0 / mesh.bounding_box.extents.max()
# print(mesh.bounding_box.extents.max())
# # mesh.apply_scale(scale_factor)

# # # save the new mesh to OBJ file
# # mesh.export('2ab18cb4ec8f4a1f8dec637602362054.obj')