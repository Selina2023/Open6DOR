import numpy as np
import json



def quaternion_to_matrix(q):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

def create_transformation_matrix(position, quaternion):
    """
    Create a 4x4 transformation matrix from position and quaternion.
    """
    x, y, z = position
    q = quaternion
    
    rotation_matrix = quaternion_to_matrix(q)
    
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]
    
    return transformation_matrix

config_path = "output/gym_outputs_task_gen_obja_0304_rot/center/Place_the_mouse_at_the_center_of_all_the_objects_on_the_table.__upright/20240630-202931_no_interaction/task_config.json"

config = json.load(open(config_path, "r"))
pos_s = config["init_obj_pos"]
for pos in pos_s:
    position = pos[:3]
    quaternion = pos[3:7]  # Example quaternion
    transformation_matrix = create_transformation_matrix(position, quaternion)

    print(transformation_matrix)
