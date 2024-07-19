import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def projection(rot_mat_A, rot_mat_B, axis):
    """
    Project the relative rotation from A to B onto the axis.
    rot_mat: 3x3 rotation matrix 
        A: ground truth rotation
        B: predicted rotation
    axis: 3x1 vector
    """
    det = np.linalg.det(rot_mat_A)
    assert det != 0 # rotation matrix should have determinant +1 or -1
    v = np.linalg.inv(rot_mat_A) @ axis
  
    w = rot_mat_B @ v  
    angle = np.arccos(np.dot(axis, w) / (np.linalg.norm(axis) * np.linalg.norm(w)))
    return np.degrees(angle)

# quat_gt = [0.884556,-0.093848,-0.436286,0.135678]
quat_gt = [-0.205673,-0.205673,-0.596955,0.772278]
rot_gt = R.from_quat(quat_gt).as_matrix()
# quat_pred = [0.972568,-0.128846,-0.164,0.103027]
# quat_pred = [0.546952,-0.013245,-0.820748,0.16444]
# quat_pred = [0.450043,-0.310077,-0.760036,0.351651]
# quat_pred = [0.270194,-0.590044,-0.570659,0.503183]

# quat_pred = [0.166216,-0.492937,-0.609121,0.59863]
# quat_pred = [-0.058748,-0.690237,-0.377434,-0.377434]



quat_pred = [0.107351,-0.684364,-0.220191,0.68676]
rot_pred = R.from_quat(quat_pred).as_matrix()
ax = "y"
axis = ax
if ax == "x":
    axis = np.array([1, 0, 0])
elif ax == "y":
    axis = np.array([0, 1, 0])
elif ax == "z":
    axis = np.array([0, 0, 1])

# if isinstance(axis, np.ndarray):
#     deviation = projection(rot_gt, rot_pred, axis)
#     print(f"Deviation along axis {axis}: {deviation}")

    
def normalize_quat(quat):
    norm = math.sqrt(sum(q ** 2 for q in quat))
    return [q / norm for q in quat]

def angle_deviation(quat0, quat1):
    # Normalize the quaternions
    quat0 = normalize_quat(quat0)
    quat1 = normalize_quat(quat1)
    
    # Compute the dot product of the two quaternions
    dot_product = sum(q0 * q1 for q0, q1 in zip(quat0, quat1))
    
    # Ensure the dot product is within the range [-1, 1] to avoid numerical errors
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Compute the angle deviation (in radians)
    angle_deviation = 2 * math.acos(dot_product)
    
    # Convert the angle deviation to degrees
    angle_deviation_degrees = math.degrees(angle_deviation)
    
    return angle_deviation_degrees

# # Example usage
# quat0 = [0.7071, 0.0, 0.7071, 0.0]  # Example quaternion 0
# quat1 = [0.7, 0.0, 0.9, 0.0]  # Example quaternion 1

# angle_deviation = angle_deviation(quat0, quat1)
# print(f"Angle deviation: {angle_deviation} degrees")



def evaluate_rot(quat_gt, quat_pred):
    """
    Evaluate the predicted rotation.
    task_id: str
    quat_pred: list of 4 floats
    """
    # load the ground truth quaternion
   
    rot_gt = R.from_quat(quat_gt).as_matrix()
    rot_pred = R.from_quat(quat_pred).as_matrix()
    task_level = 0#TODO: load task level from the dataset
    obj_category = 0#TODO: load object category from the dataset
    if task_level == 0:
        ax = "z"
    elif task_level == 1:
        ax = "y"
        if obj_category in ["mug", "binder_clips", "toy", "wallet", "headphone"] :
            ax = "n"
    elif task_level == 2:
        ax = 0#TODO: load axis from the dataset
    else:
        raise ValueError(f"Invalid task level: {task_level}")
    axis = ax
    if ax == "x":
        axis = np.array([1, 0, 0])
    elif ax == "y":
        axis = np.array([0, 1, 0])
    elif ax == "z":
        axis = np.array([0, 0, 1])

    deviation = -1
    if isinstance(axis, np.ndarray):
        deviation = projection(rot_gt, rot_pred, axis)
    else:
        deviation = angle_deviation(quat_gt, quat_pred)
    
    return deviation


def evaluate_posi(sel_pos, tar_pos, mode):
    """
    Evaluate the predicted position.
    """
    if mode in ["left", "right", "front", "back", "behind", "top"]:
        if mode == "left":
            succ += sel_pos[1] > tar_pos[1]
        elif mode == "right":
            succ += sel_pos[1] < tar_pos[1]
        elif mode == "front":
            succ += sel_pos[0] > tar_pos[0]
        elif mode == "back" or mode == "behind":
            succ += sel_pos[0] < tar_pos[0]
        elif mode == "top":
            succ += sel_pos[2] <= tar_pos[2]
    elif mode == "between":
        max_sel_pos_x = np.max([sel_pos_1[0], sel_pos_2[0]])
        max_sel_pos_y = np.max([sel_pos_1[1], sel_pos_2[1]])
        min_sel_pos_x = np.min([sel_pos_1[0], sel_pos_2[0]])
        min_sel_pos_y = np.min([sel_pos_1[1], sel_pos_2[1]])
        tar_pos = result["final_obj_pos"][-1]
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) or (min_sel_pos_y < tar_pos[0] < max_sel_pos_y)
    elif mode == "center":
        max_sel_pos_x = np.max(sel_pos_all, axis=0)[0]
        min_sel_pos_x = np.min(sel_pos_all, axis=0)[0]
        max_sel_pos_y = np.max(sel_pos_all, axis=0)[1]
        min_sel_pos_y = np.min(sel_pos_all, axis=0)[1]
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) and (min_sel_pos_y < tar_pos[1] < max_sel_pos_y)