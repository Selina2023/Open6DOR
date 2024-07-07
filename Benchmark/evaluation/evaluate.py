import numpy as np
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

if isinstance(axis, np.ndarray):
    deviation = projection(rot_gt, rot_pred, axis)
    print(f"Deviation along axis {axis}: {deviation}")

    



