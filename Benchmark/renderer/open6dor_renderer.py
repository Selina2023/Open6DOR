
import os
# os.system('pip install scipy')
import random
import bpy
import math
import numpy as np
from mathutils import Vector, Matrix
import copy
import sys
import json
import glob
import time
# import scipy
# from scipy.spatial.transform import Rotation as R
# from transforms3d.quaternions import quat2mat, mat2quat



sys.path.append(os.getcwd())
from modify_material import set_modify_material, set_modify_raw_material

RENDERING_PATH = os.getcwd()

###########################
# Parameter setting
###########################
working_root = "."                          # working root path
env_map_path = os.path.join(working_root, "envmap_lib")                                # environment map path
# output_root_path = os.path.join(working_root, "output_png", "with_trans")                        # rendered output path
output_root_path = os.path.join(working_root, "output/Open6DOR", "test")
DEVICE_LIST = [0]                                                                       # GPU id


# material randomization mode (transparent, specular, mixed, raw)
my_material_randomize_mode = 'mixed'

# set depth sensor parameter
camera_width = 1280
camera_height = 720
camera_fov = 71.28 / 180 * math.pi
baseline_distance = 0.055
# num_frame_per_scene = 10    # number of cameras per scene
LIGHT_EMITTER_ENERGY = 5
LIGHT_ENV_MAP_ENERGY_IR = 0.035
LIGHT_ENV_MAP_ENERGY_RGB = 0.5

#########################################
# set background parameter
background_size = 3.
background_position = (-0.15, -0.15, 0.)
background_scale = (1., 1., 1.)

# set camera randomized paramater
# start_point_range: (range_r, range_vector),   range_r: (r_min, r_max),    range_vector: (x_min, x_max, y_min, y_max)
# look_at_range: (x_min, x_max, y_min, y_max, z_min, z_max)
# up_range: (x_min, x_max, y_min, y_max)
start_point_range = ((0.8, 1.0), (0, 0, 0.6, -0.6))

up_range = (0.0, 0.0, 0.0, 0.0)
look_at_range = (background_position[0], background_position[0], 
                 background_position[1], background_position[1],
                 background_position[2], background_position[2])


###########################
# Utils
###########################

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)

def quaternionToRotation(q):
    w, x, y, z = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y + 2 * w * z
    r02 = 2 * x * z - 2 * w * y

    r10 = 2 * x * y - 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z + 2 * w * x

    r20 = 2 * x * z + 2 * w * y
    r21 = 2 * y * z - 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    return r

def quaternionFromRotMat(rotation_matrix):
    rotation_matrix = np.reshape(rotation_matrix, (1, 9))[0]
    w = math.sqrt(rotation_matrix[0]+rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    x = math.sqrt(rotation_matrix[0]-rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    y = math.sqrt(-rotation_matrix[0]+rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    z = math.sqrt(-rotation_matrix[0]-rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    a = [w,x,y,z]
    m = a.index(max(a))
    if m == 0:
        x = (rotation_matrix[7]-rotation_matrix[5])/(4*w)
        y = (rotation_matrix[2]-rotation_matrix[6])/(4*w)
        z = (rotation_matrix[3]-rotation_matrix[1])/(4*w)
    if m == 1:
        w = (rotation_matrix[7]-rotation_matrix[5])/(4*x)
        y = (rotation_matrix[1]+rotation_matrix[3])/(4*x)
        z = (rotation_matrix[6]+rotation_matrix[2])/(4*x)
    if m == 2:
        w = (rotation_matrix[2]-rotation_matrix[6])/(4*y)
        x = (rotation_matrix[1]+rotation_matrix[3])/(4*y)
        z = (rotation_matrix[5]+rotation_matrix[7])/(4*y)
    if m == 3:
        w = (rotation_matrix[3]-rotation_matrix[1])/(4*z)
        x = (rotation_matrix[6]+rotation_matrix[2])/(4*z)
        y = (rotation_matrix[5]+rotation_matrix[7])/(4*z)
    quaternion = (w,x,y,z)
    return quaternion

def rotVector(q, vector_ori):
    r = quaternionToRotation(q)
    x_ori = vector_ori[0]
    y_ori = vector_ori[1]
    z_ori = vector_ori[2]
    x_rot = r[0][0] * x_ori + r[1][0] * y_ori + r[2][0] * z_ori
    y_rot = r[0][1] * x_ori + r[1][1] * y_ori + r[2][1] * z_ori
    z_rot = r[0][2] * x_ori + r[1][2] * y_ori + r[2][2] * z_ori
    return (x_rot, y_rot, z_rot)

def cameraLPosToCameraRPos(q_l, pos_l, baseline_dis):
    vector_camera_l_y = (1, 0, 0)
    vector_rot = rotVector(q_l, vector_camera_l_y)
    pos_r = (pos_l[0] + vector_rot[0] * baseline_dis,
             pos_l[1] + vector_rot[1] * baseline_dis,
             pos_l[2] + vector_rot[2] * baseline_dis)
    return pos_r

def getRTFromAToB(pointCloudA, pointCloudB):

    muA = np.mean(pointCloudA, axis=0)
    muB = np.mean(pointCloudB, axis=0)

    zeroMeanA = pointCloudA - muA
    zeroMeanB = pointCloudB - muB

    covMat = np.matmul(np.transpose(zeroMeanA), zeroMeanB)
    U, S, Vt = np.linalg.svd(covMat)
    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T
    T = (-np.matmul(R, muA.T) + muB.T).reshape(3, 1)
    return R, T


##################################
def cameraPositionRandomize(start_point_range, look_at_range, up_range):
    r_range, vector_range = start_point_range
    r_min, r_max = r_range
    x_min, x_max, y_min, y_max = vector_range
    r = random.uniform(r_min, r_max)
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = math.sqrt(1 - x**2 - y**2)
    vector_camera_axis = np.array([x, y, z])

    x_min, x_max, y_min, y_max = up_range
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)    
    z = math.sqrt(1 - x**2 - y**2)
    up = np.array([x, y, z])

    x_min, x_max, y_min, y_max, z_min, z_max = look_at_range
    look_at = np.array([random.uniform(x_min, x_max),
                        random.uniform(y_min, y_max),
                        random.uniform(z_min, z_max)])
    position = look_at + r * vector_camera_axis

    vectorZ = - (look_at - position)/np.linalg.norm(look_at - position)
    vectorX = np.cross(up, vectorZ)/np.linalg.norm(np.cross(up, vectorZ))
    vectorY = np.cross(vectorZ, vectorX)/np.linalg.norm(np.cross(vectorX, vectorZ))

    # points in camera coordinates
    pointSensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

    # points in world coordinates 
    pointWorld = np.array([position,
                            position + vectorX,
                            position + vectorY * 2,
                            position + vectorZ * 3])

    resR, resT = getRTFromAToB(pointSensor, pointWorld)
    resQ = quaternionFromRotMat(resR)
    return resQ, resT, vector_camera_axis    

def quanternion_mul(q1, q2):
    s1 = q1[0]
    v1 = np.array(q1[1:])
    s2 = q2[0]
    v2 = np.array(q2[1:])
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return (s, v[0], v[1], v[2])

import math

def rotation_matrix_to_euler(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to Euler angles in radians.

    Parameters:
    - rotation_matrix: 3x3 rotation matrix (list of lists or numpy array)

    Returns:
    - euler_angles: Euler angles in radians (list with shape (3,))
    """
    sy = math.sqrt(rotation_matrix[0][0] * rotation_matrix[0][0] +  rotation_matrix[1][0] * rotation_matrix[1][0])
    
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2][1], rotation_matrix[2][2])
        y = math.atan2(-rotation_matrix[2][0], sy)
        z = math.atan2(rotation_matrix[1][0], rotation_matrix[0][0])
    else:
        x = math.atan2(-rotation_matrix[1][2], rotation_matrix[1][1])
        y = math.atan2(-rotation_matrix[2][0], sy)
        z = 0

    return [x, y, z]



def setModelPose(instance, location, rotation):
    instance.rotation_mode = 'XYZ'
    # instance.rotation_euler = R.from_matrix(rotation).as_euler('xyz', degrees=False)
    instance.rotation_euler = rotation_matrix_to_euler(rotation)
    # instance.rotation_quaternion = rotation

    instance.location = location



def setRigidBody(instance):
    bpy.context.view_layer.objects.active = instance 
    object_single = bpy.context.active_object

    # add rigid body constraints to cube
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.mass = 1
    bpy.context.object.rigid_body.kinematic = True
    bpy.context.object.rigid_body.collision_shape = 'CONVEX_HULL'
    bpy.context.object.rigid_body.restitution = 0.01
    bpy.context.object.rigid_body.angular_damping = 0.8
    bpy.context.object.rigid_body.linear_damping = 0.99

    bpy.context.object.rigid_body.kinematic = False
    object_single.keyframe_insert(data_path='rigid_body.kinematic', frame=0)






###########################
# Renderer Class
###########################
class BlenderRenderer(object):

    def __init__(self, viewport_size_x=640, viewport_size_y=360):
        '''
        viewport_size_x, viewport_size_y: rendering viewport resolution
        '''

        # remove all objects, cameras and lights
        for obj in bpy.data.meshes:
            bpy.data.meshes.remove(obj)

        for cam in bpy.data.cameras:
            bpy.data.cameras.remove(cam)

        for light in bpy.data.lights:
            bpy.data.lights.remove(light)

        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # remove all materials
        # for item in bpy.data.materials:
        #     bpy.data.materials.remove(item)

        render_context = bpy.context.scene.render

        # add left camera
        camera_l_data = bpy.data.cameras.new(name="camera_l")
        camera_l_object = bpy.data.objects.new(name="camera_l", object_data=camera_l_data)
        bpy.context.collection.objects.link(camera_l_object)

        # add right camera
        camera_r_data = bpy.data.cameras.new(name="camera_r")
        camera_r_object = bpy.data.objects.new(name="camera_r", object_data=camera_r_data)
        bpy.context.collection.objects.link(camera_r_object)

        camera_l = bpy.data.objects["camera_l"]
        camera_r = bpy.data.objects["camera_r"]

        # set the camera postion and orientation so that it is in
        # front of the object
        camera_l.location = (1, 0, 0)
        camera_r.location = (1, 0, 0)

        # add emitter light
        light_emitter_data = bpy.data.lights.new(name="light_emitter", type='SPOT')
        light_emitter_object = bpy.data.objects.new(name="light_emitter", object_data=light_emitter_data)
        bpy.context.collection.objects.link(light_emitter_object)

        light_emitter = bpy.data.objects["light_emitter"]
        light_emitter.location = (1, 0, 0)
        light_emitter.data.energy = LIGHT_EMITTER_ENERGY

        # render setting
        render_context.resolution_percentage = 100
        self.render_context = render_context

        self.camera_l = camera_l
        self.camera_r = camera_r

        self.light_emitter = light_emitter

        self.model_loaded = False
        self.background_added = None

        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

        self.my_material = {}
        self.render_mode = 'IR'

        # output setting 
        self.render_context.image_settings.file_format = 'PNG'
        self.render_context.image_settings.compression = 0
        self.render_context.image_settings.color_mode = 'BW'
        self.render_context.image_settings.color_depth = '8'

        # cycles setting
        self.render_context.engine = 'CYCLES'
        bpy.context.scene.cycles.progressive = 'BRANCHED_PATH'
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'NLM'
        bpy.context.scene.cycles.film_exposure = 0.5

        # self.render_context.use_antialiasing = False
        bpy.context.scene.view_layers["View Layer"].use_sky = True

        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
  
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
  
        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')

        # create output node
        self.fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        self.fileOutput.base_path = "./new_data/0000"
        self.fileOutput.format.file_format = 'OPEN_EXR'
        self.fileOutput.format.color_depth= '32'
        self.fileOutput.file_slots[0].path = 'depth#'
        # links.new(map.outputs[0], fileOutput.inputs[0])
        links.new(rl.outputs[2], self.fileOutput.inputs[0])
        # links.new(gamma.outputs[0], fileOutput.inputs[0])

        # depth sensor pattern
        self.pattern = []
        # environment map
        self.env_map = []


    def loadImages(self, env_map_path):
        for img in bpy.data.images:
            if img.filepath.split("/")[-1] == "pattern.png":
                self.pattern = img
                break
        for item in os.listdir(env_map_path):
            if item.split('.')[-1] == 'hdr':
                # import pdb; pdb.set_trace()
                self.env_map.append(bpy.data.images.load(filepath=os.path.join(env_map_path, item)))


    def addEnvMap(self):
        # Get the environment node tree of the current scene
        node_tree = bpy.context.scene.world.node_tree
        tree_nodes = node_tree.nodes

        # Clear all nodes
        tree_nodes.clear()

        # Add Background node
        node_background = tree_nodes.new(type='ShaderNodeBackground')

        # Add Environment Texture node
        node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
        # Load and assign the image to the node property
       
        node_environment.location = -300,0

        node_tex_coord = tree_nodes.new(type='ShaderNodeTexCoord')
        node_tex_coord.location = -700,0

        node_mapping = tree_nodes.new(type='ShaderNodeMapping')
        node_mapping.location = -500,0

        # Add Output node
        node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
        node_output.location = 200,0

        # Link all nodes
        links = node_tree.links
        links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        links.new(node_tex_coord.outputs["Generated"], node_mapping.inputs["Vector"])
        links.new(node_mapping.outputs["Vector"], node_environment.inputs["Vector"])

        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0


    def setEnvMap(self, env_map_id, rotation_elur_z):
        # Get the environment node tree of the current scene
        node_tree = bpy.context.scene.world.node_tree

        # Get Environment Texture node
        node_environment = node_tree.nodes['Environment Texture']
        # Load and assign the image to the node property
        node_environment.image = self.env_map[env_map_id]

        node_mapping = node_tree.nodes['Mapping']
        node_mapping.inputs[2].default_value[2] = rotation_elur_z


    def addMaskMaterial(self, num=20):
        material_name = "mask_background"

        # test if material exists
        # if it does not exist, create it:
        material_class = (bpy.data.materials.get(material_name) or 
            bpy.data.materials.new(material_name))

        # enable 'Use nodes'
        material_class.use_nodes = True
        node_tree = material_class.node_tree

        # remove default nodes
        material_class.node_tree.nodes.clear()

        # add new nodes  
        node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_2= node_tree.nodes.new('ShaderNodeBrightContrast')

        # link nodes
        node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        node_2.inputs[0].default_value = (1, 1, 1, 1)
        self.my_material[material_name] =  material_class

        for i in range(num):
            class_name = str(i + 1)
            # set the material of background    
            material_name = "mask_" + class_name

            # test if material exists
            # if it does not exist, create it:
            material_class = (bpy.data.materials.get(material_name) or 
                bpy.data.materials.new(material_name))

            # enable 'Use nodes'
            material_class.use_nodes = True
            node_tree = material_class.node_tree

            # remove default nodes
            material_class.node_tree.nodes.clear()

            # add new nodes  
            node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
            node_2= node_tree.nodes.new('ShaderNodeBrightContrast')

            # link nodes
            node_tree.links.new(node_1.inputs[0], node_2.outputs[0])

            if class_name.split('_')[0] == 'background':
                node_2.inputs[0].default_value = (1, 1, 1, 1)
            else:
                node_2.inputs[0].default_value = ((i + 1)/255., 0., 0., 1)

            self.my_material[material_name] =  material_class


    def addNOCSMaterial(self):
        material_name = 'coord_color'
        mat = (bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))

        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        nodes.clear()        

        links = node_tree.links
        links.clear()

        vcol_R = nodes.new(type="ShaderNodeVertexColor")
        vcol_R.layer_name = "Col_R" # the vertex color layer name
        vcol_G = nodes.new(type="ShaderNodeVertexColor")
        vcol_G.layer_name = "Col_G" # the vertex color layer name
        vcol_B = nodes.new(type="ShaderNodeVertexColor")
        vcol_B.layer_name = "Col_B" # the vertex color layer name

        node_Output = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_Emission = node_tree.nodes.new('ShaderNodeEmission')
        node_LightPath = node_tree.nodes.new('ShaderNodeLightPath')
        node_Mix = node_tree.nodes.new('ShaderNodeMixShader')
        node_Combine = node_tree.nodes.new(type="ShaderNodeCombineRGB")


        # make links
        node_tree.links.new(vcol_R.outputs[1], node_Combine.inputs[0])
        node_tree.links.new(vcol_G.outputs[1], node_Combine.inputs[1])
        node_tree.links.new(vcol_B.outputs[1], node_Combine.inputs[2])
        node_tree.links.new(node_Combine.outputs[0], node_Emission.inputs[0])

        node_tree.links.new(node_LightPath.outputs[0], node_Mix.inputs[0])
        node_tree.links.new(node_Emission.outputs[0], node_Mix.inputs[2])
        node_tree.links.new(node_Mix.outputs[0], node_Output.inputs[0])

        self.my_material[material_name] = mat


    def addNormalMaterial(self):
        material_name = 'normal'
        mat = (bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))
        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        nodes.clear()
            
        links = node_tree.links
        links.clear()
            
        # Nodes:
        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (151.59744262695312, 854.5482177734375)
        new_node.name = 'Math'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = 1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeLightPath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (602.9912719726562, 1046.660888671875)
        new_node.name = 'Light Path'
        new_node.select = False
        new_node.width = 140.0
        new_node.outputs[0].default_value = 0.0
        new_node.outputs[1].default_value = 0.0
        new_node.outputs[2].default_value = 0.0
        new_node.outputs[3].default_value = 0.0
        new_node.outputs[4].default_value = 0.0
        new_node.outputs[5].default_value = 0.0
        new_node.outputs[6].default_value = 0.0
        new_node.outputs[7].default_value = 0.0
        new_node.outputs[8].default_value = 0.0
        new_node.outputs[9].default_value = 0.0
        new_node.outputs[10].default_value = 0.0
        new_node.outputs[11].default_value = 0.0
        new_node.outputs[12].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeOutputMaterial')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.is_active_output = True
        new_node.location = (1168.93017578125, 701.84033203125)
        new_node.name = 'Material Output'
        new_node.select = False
        new_node.target = 'ALL'
        new_node.width = 140.0
        new_node.inputs[2].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeBsdfTransparent')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (731.72900390625, 721.4832763671875)
        new_node.name = 'Transparent BSDF'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0]

        new_node = nodes.new(type='ShaderNodeCombineXYZ')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (594.4229736328125, 602.9271240234375)
        new_node.name = 'Combine XYZ'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.0
        new_node.inputs[1].default_value = 0.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeMixShader')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (992.7239990234375, 707.2142333984375)
        new_node.name = 'Mix Shader'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5

        new_node = nodes.new(type='ShaderNodeEmission')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (774.0802612304688, 608.2547607421875)
        new_node.name = 'Emission'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0]
        new_node.inputs[1].default_value = 1.0

        new_node = nodes.new(type='ShaderNodeSeparateXYZ')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (-130.12167358398438, 558.1497802734375)
        new_node.name = 'Separate XYZ'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[0].default_value = 0.0
        new_node.outputs[1].default_value = 0.0
        new_node.outputs[2].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (162.43240356445312, 618.8094482421875)
        new_node.name = 'Math.002'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = 1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (126.8158187866211, 364.5539855957031)
        new_node.name = 'Math.001'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = -1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeVectorTransform')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.convert_from = 'WORLD'
        new_node.convert_to = 'CAMERA'
        new_node.location = (-397.0209045410156, 594.7037353515625)
        new_node.name = 'Vector Transform'
        new_node.select = False
        new_node.vector_type = 'VECTOR'
        new_node.width = 140.0
        new_node.inputs[0].default_value = [0.5, 0.5, 0.5]
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeNewGeometry')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (-651.8067016601562, 593.0455932617188)
        new_node.name = 'Geometry'
        new_node.width = 140.0
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[1].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[2].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[3].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[4].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[5].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[6].default_value = 0.0
        new_node.outputs[7].default_value = 0.0
        new_node.outputs[8].default_value = 0.0

        # Links :

        links.new(nodes["Light Path"].outputs[0], nodes["Mix Shader"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[0], nodes["Math"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[1], nodes["Math.002"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[2], nodes["Math.001"].inputs[0])    
        links.new(nodes["Vector Transform"].outputs[0], nodes["Separate XYZ"].inputs[0])    
        links.new(nodes["Combine XYZ"].outputs[0], nodes["Emission"].inputs[0])    
        links.new(nodes["Math"].outputs[0], nodes["Combine XYZ"].inputs[0])    
        links.new(nodes["Math.002"].outputs[0], nodes["Combine XYZ"].inputs[1])    
        links.new(nodes["Math.001"].outputs[0], nodes["Combine XYZ"].inputs[2])    
        links.new(nodes["Transparent BSDF"].outputs[0], nodes["Mix Shader"].inputs[1])    
        links.new(nodes["Emission"].outputs[0], nodes["Mix Shader"].inputs[2])    
        links.new(nodes["Mix Shader"].outputs[0], nodes["Material Output"].inputs[0])    
        links.new(nodes["Geometry"].outputs[1], nodes["Vector Transform"].inputs[0])    

        self.my_material[material_name] = mat



    def setCamera(self, quaternion, translation, fov, baseline_distance):
        self.camera_l.data.angle = fov
        self.camera_r.data.angle = self.camera_l.data.angle
        cx = translation[0]
        cy = translation[1]
        cz = translation[2]

        self.camera_l.location[0] = cx
        self.camera_l.location[1] = cy 
        self.camera_l.location[2] = cz

        self.camera_l.rotation_mode = 'QUATERNION'
        self.camera_l.rotation_quaternion[0] = quaternion[0]
        self.camera_l.rotation_quaternion[1] = quaternion[1]
        self.camera_l.rotation_quaternion[2] = quaternion[2]
        self.camera_l.rotation_quaternion[3] = quaternion[3]

        self.camera_r.rotation_mode = 'QUATERNION'
        self.camera_r.rotation_quaternion[0] = quaternion[0]
        self.camera_r.rotation_quaternion[1] = quaternion[1]
        self.camera_r.rotation_quaternion[2] = quaternion[2]
        self.camera_r.rotation_quaternion[3] = quaternion[3]
        cx, cy, cz = cameraLPosToCameraRPos(quaternion, (cx, cy, cz), baseline_distance)
        self.camera_r.location[0] = cx
        self.camera_r.location[1] = cy 
        self.camera_r.location[2] = cz


    def setLighting(self):
        # emitter        
        #self.light_emitter.location = self.camera_r.location
        self.light_emitter.location = self.camera_l.location #+ 0.51 * (self.camera_r.location - self.camera_l.location)
        self.light_emitter.rotation_mode = 'QUATERNION'
        # self.light_emitter.rotation_quaternion = self.camera_r.rotation_quaternion
        self.light_emitter.rotation_quaternion = self.camera_l.rotation_quaternion

        # emitter setting
        bpy.context.view_layer.objects.active = None
        # bpy.ops.object.select_all(action="DESELECT")
        self.render_context.engine = 'CYCLES'
        self.light_emitter.select_set(True)
        self.light_emitter.data.use_nodes = True
        self.light_emitter.data.type = "POINT"
        self.light_emitter.data.shadow_soft_size = 0.001
        random_energy = LIGHT_EMITTER_ENERGY#random.uniform(LIGHT_EMITTER_ENERGY * 0.9, LIGHT_EMITTER_ENERGY * 1.1)
        self.light_emitter.data.energy = random_energy

        # remove default node
        light_emitter = bpy.data.objects["light_emitter"].data
        light_emitter.node_tree.nodes.clear()

        # add new nodes
        light_output = light_emitter.node_tree.nodes.new("ShaderNodeOutputLight")
        node_1 = light_emitter.node_tree.nodes.new("ShaderNodeEmission")
        node_2 = light_emitter.node_tree.nodes.new("ShaderNodeTexImage")
        node_3 = light_emitter.node_tree.nodes.new("ShaderNodeMapping")
        node_4 = light_emitter.node_tree.nodes.new("ShaderNodeVectorMath")
        node_5 = light_emitter.node_tree.nodes.new("ShaderNodeSeparateXYZ")
        node_6 = light_emitter.node_tree.nodes.new("ShaderNodeTexCoord")

        # link nodes
        light_emitter.node_tree.links.new(light_output.inputs[0], node_1.outputs[0])
        light_emitter.node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        light_emitter.node_tree.links.new(node_2.inputs[0], node_3.outputs[0])
        light_emitter.node_tree.links.new(node_3.inputs[0], node_4.outputs[0])
        light_emitter.node_tree.links.new(node_4.inputs[0], node_6.outputs[1])
        light_emitter.node_tree.links.new(node_4.inputs[1], node_5.outputs[2])
        light_emitter.node_tree.links.new(node_5.inputs[0], node_6.outputs[1])

        # set parameter of nodes
        node_1.inputs[1].default_value = 1.0        # scale
        node_2.extension = 'CLIP'
        # node_2.interpolation = 'Cubic'

        node_3.inputs[1].default_value[0] = 0.5
        node_3.inputs[1].default_value[1] = 0.5
        node_3.inputs[1].default_value[2] = 0
        node_3.inputs[2].default_value[0] = 0
        node_3.inputs[2].default_value[1] = 0
        node_3.inputs[2].default_value[2] = 0.05

        # scale of pattern
        node_3.inputs[3].default_value[0] = 0.6
        node_3.inputs[3].default_value[1] = 0.85
        node_3.inputs[3].default_value[2] = 0
        node_4.operation = 'DIVIDE'

        # pattern path
        node_2.image = self.pattern


    def lightModeSelect(self, light_mode):
        if light_mode == "RGB":
            self.light_emitter.hide_render = True
            # set the environment map energy
            random_energy = random.uniform(LIGHT_ENV_MAP_ENERGY_RGB * 0.9, LIGHT_ENV_MAP_ENERGY_RGB * 1.1)
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random_energy

        elif light_mode == "IR":
            self.light_emitter.hide_render = False
            # set the environment map energy
            random_energy = random.uniform(LIGHT_ENV_MAP_ENERGY_IR * 0.9, LIGHT_ENV_MAP_ENERGY_IR * 1.1)
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random_energy
        
        elif light_mode == "Mask" or light_mode == "NOCS" or light_mode == "Normal":
            self.light_emitter.hide_render = True
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0

        else:
            print("Don't support this lightmode:", light_mode)    


    def outputModeSelect(self, output_mode):
        if output_mode == "RGB":
            self.render_context.image_settings.file_format = 'PNG'
            self.render_context.image_settings.compression = 0
            self.render_context.image_settings.color_mode = 'RGB'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Filmic'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 1280
            self.render_context.resolution_y = 720
        elif output_mode == "IR":
            self.render_context.image_settings.file_format = 'PNG'
            self.render_context.image_settings.compression = 0
            self.render_context.image_settings.color_mode = 'BW'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Filmic'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 1280
            self.render_context.resolution_y = 720
        elif output_mode == "Mask":
            self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.color_mode = 'RGB'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 0
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        elif output_mode == "NOCS":
            # self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.file_format = 'PNG'            
            self.render_context.image_settings.color_mode = 'RGB'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 0
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        elif output_mode == "Normal":
            self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.color_mode = 'RGB'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        else:
            print("Not support the mode!")    


    def renderEngineSelect(self, engine_mode):

        if engine_mode == "CYCLES":
            self.render_context.engine = 'CYCLES'
            bpy.context.scene.cycles.progressive = 'BRANCHED_PATH'
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.denoiser = 'NLM'
            bpy.context.scene.cycles.film_exposure = 1.0
            bpy.context.scene.cycles.aa_samples = 64 #32

            ## Set the device_type
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"
            ## Set the device and feature set
            # bpy.context.scene.cycles.device = "CPU"

            ## get_devices() to let Blender detects GPU device
            cuda_devices, _ = bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                d["use"] = 1 # Using all devices, include GPU and CPU
                print(d["name"], d["use"])
            '''
            '''
            device_list = DEVICE_LIST
            activated_gpus = []
            for i, device in enumerate(cuda_devices):
                if (i in device_list):
                    device.use = True
                    activated_gpus.append(device.name)
                else:
                    device.use = False


        elif engine_mode == "EEVEE":
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        else:
            print("Not support the mode!")    


    def addBackground(self, size, position, scale):
        # set the material of background    
        material_name = "default_background"

        # test if material exists
        # if it does not exist, create it:
        material_background = (bpy.data.materials.get(material_name) or 
            bpy.data.materials.new(material_name))

        # enable 'Use nodes'
        material_background.use_nodes = True
        node_tree = material_background.node_tree

        # remove default nodes
        material_background.node_tree.nodes.clear()
        # material_background.node_tree.nodes.remove(material_background.node_tree.nodes.get('Principled BSDF')) #title of the existing node when materials.new
        # material_background.node_tree.nodes.remove(material_background.node_tree.nodes.get('Material Output')) #title of the existing node when materials.new

        # add new nodes  
        node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_2 = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        node_3 = node_tree.nodes.new('ShaderNodeTexImage')

        # link nodes
        node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        node_tree.links.new(node_2.inputs[0], node_3.outputs[0])

        # add texture image
        node_3.image = bpy.data.images.load(filepath=os.path.join(working_root, "texture/texture_0.jpg"))
        self.my_material['default_background'] = material_background

        # add background plane
        for i in range(-2, 3, 1):
            for j in range(-2, 3, 1):
                position_i_j = (i * size + position[0], j * size + position[1], position[2])
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, scale=scale)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'
        for i in range(-2, 3, 1):
            for j in [-2, 2]:
                position_i_j = (i * size + position[0], j * size + position[1], position[2] - 0.25)
                rotation_elur = (math.pi / 2., 0., 0.)
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, rotation = rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'    
        for j in range(-2, 3, 1):
            for i in [-2, 2]:
                position_i_j = (i * size + position[0], j * size + position[1], position[2] - 0.25)
                rotation_elur = (0, math.pi / 2, 0)
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, rotation = rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'        
        count = 0
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                obj.name = "background_" + str(count)
                obj.data.name = "background_" + str(count)
                obj.active_material = material_background
                count += 1

        self.background_added = True


    def clearModel(self):
        '''
        # delete all meshes
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)
        '''

        # remove all objects except background
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.meshes.remove(obj.data)
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.objects.remove(obj, do_unlink=True)

        # remove all default material
        for mat in bpy.data.materials:
            name = mat.name.split('.')
            if name[0] == 'Material':
                bpy.data.materials.remove(mat)


    def loadModel(self, file_path):
        self.model_loaded = True
        try:
            if file_path.endswith('obj'):
                bpy.ops.import_scene.obj(filepath=file_path)
            elif file_path.endswith('3ds'):
                bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
            elif file_path.endswith('dae'):
                # Must install OpenCollada. Please read README.md
                bpy.ops.wm.collada_import(filepath=file_path)
            else:
                self.model_loaded = False
                raise Exception("Loading failed: %s" % (file_path))
        
        except Exception:
            self.model_loaded = False



    def render(self, image_name="tmp", image_path=RENDERING_PATH):
        # Render the object
        if not self.model_loaded:
            print("Model not loaded.")
            return      

        if self.render_mode == "IR":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("IR")
            self.outputModeSelect("IR")
            self.renderEngineSelect("CYCLES")

        elif self.render_mode == 'RGB':
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("RGB")
            self.outputModeSelect("RGB")
            self.renderEngineSelect("CYCLES")

        elif self.render_mode == "Mask":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("Mask")
            self.outputModeSelect("Mask")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 1

        elif self.render_mode == "NOCS":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("NOCS")
            self.outputModeSelect("NOCS")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 1

        elif self.render_mode == "Normal":
            bpy.context.scene.use_nodes = True
            self.fileOutput.base_path = image_path
            self.fileOutput.file_slots[0].path = image_name[:5] + 'depth_#'

            # set light and render mode
            self.lightModeSelect("Normal")
            self.outputModeSelect("Normal")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 32

        else:
            print("The render mode is not supported")
            return 

        bpy.context.scene.render.filepath = os.path.join(image_path, image_name)
        bpy.ops.render.render(write_still=True)  # save straight to file


       

    def get_instance_pose(self):
        instance_pose = {}
        bpy.context.view_layer.update()
        cam = self.camera_l
        mat_rot_x = Matrix.Rotation(math.radians(180.0), 4, 'X')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                instance_id = obj.name.split('_')[0]
                mat_rel = cam.matrix_world.inverted() @ obj.matrix_world
                # location
                relative_location = [mat_rel.translation[0],
                                     - mat_rel.translation[1],
                                     - mat_rel.translation[2]]
                # rotation
                # relative_rotation_euler = mat_rel.to_euler() # must be converted from radians to degrees
                relative_rotation_quat = [mat_rel.to_quaternion()[0],
                                          mat_rel.to_quaternion()[1],
                                          mat_rel.to_quaternion()[2],
                                          mat_rel.to_quaternion()[3]]
                quat_x = [0, 1, 0, 0]
                quat = quanternion_mul(quat_x, relative_rotation_quat)
                quat = [quat[0], - quat[1], - quat[2], - quat[3]]
                instance_pose[str(instance_id)] = [quat, relative_location]

        return instance_pose




###########################
# Main
###########################



def open6dor_render(output_root_path, task_name, mesh_root, obj_ids, obj_poses, background_material_id = 44, env_map_id = 25, cam_quaternion = [0, 0, 0.342, 0.940], cam_translation = [-0.15, 0.4, 0.7]): 
    max_instance_num = 20


    renderer = BlenderRenderer(viewport_size_x=camera_width, viewport_size_y=camera_height)
    renderer.loadImages(env_map_path)
    renderer.addEnvMap()
    renderer.addBackground(background_size, background_position, background_scale)
    # renderer.addMaterialLib()
    renderer.addMaskMaterial(max_instance_num)
    renderer.addNOCSMaterial()
    renderer.addNormalMaterial()


    # import pdb; pdb.set_trace()
    background_material = bpy.data.materials[f'background_{background_material_id}']
    # bpy.data.objects['background'].active_material = material_selected


    # read objects from floder

#########################

    obj_poses = [obj_poses[obj_code] for obj_code in obj_ids]

    # import pdb; pdb.set_trace()
    for obj_i, obj_code in enumerate(obj_ids):
        instance_path = os.path.join(mesh_root, obj_code, f'material.obj') #.obj
        instance_name = obj_code

        obj_name = ""
        # download CAD model and rename
        renderer.loadModel(instance_path)
        for asset in bpy.data.objects:
            if asset.type == 'MESH':
                name = asset.name
                if "textured" in name:
                    obj_name = name
                    break
        
        obj = bpy.data.objects[obj_name] 
        obj.name = instance_name
        obj.data.name = instance_name
        obj_pose = obj_poses[obj_code] # 4 * 4
        obj_loc = obj_pose[:3, 3]

        obj_rot = obj_pose[:3, :3]

        

        setModelPose(obj, obj_loc, obj_rot)

        # set object as rigid body
        setRigidBody(obj)

        
    
        print("object:", obj.name)

        # set material
        set_modify_raw_material(obj)
        
    
    scene = bpy.data.scenes['Scene']

    

    instance_pose_list = []


    # generate visible objects list
    renderer.setCamera(cam_quaternion, cam_translation, camera_fov, baseline_distance)

    # generate object pose list
    instance_pose_list = renderer.get_instance_pose()

    renderer.setLighting()
    rotation_elur_z = 0.0
    renderer.setEnvMap(env_map_id, rotation_elur_z)
 

    renderer.render_mode = "RGB"
    camera = bpy.data.objects['camera_l']
    scene.camera = camera

    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    render_output_file = os.path.join(output_root_path, task_name)
    save_path = render_output_file
    save_name ='color'
    renderer.render(save_name, save_path)


    # generate meta.txt
    path_meta = os.path.join(render_output_file, "meta.txt")
    if os.path.exists(path_meta):
        os.remove(path_meta)

    file_write_obj = open(path_meta, 'w')
    file_write_obj.write('task_index:')
    file_write_obj.write(task_name)
    file_write_obj.write('\nobjects:')
    for obj_code in obj_ids:
        file_write_obj.write('\n')
        file_write_obj.write(obj_code)
        file_write_obj.write(':\n')
        file_write_obj.write(str(obj_poses[obj_code]))

    file_write_obj.write('\ncamera:\n')

    file_write_obj.write(str(cam_quaternion))
    file_write_obj.write(' ')

    file_write_obj.write('\n')
    file_write_obj.write(str(cam_translation))
    file_write_obj.write(' ')


    file_write_obj.write('\n')
    file_write_obj.write("background material:")
    file_write_obj.write(str(background_material))
    file_write_obj.write('\n')
    file_write_obj.write("env_map_id:")
    file_write_obj.write(str(env_map_id))



    file_write_obj.close()


    output_blend = os.path.join(output_root_path, "blend")
    import time
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(os.path.join(output_blend, task_name), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_blend, task_name, f'render_{time_str}.blend'))

    context = bpy.context
    for ob in context.selected_objects:
        ob.animation_data_clear()

    print(bpy.data.materials) 
    print(len(bpy.data.materials))



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



if __name__ == '__main__':
    output_root_path = "/home/ubuntu/Desktop/projects/DexGraspNet1B-render/rendering/output/Open6DOR/test"
    task_name = "0004"
    # mesh_root = "../../DexGraspNet1B-render/mesh_data_new/meshdata"
    mesh_root = "/Users/selina/Desktop/projects/ObjectPlacement/assets/mesh/final_norm"
    # scene_file = f"/home/ubuntu/Desktop/projects/DexGraspNet1B-render/mesh_data_new/renderdata/{task_name}/object_pose_dict.npz"
    config_file = "/Users/selina/Desktop/projects/Open6DOR/Benchmark/dataset/tasks/test/output/gym_outputs_task_gen_obja_0304_rot/center/Place_the_mouse_at_the_center_of_all_the_objects_on_the_table.__upright/20240630-202931_no_interaction"
    config = json.load(open(config_file, "r"))
    pos_s = config["init_obj_pos"]
    obj_paths = config["selected_urdfs"] # e.g. "objaverse_final_norm/02f7f045679d402da9b9f280030821d4/material_2.urdf"

    obj_ids = [path.split("/")[-2] for path in obj_paths]

    obj_poses = {}


    for i in len(obj_ids):
        pos = pos_s[i]
        id = obj_ids[i]
        position = pos[:3]
        quaternion = pos[3:7] 
        transformation_matrix = create_transformation_matrix(position, quaternion)
        obj_poses[id] = transformation_matrix

    # obj_poses = np.load(scene_file, allow_pickle=True)

 
    background_material_id = 44
    env_map_id = 25
    cam_quaternion = [0, 0, 0.342, 0.940]
    cam_translation = [-0.15, 0.4, 0.7]
    open6dor_render(output_root_path, task_name, mesh_root, obj_poses, background_material_id, env_map_id, cam_quaternion, cam_translation)