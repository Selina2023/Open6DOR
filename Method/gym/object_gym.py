
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import imageio
import open3d as o3d
import cv2
import math
import numpy as np
import torch
import random
import time
import glob
import trimesh as tm
from gym.utils import images_to_video, quat_axis, orientation_error, cube_grasping_yaw, read_yaml_config, \
    get_downsampled_pc, get_point_cloud_from_rgbd, get_point_cloud_from_rgbd_GPU
import os, json
import yaml
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("../")
sys.path.append("../vision")
sys.path.append(". /gym")

if True:
    from vision.grounded_sam_demo import prepare_GroundedSAM_for_inference, inference_one_image
    from gym.vlm_utils import infer_path
    
import trimesh

import plotly.graph_objects as go
import os


if False:
    from sudoai import SudoAI
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.util.logger import setup_curobo_logger
    from curobo.util_file import (
        get_robot_configs_path,
        get_world_configs_path,
        join_path,
        load_yaml,
        )
    from curobo.geom.types import Mesh, WorldConfig, Cuboid
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.util_file import get_robot_path, join_path, load_yaml
    
class ObjectGym():
    def __init__(
            self, 
            cfgs,
            grounded_dino_model = None, 
            sam_predictor = None,
            save_root = None,
            pre_steps = 100,
        ):
        self.cfgs = cfgs
        self.debug = cfgs["debug"]
        self.use_cam = cfgs["cam"]["use_cam"]
        self.steps = cfgs["steps"]
        
        # configure env grid
        self.num_envs = cfgs["num_envs"]
        self.num_per_row = int(math.sqrt(self.num_envs))
        self.spacing = cfgs["env_spacing"]
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        print("Creating %d environments" % self.num_envs)
        
        if self.use_cam:
            self.cam_w = cfgs["cam"]["cam_w"]
            self.cam_h = cfgs["cam"]["cam_h"]
            self.cam_far_plane = cfgs["cam"]["cam_far_plane"]
            self.cam_near_plane = cfgs["cam"]["cam_near_plane"]
            self.horizontal_fov = cfgs["cam"]["cam_horizontal_fov"]
            self.cam_poss = cfgs["cam"]["cam_poss"]
            self.cam_targets = cfgs["cam"]["cam_targets"]
            self.num_cam_per_env = len(self.cam_poss)
            self.point_cloud_bound = cfgs["cam"]["point_cloud_bound"]
            
        # segmentation
        self.franka_seg_id = cfgs["asset"]["franka_seg_id"]
        self.asset_seg_ids = cfgs["asset"]["asset_seg_ids"]
        self.table_seg_id = cfgs["asset"]["table_seg_id"]    
        
        self.headless = cfgs["HEADLESS"]
    
        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        
        # Add custom arguments
        self.args = gymutil.parse_arguments(description="Placement",
            custom_parameters=[
                {"name": "--mode", "type": str, "default": ""},
                {"name": "--task_root", "type": str, "default": "gym_outputs_task_gen_ycb_0229"},
                {"name": "--config", "type": str, "default": "config_render_api2"},
                {"name": "--device", "type": str, "default": "cuda"},
                ]
            )

        # set torch device
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        assert self.args.physics_engine == gymapi.SIM_PHYSX
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu
        
        # Grab controller
        self.controller_name = cfgs["controller"]
        assert self.controller_name in {"ik", "osc", "curobo"}, f"Invalid controller specified -- options are (ik, osc). Got: {self.controller_name}"
       # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        self.asset_root = self.cfgs["asset"]["asset_root"]
        self.save_root = save_root
        self.prepare_franka_asset()
        self.prepare_obj_assets()
        if self.cfgs["USE_ARTI"]:
            self.prepare_arti_obj_assets()
        self.load_env(load_cam=self.use_cam)
        
        
        self.init_observation()
        self.run_steps(pre_steps = pre_steps)
        self.refresh_observation(get_visual_obs=False)
        
        if self.cfgs["INFERENCE_GSAM"] and grounded_dino_model is None and \
            sam_predictor is None:
            self.prepare_groundedsam()
        else:
            self.grounded_dino_model = grounded_dino_model
            self.sam_predictor = sam_predictor
            self.box_threshold = 0.3
            self.text_threshold = 0.25
        if self.cfgs["USE_CUROBO"]:
            self.prepare_curobo(use_mesh=self.cfgs["USE_MESH_COLLISION"])
        if self.cfgs["USE_GRASPNET"]:
            self.prepare_graspnet() 
        if self.cfgs["USE_SUDOAI"]:
            self.prepare_sudo_ai(self.save_root)
        
    def prepare_sudo_ai(self, save_root):
        self.sudoai_api = SudoAI(output_dir=save_root)
        
    def inference_sudo_ai(self, img_path):
        meshfile = self.sudoai_api.image_to_3d(img_path, save_root= self.save_root)
        print(f"Mesh file saved to {meshfile}")
        assert os.path.exists(meshfile), "BUG!"
        glb_path = meshfile
        mesh=trimesh.load(glb_path)
        # os.makedirs('')
        mesh.export(f'{self.save_root}/material.obj')
        
    def prepare_graspnet(self):
        
        from gym.test_files.infer_vis_grasp import MyGraspNet
        self.graspnet = MyGraspNet(self.cfgs["graspnet"])
    
    def inference_graspnet(self, pcs, keep = 1000):
        gg = self.graspnet.inference(pcs)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if self.cfgs["graspnet"]["vis"]:
            if gg.__len__() > keep:
                gg_vis = gg[:keep]
            else:
                gg_vis = gg
            grippers = gg_vis.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, *grippers])   
        
        return gg

    def prepare_curobo(self, use_mesh = False):
        setup_curobo_logger("error")
        tensor_args = TensorDeviceType()
        world_file = "/home/haoran/Projects/ObjectPlacement/gym/curobo/src/curobo/content/configs/world/collision_empty.yml"
        robot_file = "/home/haoran/Projects/ObjectPlacement/gym/curobo/src/curobo/content/configs/robot/franka.yml"
        
        if not use_mesh:
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_file,
                world_file,
                tensor_args,
                interpolation_dt=0.01,
            )
        else:
            asset_files = self.cfgs["asset"]["asset_files"]
            asset_obj_files = [os.path.join(self.cfgs["asset"]["asset_root"],  "/".join(asset_file.split("/")[:-1]), "textured.obj") for asset_file in self.cfgs["asset"]["asset_files"]]
            # import pdb; pdb.set_trace()
            object_meshes = [tm.load(asset_obj_file) for asset_obj_file in asset_obj_files]
            states = self.root_states[2:, :7].cpu().numpy()
            assert len(states) == len(object_meshes), "BUG!"
            obstables = [
                Mesh(
                    name=f'object_{object_meshes_i}', 
                    pose=states[object_meshes_i],
                    vertices=object_meshes[object_meshes_i].vertices,
                    faces=object_meshes[object_meshes_i].faces
                    ) 
                for object_meshes_i in range(len(object_meshes))
                ]
            
            # import pdb; pdb.set_trace()
            table = Cuboid(
                name='table',
                dims=[self.table_scale[0], self.table_scale[1], self.table_scale[2]],
                # dims=[0, 0, 0],
                pose=[self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z, self.table_pose.r.x, self.table_pose.r.y, self.table_pose.r.z, self.table_pose.r.w],
                scale=1.0
            )
            world_model = WorldConfig(
                mesh=obstables,
                cuboid=[table],
            )
            world_model = WorldConfig.create_collision_support_world(world_model)
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_file,
                world_model,
                tensor_args,
                # interpolation_dt=0.1,
                # trajopt_tsteps=8,
                collision_checker_type=CollisionCheckerType.MESH,
                use_cuda_graph=False,
                # num_trajopt_seeds=12,
                # num_graph_seeds=12,
                # interpolation_dt=0.03,
                collision_cache={"obb": 30, "mesh": 10},
                # collision_activation_distance=0.01,
                # acceleration_scale=1.0,
                self_collision_check=True,
                # maximum_trajectory_dt=0.25,
                # fixed_iters_trajopt=True,
                # finetune_dt_scale=1.05,
                # velocity_scale=None,
                # interpolation_type=InterpolateType.CUBIC,
                # use_gradient_descent=True,
                store_debug_in_result=False,
            )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True)
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        
    def prepare_groundedsam(self):
        
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        sam_version = "vit_h"
        sam_checkpoint = "../assets/ckpts/sam_vit_h_4b8939.pth"
        grounded_checkpoint = "../assets/ckpts/groundingdino_swint_ogc.pth"
        config = "../vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

        self.grounded_dino_model, self.sam_predictor = prepare_GroundedSAM_for_inference(sam_version=sam_version, sam_checkpoint=sam_checkpoint,
                grounded_checkpoint=grounded_checkpoint, config=config, device=self.device)

    def prepare_franka_asset(self):
        # load franka asset
        franka_asset_file = self.cfgs["asset"]["franka_asset_file"]
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, franka_asset_file, asset_options)

        # configure franka dofs
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = self.franka_dof_props["lower"]
        franka_upper_limits = self.franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
        
        # Set controller parameters
        # IK params
        self.damping = 0.05

        # use position drive for all dofs
        if self.controller_name == "ik" or self.controller_name == "curobo":
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            self.franka_dof_props["stiffness"][:7].fill(400.0)
            self.franka_dof_props["damping"][:7].fill(40.0)
        else:       # osc
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self.franka_dof_props["stiffness"][:7].fill(0.0)
            self.franka_dof_props["damping"][:7].fill(0.0)
        # grippers
        self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][7:].fill(800.0)
        self.franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        self.franka_default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        self.franka_default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        self.franka_default_dof_pos[7:] = franka_upper_limits[7:]

        self.franka_default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.franka_default_dof_state["pos"] = self.franka_default_dof_pos

        # send to torch
        self.default_dof_pos_tensor = to_torch(self.franka_default_dof_pos, device=self.device)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_num_links = len(franka_link_dict)
        # print("franka dof:", self.franka_num_dofs, "franka links:", self.franka_num_links)
        self.franka_hand_index = franka_link_dict["panda_hand"]

    def prepare_obj_assets(self):
        table_pose_p = self.cfgs["asset"]["table_pose_p"]
        table_scale = self.cfgs["asset"]["table_scale"]
        self.table_scale = self.cfgs["asset"]["table_scale"]
        table_dims = gymapi.Vec3(table_scale[0], table_scale[1], table_scale[2])
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(table_pose_p[0], table_pose_p[1], table_pose_p[2])

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        obj_asset_files = self.cfgs["asset"]["asset_files"]
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000
        self.num_asset_per_env = len(obj_asset_files)
        self.obj_assets = [self.gym.load_asset(self.sim, self.asset_root, obj_asset_file, asset_options) for obj_asset_file in obj_asset_files]
        self.obj_num_links_dict = [self.gym.get_asset_rigid_body_dict(asset_i) for asset_i in self.obj_assets]
        self.obj_num_links = sum([len(obj_num_links) for obj_num_links in self.obj_num_links_dict])
        self.obj_num_dofs = sum([self.gym.get_asset_dof_count(asset_i) for asset_i in self.obj_assets])
        self.table_num_links = 1
     
    def prepare_arti_obj_assets(self):
        ### TODO: support multiple loading
        self.gapartnet_ids = self.cfgs["asset"]["arti_gapartnet_ids"]
        arti_obj_paths = [f"gapartnet/{gapartnet_id}/mobility_relabel_gapartnet.urdf" for gapartnet_id in self.gapartnet_ids]

        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.use_mesh_materials = True
        arti_obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        arti_obj_asset_options.override_inertia = True
        arti_obj_asset_options.override_com = True
        arti_obj_asset_options.fix_base_link = True
        # obj_asset_options.vhacd_enabled = True
        # obj_asset_options.vhacd_params = gymapi.VhacdParams()
        # obj_asset_options.vhacd_params.resolution = 1000000
        self.arti_obj_assets = [self.gym.load_asset(self.sim, self.asset_root, arti_obj_path, arti_obj_asset_options)
                                for arti_obj_path in arti_obj_paths]

        ### TODO: support multiple loading from here
        self.arti_obj_asset = self.arti_obj_assets[0]
        self.arti_obj_num_dofs = self.gym.get_asset_dof_count(self.arti_obj_asset)
        arti_obj_link_dict = self.gym.get_asset_rigid_body_dict(self.arti_obj_asset)
        self.arti_obj_num_links = len(arti_obj_link_dict)
        print("obj dof:", self.arti_obj_num_dofs, "obj links:", self.arti_obj_num_links)
        
        self.arti_obj_default_dof_pos = np.zeros(self.arti_obj_num_dofs, dtype=np.float32)
        self.arti_obj_default_dof_state = np.zeros(self.arti_obj_num_dofs, gymapi.DofState.dtype)
        self.arti_obj_default_dof_state["pos"] = self.arti_obj_default_dof_pos
        self.arti_default_dof_pos_tensor = to_torch(self.arti_obj_default_dof_pos, device=self.device)

        # configure obj dofs
        self.arti_obj_dof_props = self.gym.get_asset_dof_properties(self.arti_obj_asset)
        self.arti_obj_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) ## TODO: check this
        self.arti_obj_dof_props["stiffness"].fill(4000)
        self.arti_obj_dof_props["damping"].fill(4000)
   
    def load_env(self, load_cam = True):
        self.envs = []
        self.obj_actor_idxs = []
        self.hand_idxs = []
        self.init_franka_pos_list = []
        self.init_franka_rot_list = []
        self.init_obj_pos_list = []
        self.init_obj_rot_list = []
        self.arti_init_obj_pos_list = []
        self.arti_init_obj_rot_list = []
        self.env_offsets = []
        self.arti_obj_actor_idxs = []

        # init pose
        franka_pose = gymapi.Transform()
        franka_pose_p = self.cfgs["asset"]["franka_pose_p"]
        franka_pose.p = gymapi.Vec3(franka_pose_p[0], franka_pose_p[1], franka_pose_p[2])
        # import pdb; pdb.set_trace()
        obj_pose_ps = [self.cfgs["asset"]["obj_pose_ps"][obj_i] for obj_i in range(self.num_asset_per_env)]
        if self.cfgs["asset"]["obj_pose_rs"] is not None:
            obj_pose_rs = [self.cfgs["asset"]["obj_pose_rs"][obj_i] for obj_i in range(self.num_asset_per_env)]
        else:
            obj_pose_rs = None
        position_noise = self.cfgs["asset"]["position_noise"]
        rotation_noise = self.cfgs["asset"]["rotation_noise"]
        
        arti_obj_pose_ps = self.cfgs["asset"]["arti_obj_pose_ps"]
        arti_obj_pose_p = arti_obj_pose_ps[0]
        arti_position_noise = self.cfgs["asset"]["arti_position_noise"]
        arti_rotation_noise = self.cfgs["asset"]["arti_rotation_noise"]
        arti_rotation = self.cfgs["asset"]["arti_rotation"]
        
        if load_cam:
            self.cams = []
            self.rgb_tensors = []
            self.depth_tensors = []
            self.seg_tensors = []
            self.cam_vinvs = []
            self.cam_projs = []
            

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)
            origin = self.gym.get_env_origin(env)
            self.env_offsets.append([origin.x, origin.y, origin.z])

            # add franka
            franka_handle = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 2, self.franka_seg_id)

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, self.franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, self.franka_default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, self.franka_default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.init_franka_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_franka_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)
            
            ### Table
            self.table_handle = self.gym.create_actor(env, self.table_asset, self.table_pose, "table", i, 0, self.table_seg_id)
            
            ## Object Assets
            self.init_obj_pos_list.append([])
            self.init_obj_rot_list.append([])
            self.obj_actor_idxs.append([])
            for asset_i in range(self.num_asset_per_env):
                initial_pose = gymapi.Transform()
                initial_pose.p.x = obj_pose_ps[asset_i][0] + np.random.uniform(-1.0, 1.0) * position_noise[0]
                initial_pose.p.y = obj_pose_ps[asset_i][1] + np.random.uniform(-1.0, 1.0) * position_noise[1]
                initial_pose.p.z = obj_pose_ps[asset_i][2]
                if obj_pose_rs is None:
                    initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))
                else:
                    initial_pose.r.x =  obj_pose_rs[asset_i][0]
                    initial_pose.r.y =  obj_pose_rs[asset_i][1]
                    initial_pose.r.z =  obj_pose_rs[asset_i][2]
                    initial_pose.r.w =  obj_pose_rs[asset_i][3]
                # initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))

                self.init_obj_pos_list[-1].append([initial_pose.p.x, initial_pose.p.y, initial_pose.p.z])
                self.init_obj_rot_list[-1].append([initial_pose.r.x, initial_pose.r.y, initial_pose.r.z, initial_pose.r.w])
                
                obj_actor_handle = self.gym.create_actor(env, self.obj_assets[asset_i], initial_pose, f'actor_{asset_i}', i, 0, self.asset_seg_ids[asset_i])
                
                obj_actor_idx = self.gym.get_actor_rigid_body_index(env, obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                self.obj_actor_idxs[i].append(obj_actor_idx)
                self.gym.set_actor_scale(env, obj_actor_handle, self.cfgs["asset"]["obj_scale"])
            
            if self.cfgs["USE_ARTI"]:
                ### Articulated Object
                arti_initial_pose = gymapi.Transform()
                arti_initial_pose.p.x = arti_obj_pose_p[0] + np.random.uniform(-1.0, 1.0) * arti_position_noise
                arti_initial_pose.p.y = arti_obj_pose_p[1] + np.random.uniform(-1.0, 1.0) * arti_position_noise
                arti_initial_pose.p.z = arti_obj_pose_p[2]
                arti_initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), arti_rotation/180.0*math.pi + arti_rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))
                self.arti_init_obj_pos_list.append([arti_initial_pose.p.x, arti_initial_pose.p.y, arti_initial_pose.p.z])
                self.arti_init_obj_rot_list.append([arti_initial_pose.r.x, arti_initial_pose.r.y, arti_initial_pose.r.z, arti_initial_pose.r.w])
                arti_obj_actor_handle = self.gym.create_actor(env, self.arti_obj_asset, arti_initial_pose, 'arti_actor', i, 1, self.asset_seg_ids[-1] + 1)
                self.gym.set_actor_dof_properties(env, arti_obj_actor_handle, self.arti_obj_dof_props)
                # set initial dof states
                ### TODO check
                # self.arti_obj_default_dof_state["pos"][:3] = 2 + np.random.uniform(-1.0, 1.0) * 0.5
                self.gym.set_actor_dof_states(env, arti_obj_actor_handle, self.arti_obj_default_dof_state, gymapi.STATE_ALL)
                # set initial position targets
                self.gym.set_actor_dof_position_targets(env, arti_obj_actor_handle, self.arti_obj_default_dof_state["pos"])
                arti_obj_actor_idx = self.gym.get_actor_rigid_body_index(env, arti_obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                self.arti_obj_actor_idxs.append(arti_obj_actor_idx)
                self.gym.set_actor_scale(env, arti_obj_actor_handle, self.cfgs["asset"]["arti_obj_scale"])
                
            
            if load_cam:
                # add camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cam_w
                cam_props.height = self.cam_h
                cam_props.far_plane = self.cam_far_plane
                cam_props.near_plane = self.cam_near_plane 
                cam_props.horizontal_fov = self.horizontal_fov
                cam_props.enable_tensors = True
                self.cams.append([])
                self.depth_tensors.append([])
                self.rgb_tensors.append([])
                self.seg_tensors.append([])
                self.cam_vinvs.append([])
                self.cam_projs.append([])
                for i in range(self.num_cam_per_env):
                    cam_handle = self.gym.create_camera_sensor(env, cam_props)
                    self.gym.set_camera_location(cam_handle, env, 
                        gymapi.Vec3(self.cam_poss[i][0], self.cam_poss[i][1], self.cam_poss[i][2]), 
                        gymapi.Vec3(self.cam_targets[i][0], self.cam_targets[i][1], self.cam_targets[i][2]))
                    self.cams[-1].append(cam_handle)
                
                    proj = self.gym.get_camera_proj_matrix(self.sim, env, cam_handle)
                    # view_matrix_inv = torch.inverse(torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, cam_handle))).to(self.device)
                    vinv = np.linalg.inv(np.matrix(self.gym.get_camera_view_matrix(self.sim, env, cam_handle)))
                    self.cam_vinvs[-1].append(vinv)
                    self.cam_projs[-1].append(proj)

                    # obtain rgb tensor
                    rgb_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                    # wrap camera tensor in a pytorch tensor
                    torch_rgb_tensor = gymtorch.wrap_tensor(rgb_tensor)
                    self.rgb_tensors[-1].append(torch_rgb_tensor)
                    
                    # obtain depth tensor
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_DEPTH)
                    # wrap camera tensor in a pytorch tensor
                    torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor)
                    self.depth_tensors[-1].append(torch_depth_tensor)
        

                    # obtain depth tensor
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    # wrap camera tensor in a pytorch tensor
                    torch_seg_tensor = gymtorch.wrap_tensor(seg_tensor)
                    self.seg_tensors[-1].append(torch_seg_tensor)
                    
                    
            
        self.env_offsets = np.array(self.env_offsets)
        
        # point camera at middle env
        if not self.headless:
            
            viewer_cam_pos = gymapi.Vec3(self.cam_poss[0][0], self.cam_poss[0][1], self.cam_poss[0][2])
            viewer_cam_target = gymapi.Vec3(self.cam_targets[0][0], self.cam_targets[0][1], self.cam_targets[0][2])
            middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, viewer_cam_pos, viewer_cam_target)

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

    def control_ik(self, dpose):
        # global damping, j_eef, num_envs
        damping = 0.05
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
    
    def plan_to_pose_ik(self, goal_position, goal_roation, close_gripper = True, save_video = False, save_root = "", start_step = 0, control_steps = 10):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        hand_rot_now = self.hand_rot
        goal_roation = goal_roation.to(self.device).reshape(1,-1)
        goal_position = goal_position.to(self.device).reshape(1,-1)
        orn_err = orientation_error(goal_roation, hand_rot_now)
        pos_err = goal_position - self.hand_pos
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        pos_action[:, :7] = self.dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
        if close_gripper:
            grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
        else:
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device)
        # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
        #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        pos_action[:, 7:9] = grip_acts
        for step_i in range(control_steps):
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))            
            self.run_steps(pre_steps = 1)
            if save_video:
                self.gym.render_all_camera_sensors(self.sim)
                step_str = str(start_step + step_i).zfill(4)
                os.makedirs(f"{save_root}/video", exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")

    def control_ik_numpy(self, dpose):
        # global damping, j_eef, num_envs
        damping = 0.05
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        j_eef_T = j_eef_T.cpu().numpy()
        j_eef = self.j_eef.cpu().numpy()
        lmbda = lmbda.cpu().numpy()
        # dpose = dpose.cpu().numpy()
        u = ((j_eef_T @ np.linalg.inv(j_eef @ j_eef_T + lmbda.reshape(j_eef.shape[0], 6, 6))) @ dpose.reshape(j_eef.shape[0], 6, 1)).reshape(self.num_envs, 7)
        u = torch.from_numpy(u).to(self.device)
        
        # u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def control_osc(self, dpose):
        kp = 150.
        kd = 2.0 * np.sqrt(kp)
        kp_null = 10.
        kd_null = 2.0 * np.sqrt(kp_null)
        # global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
        mm_inv = torch.inverse(self.mm)
        m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (
            kp * dpose - kd * self.hand_vel.unsqueeze(-1))

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self.j_eef @ mm_inv
        u_null = kd_null * -self.dof_vel + kp_null * (
            (self.default_dof_pos_tensor.view(1, -1, 1) - self.dof_pos + np.pi) % (2 * np.pi) - np.pi)
        u_null = u_null[:, :7]
        u_null = self.mm @ u_null
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null
        return u.squeeze(-1)

    def init_observation(self):
        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = self.jacobian[:, self.franka_hand_index - 1, :, :7]

        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = self.mm[:, :7, :7]          # only need elements corresponding to the franka arm
        
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        num_rb = int(self.rb_states.shape[0]/self.num_envs)
        
        self.root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.gym.refresh_actor_root_state_tensor(self.sim)
            
        
        if self.cfgs["USE_ARTI"]:
            assert num_rb == self.franka_num_links + self.obj_num_links + self.table_num_links + self.arti_obj_num_links, "Number of rigid bodies in tensor does not match franka & obj asset"
        else:
            assert num_rb == self.franka_num_links + self.obj_num_links + self.table_num_links, "Number of rigid bodies in tensor does not match franka & obj asset"
        
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        num_dof = int(self.dof_states.shape[0]/self.num_envs)
        if self.cfgs["USE_ARTI"]:
            assert num_dof == self.franka_num_dofs + self.obj_num_dofs + self.arti_obj_num_dofs, "Number of dofs in tensor does not match franka & obj asset"
        else:
            assert num_dof == self.franka_num_dofs + self.obj_num_dofs, "Number of dofs in tensor does not match franka & obj asset"

        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, num_dof, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, num_dof, 1)

    def refresh_observation(self, get_visual_obs = True):
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        # state obs
        self.hand_pos = self.rb_states[self.hand_idxs, :3]
        self.hand_rot = self.rb_states[self.hand_idxs, 3:7]
        self.hand_vel = self.rb_states[self.hand_idxs, 7:]

        ### TODO: support different dof tensor shapes in different envs
        self.robot_dof_qpos_qvel = self.dof_states.reshape(self.num_envs,-1,2)[:,:self.franka_num_dofs, :].view(self.num_envs, self.franka_num_dofs, 2)
        
        # render sensors and refresh camera tensors
        if self.use_cam and get_visual_obs:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            points_envs = []
            colors_envs = []
            ori_points_envs = []
            ori_colors_envs = []
            rgb_envs = []
            depth_envs = []
            seg_envs = []
            # bbox_axis_aligned_envs = []
            # bbox_center_envs = []
            # import pdb; pdb.set_trace()
            for env_i in range(self.num_envs):
                points_env = []
                colors_env = []
                rgb_env = []
                depth_env = []
                seg_env = []
                for cam_i_per_env in range(self.num_cam_per_env):
                    # write tensor to image
                    cam_img = self.rgb_tensors[env_i][cam_i_per_env].cpu().numpy()
                    depth = self.depth_tensors[env_i][cam_i_per_env].cpu().numpy() # W * H
                    seg = self.seg_tensors[env_i][cam_i_per_env].cpu().numpy() # W * H

                    rgb_env.append(cam_img)
                    depth_env.append(depth)
                    seg_env.append(seg)
                    
                    # if self.cfgs["INFERENCE_GSAM"]:
                    #     masks = inference_one_image(cam_img[..., :3], self.grounded_dino_model, self.sam_predictor, box_threshold=self.box_threshold, 
                    #         text_threshold=self.text_threshold, text_prompt=text_prompt, device=self.device)
                    
                    #     if self.cfgs["SAVE_RENDER"]:
                    #         save_dir = self.cfgs["SAVE_ROOT"]
                    #         os.makedirs(save_dir, exist_ok=True)
                    #         import cv2
                    #         for i in range(masks.shape[0]):
                    #             cam_img_ = cam_img.copy()
                    #             cam_img_[masks[i][0].cpu().numpy()] = 0
                    #             fname = os.path.join(save_dir, text_prompt + "-mask-%04d-%04d-%04d-%04d.png" % (0, env_i, cam_i_per_env, i))
                    #             imageio.imwrite(fname, cam_img_)

                        
                    ### RGBD -> Point Cloud with CPU
                    # s = time.time()
                    # points, colors = get_point_cloud_from_rgbd(depth, cam_img, None, self.cam_vinvs[env_i][cam_i_per_env], self.cam_projs[env_i][cam_i_per_env], self.cam_w, self.cam_h)
                    # points = np.transpose(points(0, 2, 1))
                    # e = time.time()
                    # print("Time to get point cloud: ", e-s)
                    
                    ### RGBD -> Point Cloud with GPU
                    s = time.time()
                    pointclouds = get_point_cloud_from_rgbd_GPU(
                        self.depth_tensors[env_i][cam_i_per_env], 
                        self.rgb_tensors[env_i][cam_i_per_env],
                        None,
                        self.cam_vinvs[env_i][cam_i_per_env], 
                        self.cam_projs[env_i][cam_i_per_env], 
                        self.cam_w, self.cam_h
                    )
                    points = pointclouds[:, :3].cpu().numpy()
                    colors = pointclouds[:, 3:6].cpu().numpy()
                    i_indices, j_indices = np.meshgrid(np.arange(self.cam_w), np.arange(self.cam_h), 
                            indexing='ij')
                    pointid2pixel = np.stack((i_indices, j_indices), axis=-1).reshape(-1, 2)
                    pixel2pointid = np.arange(self.cam_w * self.cam_h).reshape(self.cam_w, self.cam_h)
                    pointid2pixel = None
                    pixel2pointid = None
                    points_env.append(points)
                    colors_env.append(colors)
                    
                    # e = time.time()
                    # print("Time to get point cloud: ", e-s)
                    
                    # if self.cfgs["INFERENCE_GSAM"]:
                    #     pc_mask = masks[0][0].cpu().numpy().reshape(-1)
                    #     target_points = points[pc_mask]
                    #     target_colors = colors[pc_mask]
                    #     point_cloud = o3d.geometry.PointCloud()
                    #     point_cloud.points = o3d.utility.Vector3dVector(target_points[:, :3])
                    #     point_cloud.colors = o3d.utility.Vector3dVector(target_colors[:, :3]/255.0)
                        
                    #     if self.cfgs["SAVE_RENDER"]:
                    #         # save_to ply
                    #         fname = os.path.join(save_dir, "point_cloud-%04d-%04d-target.ply" % (env_i, cam_i_per_env))
                    #         o3d.io.write_point_cloud(fname, point_cloud)
                    #     bbox_axis_aligned = np.array([target_points.min(axis=0), target_points.max(axis=0)])
                    #     bbox_center = bbox_axis_aligned.mean(axis=0)
                    #     bbox_axis_aligned_envs.append(bbox_axis_aligned)
                    #     bbox_center_envs.append(bbox_center)
                    #     masks_envs.append(masks)
                    

                ori_points_envs.append(points_env)
                ori_colors_envs.append(colors_env)
                rgb_envs.append(rgb_env)
                depth_envs.append(depth_env)
                seg_envs.append(seg_env)
                # points_env = np.concatenate(points_env, axis=0) - self.env_offsets[env_i]
                # colors_env = np.concatenate(colors_env, axis=0) - self.env_offsets[env_i]
                # pc_mask_bound = (points_env[:, 0] > self.point_cloud_bound[0][0]) & (points_env[:, 0] < self.point_cloud_bound[0][1]) & \
                #                 (points_env[:, 1] > self.point_cloud_bound[1][0]) & (points_env[:, 1] < self.point_cloud_bound[1][1]) & \
                #                 (points_env[:, 2] > self.point_cloud_bound[2][0]) & (points_env[:, 2] < self.point_cloud_bound[2][1])
                # points_env = points_env[pc_mask_bound]
                # colors_env = colors_env[pc_mask_bound]

                # s = time.time()
                # points_env, colors_env, pcs_mask = get_downsampled_pc(points_env, colors_env, 
                #     sampled_num=self.cfgs["cam"]["sampling_num"], sampling_method = self.cfgs["cam"]["sampling_method"])
                # e = time.time()
                # points_envs.append(points_env)
                # colors_envs.append(colors_env)
                # print("Time to get point cloud: ", e-s)

            self.gym.end_access_image_tensors(self.sim)

        
            return points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel

    def save_render(self, rgb_envs, depth_envs, ori_points_env, ori_colors_env, points, colors, save_dir, save_name = "render", save_pc = False, save_depth = False, save_single = False):
        for env_i in range(len(rgb_envs)):
            for cam_i in range(len(rgb_envs[0])):
                fname = os.path.join(save_dir, f"{save_name}-rgb-{env_i}-{cam_i}.png")
                os.makedirs(save_dir, exist_ok=True)
                imageio.imwrite(fname, rgb_envs[env_i][cam_i].astype(np.uint8))
                if save_single:
                    return
        
                if depth_envs is not None and save_depth:
                    depth = depth_envs[env_i][cam_i]
                    # depth clip to 0.1m - 10m and scale to 0-255
                    depth_clip = np.clip(depth, -1, 1)
                    depth_rgb = (depth_clip + 1) / 2 * 255.0
                    # W * H * 3
                    depth_img = np.zeros((depth_rgb.shape[0], depth_rgb.shape[1], 3))
                    depth_img[:, :, 0] = depth_rgb
                    depth_img[:, :, 1] = depth_rgb
                    depth_img[:, :, 2] = depth_rgb
                    fname = os.path.join(save_dir, f"{save_name}-depth-{env_i}-{cam_i}.png")
                    os.makedirs(save_dir, exist_ok=True)
                    imageio.imwrite(fname, depth_img.astype(np.uint8))
            
                if ori_points_env is not None and save_pc:
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(ori_points_env[env_i][cam_i][:, :3])
                    point_cloud.colors = o3d.utility.Vector3dVector(ori_colors_env[env_i][cam_i][:, :3]/255.0)
                    # save_to ply
                    fname = os.path.join(save_dir, f"{save_name}-partial-point_cloud--{env_i}-{cam_i}.ply")
                    o3d.io.write_point_cloud(fname, point_cloud)
            # o3d.visualization.draw_geometries([point_cloud])
            if points is not None and save_pc:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points[env_i][:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(colors[env_i][:, :3]/255.0)
                # save_to ply
                fname = os.path.join(save_dir, f"{save_name}-{env_i}-all-point_cloud.ply")
                o3d.io.write_point_cloud(fname, point_cloud)

    def inference_gsam(self, rgb_img, points, colors, text_prompt, save_dir, save_name = "gsam"):
        
        bbox_axis_aligned_envs = []
        bbox_center_envs = []
        
        assert self.cfgs["INFERENCE_GSAM"]
        masks = inference_one_image(rgb_img[..., :3], self.grounded_dino_model, self.sam_predictor, box_threshold=self.box_threshold, text_threshold=self.text_threshold, text_prompt=text_prompt, device=self.device)

        if masks is None:
            # import pdb; pdb.set_trace()
            return None, None, None
        if self.cfgs["SAVE_RENDER"]:
            os.makedirs(save_dir, exist_ok=True)
            for i in range(masks.shape[0]):
                cam_img_ = rgb_img.copy()
                cam_img_[masks[i][0].cpu().numpy()] = 0
                fname = os.path.join(save_dir, f"{save_name}-gsam-mask-{text_prompt}-{i}.png")
                imageio.imwrite(fname, cam_img_)
                np.save(fname.replace(".png", ".npy"), masks[i][0].cpu().numpy())
                
        
        for i in range(masks.shape[0]):
            pc_mask = masks[i][0].cpu().numpy().reshape(-1)
            target_points = points[pc_mask]
            target_colors = colors[pc_mask]
            
            if self.cfgs["SAVE_RENDER"]:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(target_points[:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(target_colors[:, :3]/255.0)
                # save_to ply
                fname = os.path.join(save_dir, f"{save_name}-gsam-mask-{text_prompt}-{i}.ply")
                o3d.io.write_point_cloud(fname, point_cloud)
                
            bbox_axis_aligned = np.array([target_points.min(axis=0), target_points.max(axis=0)])
            bbox_center = bbox_axis_aligned.mean(axis=0)
            bbox_axis_aligned_envs.append(bbox_axis_aligned)
            bbox_center_envs.append(bbox_center)

        return masks, bbox_axis_aligned_envs, bbox_center_envs

    def plan_to_pose_curobo(self, position, quaternion, max_attempts=100, start_state= None):
        '''
        start_state: JointState
            if None, use current state as start state
            else, use given start_state
            
        position: list or np.array
            target position
        quaternion: list or np.array
            target orientation
        '''
        if start_state == None:
            start_state = JointState.from_position(self.robot_dof_qpos_qvel[:,:7,0])
        goal_state = Pose(torch.tensor(torch.tensor(position)-torch.tensor(self.cfgs["asset"]["franka_pose_p"]), device = self.device, dtype = torch.float64), 
                          quaternion=torch.tensor(quaternion, device = self.device, dtype = torch.float64))
        result = self.motion_gen.plan_single(start_state, goal_state, MotionGenPlanConfig(max_attempts=max_attempts))

        traj = result.get_interpolated_plan()
        # if result.optimized_dt == None or result.success[0] == False:
        #     return None
        try:
            print("Trajectory Generated: ", result.success, result.optimized_dt.item(), traj.position.shape)
        except:
            print("Trajectory Generated: ", result.success)
        return traj

    def move_to_traj(self, traj, close_gripper = True, save_video = False, save_root = "", start_step = 0):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        #import pdb; pdb.set_trace()
        for step_i in range(len(traj)):
            # print("Step: ", step_i)
            # Deploy actions
            pos_action[:, :7] = traj.position.reshape(-1, 7)[step_i]
            if close_gripper:
                grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
            else:
                grip_acts = torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device)
            # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
            #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
            pos_action[:, 7:9] = grip_acts
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))
            self.run_steps(pre_steps = 1)
            if save_video:
                self.gym.render_all_camera_sensors(self.sim)
                # print("Saving video frame:", start_step + step_i)
                step_str = str(start_step + step_i).zfill(4)
                os.makedirs(f"{save_root}/video", exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")
                # self.gym.write_viewer_image_to_file(self.viewer, f"{save_root}/step-{start_step + step_i}.png")
          
    def move_gripper(self, close_gripper = True, save_video = False, save_root = "", start_step = 0):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        if close_gripper:
            grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
        else:
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device)
        # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
        #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        pos_action[:, :7] = self.robot_dof_qpos_qvel[:,:7,0]
        pos_action[:, 7:9] = grip_acts
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))
        self.run_steps(pre_steps = 5)
        if save_video:
            self.gym.render_all_camera_sensors(self.sim)
            # print("Saving video frame:", start_step)
            # start_step string, 4 digit
            step_str = str(start_step).zfill(4)
            os.makedirs(f"{save_root}/video", exist_ok=True)
            self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")
        return start_step + 1
        
    def control_to_pose(self, pose, close_gripper = True, save_video = False, save_root = "", step_num = 0, controller = "curobo", start_qpos = None):
        # move to pre-grasp
        self.refresh_observation(get_visual_obs=False)
        USE_IK_CONTROL = False
        if USE_IK_CONTROL:
            self.plan_to_pose_ik(
                torch.tensor(pose[:3], dtype = torch.float32), 
                torch.tensor(pose[3:], dtype = torch.float32),
                close_gripper=close_gripper,
                save_video=save_video,
                save_root = save_root,
                start_step = step_num,
                control_steps = 10
                )
            step_num += 10
        else:
            traj = self.plan_to_pose_curobo(torch.tensor(pose[:3], dtype = torch.float32), torch.tensor(pose[3:], dtype = torch.float32), start_state=start_qpos)
            if traj == None:
                # os.system(f"rm -r {save_root}/video")
                print("traj planning error")
                return step_num, traj
            self.move_to_traj(traj, close_gripper=close_gripper, save_video=save_video, save_root = save_root, start_step = step_num)
            step_num += len(traj)
        return step_num, traj
        
    def run_gpt4v_gsam_graspnet_inference(self, instruction, save_root, use_3d_info = True, use_gsnet = False):
        
        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)
        ### save rendered data
        # self.save_render(rgb_envs=rgb_envs[0], depth_envs=depth_envs[0], ori_points_env=ori_points_envs[0], 
        #                 ori_colors_env=ori_colors_envs[0], points=points_envs[0], colors=colors_envs[0], 
        #                 save_dir = save_root, save_name = "before_plc")
        self.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, 
                        save_dir = save_root, save_name = "before_plc")


        ### save to a tmp path for gpt4v inference
        tmp_img_path = save_root + "/rgb_tmp.png"
        imageio.imwrite(tmp_img_path, rgb_envs[0][0])
        
        if not use_3d_info:
            ### prompt for gpt4v inference: find the grasping pixel
            prompt = f"Now, I'd like to guide a robot to perform a specific task: {instruction}. \
At this stage you don't need to think about rotation, just 3d position. All the objects are placed on the table. \
The input image have the resolution of [1440, 2160], which means the height is 1440 pixels and the width is 2160 pixels. \
Kindly output the object name you want to grasp and then place and also the pixel in this image you want to place, separating them with ',', without \
including any additional responses or text. For example, if the instruction is 'Place the cup \
between the book and the pear,', if the book is at [200, 500] and the pear is at [1200, 400], \
you should output 'cup,700,450', which is the middle of the two objects. Please note that you should \
only provide the pixel position, separated by ',', and refrain from including any other information or responses."

            ### gpt4v inference
            try:
                
                response = infer_path(prompt, tmp_img_path)
                while 'choices' not in response.json():
                    response = infer_path(prompt, tmp_img_path)
                print(response.json()['choices'][0]['message']['content'])
                
                grasp_obj_name = response.json()['choices'][0]['message']['content'].split(",")[0]
                pixels = response.json()['choices'][0]['message']['content'].split(",")[1:]
                pixels = [int(pixel) for pixel in pixels]
                pixel = np.array(pixels)
            except:
                response = infer_path(prompt, tmp_img_path)
                while 'choices' not in response.json():
                    response = infer_path(prompt, tmp_img_path)
                print(response.json()['choices'][0]['message']['content'])
                
                grasp_obj_name = response.json()['choices'][0]['message']['content'].split(",")[0]
                pixels = response.json()['choices'][0]['message']['content'].split(",")[1:]
                pixels = [int(pixel) for pixel in pixels]
                pixel = np.array(pixels)

            # bound the pixel
            pixel[0] = np.clip(pixel[0], 10, 1430)
            pixel[1] = np.clip(pixel[1], 10, 2150)
            
            img_tmp = rgb_envs[0][0][:, :,:3].copy()
            img_tmp[pixel[0]-10:pixel[0]+10, pixel[1]-10:pixel[1]+10] = [255, 0, 0]
            
            tmp_img_path2 = save_root + "/rgb_tmp_select.png"
            imageio.imwrite(tmp_img_path2, img_tmp)
            
            place_position =  ori_points_envs[0][0][pixel[0] * 2160 + pixel[1]]
        else:
            ### prompt for gpt4v inference: find the object to query
    #         prompt = f"Now I want to control a robot to {instruction}. You have a detection model that can tell \
    # you the accurate position of any given object name. You can invoke this api now. Please give \
    # me the object you want to invoke the api and then you will know the position of the object you asked. After that I will give you \
    # the accurate position for you to make further decision. Please list their names splitted by \
    # space without any other response and text. For example, if the instruction is 'Place the \
    # cup between the book and the pear', you can output 'cup book pear'. Notice that you should \
    # only output object names splitted by space without any other response and text."

            prompt = f"Now, I'd like to guide a robot to perform a specific task: {instruction}. \
At this stage you don't need to think about rotation, just 3d position. All the objects are placed on the table. \
You have access to a precise object detection model capable of determining the exact position \
of objects by name. Please specify the object you'd like to identify using this API. Once you \
provide the object name, we will retrieve its position, enabling you to make informed decisions. \
Kindly list the names of the objects you want to identify, separating them with ',', without \
including any additional responses or text. For example, if the instruction is 'Place the cup \
between the book and the pear,' you should output 'cup,book,pear' Please note that you should \
only provide the object names, separated by ',', and refrain from including any other information or responses."

            ### gpt4v inference
            response = infer_path(prompt, tmp_img_path)
            while 'choices' not in response.json():
                response = infer_path(prompt, tmp_img_path)
            print(response.json()['choices'][0]['message']['content'])

            ### get the object bbox information with gsam
            objs = response.json()['choices'][0]['message']['content'].split(",")
            position_info = "The position information is as follow. The axis-aligned bounding box information \
is given by two diagonal corner points"
            
            print(objs)
            for obj in objs:
                masks, bbox_axis_aligned_envs, bbox_center_envs = self.inference_gsam(
                    rgb_envs[0][0], ori_points_envs[0][0], ori_colors_envs[0][0], 
                    text_prompt = obj, save_dir = save_root, save_name = "gsam")
                
                
                if bbox_center_envs is None:
                    position_info += f"The position information of {obj} is unknown."
                else:
                    bbox = list(bbox_axis_aligned_envs[0])
                    bbox_center = bbox_center_envs[0]
                    position_info += f"The position information of {obj} is: the axis-aligned bounding box of {obj} is {bbox},\
the bounding box of {obj} center is {bbox_center}."

                objs_str = ",".join(objs)
            ### prompt for gpt4v inference: find the position to place the object with the given information
    #         prompt = f"Now I want to control a robot to {instruction}. You have a detection model that can tell you the \
    # accurate position of any given object name. You can invoke this api now. In the last query, \
    # you choose to check the position of {objs_str}: {position_info} Please output the position you want \
    # to target to be to following the given instruction. You can take y axis as left and right, \
    # x axis as front and back. +y means right, -y means left, +x means back and -x means front. \
    # Please output x y z position splitted by space without any other response and text. Important: \
    # Your answer will used in downstream api, so please only output 3 float numbers without any other response and text."

            prompt = f"Now, I want to guide a robot in performing a specific task: {instruction}. \
    At this stage you don't need to think about rotation, just 3d position. \
    You have access to a highly accurate object detection model that can precisely \
    locate objects by their names. In the previous query, you selected {objs_str} \
    for position information, and you received the following data: {position_info}. \
    Please provide the target position you'd like the robot to reach, following the \
    provided instruction. In this context, consider the Y-axis to represent left and \
    right movements, and the X-axis to represent forward and backward movements. \
    To clarify, +Y denotes moving to the right, -Y signifies moving to the left, \
    +X corresponds to moving backward, and -X indicates moving forward. For example, if you want to place the \
    target object in front of an apple at [0,0,0], the answer can be [-0.1,0,0] with smaller x axis, which means moving forward. \
    If you want to place the \
    target object behind an object at [1,0,0.2], the answer can be [1.1,0,0.2] with larger x axis, which means moving backward. \
    In simpler terms, if the task is to place an object behind another object, increase the x-coordinate. \
    To place it in front, decrease the x-coordinate. To place it on the left, decrease the y-coordinate, \
    and to place it on the right, increase the y-coordinate. \
    Output the desired object name to grasp and the position to place the \
    object as one string and three float numbers 'name_string,x,y,z' separated by ',', \
    without any additional responses or text in the last line. 'name_string' \
    indicate the name of the target object to grasp. \
    'x,y,z' indicated the exact 3d position to place the target object. It's essential to note that your \
    response will be utilized in downstream API processes, so think it step by step but ensure in your response, the last line only contains \
    one string and three float values."

            ### gpt4v inference
            response = infer_path(prompt, tmp_img_path)
            print(response.json()['choices'][0]['message']['content'])
            try:
                infer_data = response.json()['choices'][0]['message']['content'].split(",")
                grasp_obj_name = infer_data[0]
                place_position = [float(p) for p in infer_data[1:]]
            except:
                response = infer_path(prompt, tmp_img_path)
                print(response.json()['choices'][0]['message']['content'])
                infer_data = response.json()['choices'][0]['message']['content'].split(",")
                grasp_obj_name = infer_data[0]
                place_position = [float(p) for p in infer_data[1:]]
            
        masks, bbox_axis_aligned_envs, bbox_center_envs = self.inference_gsam(rgb_envs[0][0], ori_points_envs[0][0], ori_colors_envs[0][0], text_prompt = grasp_obj_name, save_dir = save_root, save_name = "gsam")
        
        if self.cfgs["USE_SUDOAI"]:
            image = rgb_envs[0][0]
            mask = masks[0][0].cpu().numpy()
            image_masked = image.copy()
            image_masked[~mask] = 255
            mask_pixels = np.where(mask)
            mask_bbox = np.array([[mask_pixels[0].min() - 50, mask_pixels[1].min()- 50], [mask_pixels[0].max()+50, mask_pixels[1].max()+50]])
            mask_bbox[:,0] = np.clip(mask_bbox[:,0], 0, 1439)
            mask_bbox[:,1] = np.clip(mask_bbox[:,1], 0, 2159)
            cropped_image = image_masked[mask_bbox[0,0]:mask_bbox[1,0], mask_bbox[0,1]:mask_bbox[1,1]]
            
            # save
            fname = os.path.join(save_root, "gsam-mask.png")
            imageio.imwrite(fname, image_masked)
            fname = os.path.join(save_root, "gsam-mask-crop.png")
            imageio.imwrite(fname, cropped_image)
            
            self.inference_sudo_ai(fname)
        
        if use_gsnet:
            pcs_input = points_envs[0].copy()
            
            pcs_input[...,2] = -pcs_input[...,2]
            gg = self.inference_graspnet(pcs_input)
            
            thre = 0.02
            filtered_gg = [g_i for g_i in range(len(gg)) 
                if gg[g_i].translation[0] > bbox_axis_aligned_envs[0][0][0] - thre and gg[g_i].translation[0] < bbox_axis_aligned_envs[0][1][0] + thre
                and gg[g_i].translation[1] > bbox_axis_aligned_envs[0][0][1] - thre and gg[g_i].translation[1] < bbox_axis_aligned_envs[0][1][1] + thre
                and -gg[g_i].translation[2] > bbox_axis_aligned_envs[0][0][2] - thre and -gg[g_i].translation[2] < bbox_axis_aligned_envs[0][1][2] + thre]
            filtered_gg = gg[filtered_gg]
            if len(filtered_gg) == 0:
                print("**************No Good Grasp Found**************")
                filtered_gg = gg
        else:
            filtered_gg = None
            
        if False:
            grippers = filtered_gg[:1].to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcs_input)
            # o3d.visualization.draw_geometries([cloud, *grippers])
        
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(cloud)
            vis.add_geometry(*grippers)
            
            vis.poll_events()
            vis.update_renderer()
            # o3d.io.write_image(f"{save_root}/grasp_pc.png", vis.capture_screen_image())
            vis.capture_screen_image(f"{save_root}/grasp_pc.png")
            vis.destroy_window()
        
        # print summary of gpt inference
        print("#"*20, "Summary of GPT Inference", "#"*20)
        print("Instruction: ", instruction)
        print("Grasp Object: ", grasp_obj_name)
        print("Place Position: ", place_position)
        print("#"*50)
        
        
        return filtered_gg, place_position
    
    def move_obj_to_pose(self, position, quaternion = None):
        
        root_states = self.root_states.clone()
        root_states[-1, :3] = torch.tensor(position, dtype=torch.float32, device=self.device)
        if quaternion is not None:
            root_states[-1, 3:7] = torch.tensor(quaternion, dtype=torch.float32, device=self.device)
        # self.rb_states[:, self.actor_id, :7] = target_pose
        root_reset_actors_indices = torch.unique(torch.tensor(np.arange(root_states.shape[0]), dtype=torch.float32, device=self.device)).to(dtype=torch.int32)
        res = self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(root_reset_actors_indices),len(root_reset_actors_indices))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        assert res == True
        self.run_steps(1)
        if False:
            points_envs, colors_envs, rgb_envs, depth_envs, seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation()
            cv2.imwrite(f"test.png", rgb_envs[0][0])
        
    def interaction(self, instruction = "", grasp_obj_urdf = None, final_rotation = None, save_root = "test", save_video = False, use_3d = True):

        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        
        if use_3d:
            filtered_gg, place_position = self.run_gpt4v_gsam_graspnet_inference(instruction, save_root, use_gsnet=self.cfgs["USE_CUROBO"])
        else:
            filtered_gg, place_position = self.run_gpt4v_gsam_graspnet_inference(instruction, save_root, use_3d_info = False, use_gsnet=self.cfgs["USE_CUROBO"])    
        
        
        if self.cfgs["USE_CUROBO"]:
            grasp_position = filtered_gg[0].translation
            grasp_position[2] = -grasp_position[2]

            print(grasp_position, place_position)
            delta_m = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
            # import pdb; pdb.set_trace()
            rotation_output = filtered_gg[0].rotation_matrix.copy()
            print(rotation_output)
            rotation_output[2, :] = -rotation_output[2, :]
            rotation_output[:, 2] = -rotation_output[:, 2]
            print(rotation_output)
            

            # grippers = filtered_gg[:1].to_open3d_geometry_list()
            # cloud = o3d.geometry.PointCloud()
            # cloud.points = o3d.utility.Vector3dVector(points_envs[0])
            # o3d.visualization.draw_geometries([cloud, *grippers])   
            
            rotation_output = np.dot(rotation_output, delta_m)
            print(rotation_output)
            grasp_quat_R = R.from_matrix(rotation_output).as_quat()
            print(grasp_quat_R)
            rotation_input = grasp_quat_R
            rotation_input = np.array([rotation_input[3],rotation_input[0],rotation_input[1],rotation_input[2]])
            grasp_pose = np.concatenate([grasp_position, rotation_input])
            
            place_pose = np.concatenate([place_position, rotation_input])
            
            # R: xyzw
            # IsaacGym: xyzw
            
            # traj = self.plan_to_pose_curobo(torch.tensor(grasp_pose[:3], dtype = torch.float32), torch.tensor(rotation_input, dtype = torch.float32))
            # self.move_to_traj(traj, close_gripper=False, save_video=save_video, save_root = save_root, start_step = step_num)
            
            # self.refresh_observation(get_visual_obs=False)
            
            # R.from_quat(grasp_pose[3:]).as_matrix()
            # R.from_quat(rotation_input).as_matrix()
            # R.from_quat(self.hand_rot.cpu().numpy()).as_matrix()
            
            
            rotation_unit_vect = rotation_output[:,2]
            
            grasp_pre_grasp = grasp_pose.copy()
            grasp_pre_grasp[:3] -= rotation_unit_vect*0.2
            
            grasp_grasp = grasp_pose.copy()
            grasp_grasp[:3] -= rotation_unit_vect*0.05
            
            grasp_lift = grasp_pose.copy()
            grasp_lift[2] += 0.3
            # grasp_lift[:3] -= rotation_unit_vect*0.2
            
            place_pose[:3] -= rotation_unit_vect*0.05
            place_position_lift = place_pose.copy()
            place_position_lift[2] += 0.3
            place_position_place = place_pose.copy()
            place_position_place[2] += 0.05
            place_position_up = place_pose.copy()
            place_position_up[2] += 0.3
            
            finger_front = np.array([0, 0, -1])
            finger_side = np.array([0, 1, 0])
            finger_front_norm = finger_front / np.linalg.norm(finger_front)
            finger_side_norm = finger_side / np.linalg.norm(finger_side)
            finger_face_norm = np.cross(finger_side_norm, finger_front_norm)

            quaternion = R.from_matrix(np.concatenate([finger_face_norm.reshape(-1,1), finger_side_norm.reshape(-1,1), finger_front_norm.reshape(-1,1)], axis = 1)).as_quat()

            # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)
            # prompt = grasp_obj_name
            # masks, bbox_axis_aligned_envs, grasp_envs = self.inference_gsam(rgb_envs[0][0], ori_points_envs[0][0], ori_colors_envs[0][0], text_prompt=prompt, save_dir=self.cfgs["SAVE_ROOT"])

            # grasp_envs[0] += 0.00
            
            step_num = 0
            #import pdb; pdb.set_trace()
            # move to pre-grasp
            print("grasp_pre_grasp: ", grasp_pre_grasp)
            
            
            self.prepare_curobo(use_mesh=self.cfgs["USE_MESH_COLLISION"])
            step_num, traj = self.control_to_pose(grasp_pre_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
            import pdb; pdb.set_trace()
            points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)
            
            trajs = []
            fig_data = []
            for _ in range(5):
                # add noise
                _=0
                target = filtered_gg[_].translation
                target[2] = -target[2]
                grasp_grasp[:3] = target
                step_num, traj = self.control_to_pose(grasp_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
                trajs.append(traj)
                config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
                urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]  # Send global path starting with "/"
                base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
                ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
                tensor_args = TensorDeviceType()
                robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
                kin_model = CudaRobotModel(robot_cfg.kinematics)
                qpos = torch.tensor(traj.position, **vars(tensor_args))
                out = kin_model.get_state(qpos)
                traj_p = out.ee_position.cpu().numpy()
                fig_data.append(go.Scatter3d(x=traj_p[:,0], y=traj_p[:,1], z=traj_p[:,2], mode='markers', name='waypoints', marker=dict(size=10, color='red')))
                for i in range(0, traj_p[:,0].shape[0] - 1): fig_data.append(go.Scatter3d(x=traj_p[:,0][i:i+2], y=traj_p[:,1][i:i+2], z=traj_p[:,2][i:i+2], mode='lines', name='path', line=dict(width=10, color='yellow')))
            
            fig_data.append(go.Scatter3d(x=points_envs[0][:,0], y=points_envs[0][:,1], z=points_envs[0][:,2], mode='markers', name='waypoints', marker=dict(size=4, color=colors_envs[0])))
            # add lines between waypoints
            fig = go.Figure(data = fig_data)
            fig.show()
            fig.write_html("test.html")
            
            
            # move to grasp
            print("grasp_grasp: ", grasp_grasp)
            step_num, traj = self.control_to_pose(grasp_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
            step_num = self.move_gripper(close_gripper = True, save_video=save_video, save_root = save_root, start_step = step_num)
            
            # move to lift
            print("grasp_lift: ", grasp_lift)
            step_num, traj = self.control_to_pose(grasp_lift, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)
            
            # move to pre-place
            print("place_position_lift: ", place_position_lift)
            step_num, traj = self.control_to_pose(place_position_lift, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)            

            # move to place
            print("place_position_place: ", place_position_place)
            step_num, traj = self.control_to_pose(place_position_place, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)
            step_num = self.move_gripper(close_gripper = False, save_video=save_video, save_root = save_root, start_step = step_num)
            
            # move to pre-place
            
            print("place_position_up: ", place_position_up)
            step_num, traj = self.control_to_pose(place_position_up, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)


            if save_video:
                images_to_video(f"{save_root}/video", f"{save_root}/video.mp4", frame_size=(int(self.cam_w/2), int(self.cam_h/2)))
                os.system(f"rm {save_root}/video/step-*.png")
        else:
            self.run_steps(10)
            # import pdb; pdb.set_trace()
            # rotation_matrix = np.array([[0, 0, -1],[0, 1, 0],[1, 0, 0]])
            # rotation_matrix = np.array([[0, 0,1, ],[1, 0, 0,],[0,1,  0,]])
            # q_new = R.from_matrix(rotation_matrix).as_quat()
            if final_rotation is None:
                final_rotation_new = None
            else:
                # original_rotation = R.from_quat(final_rotation)
                # rotation_rotation = R.from_quat(q_new)
                final_rotation_new = final_rotation #(original_rotation * rotation_rotation).as_quat()
            #import pdb; pdb.set_trace()
            # rotation_matrix = np.array([[1, 0, 0,],[0, 1, 0,],[0, 0, -1,]])
            # q_new = R.from_matrix(rotation_matrix).as_quat()
            # original_rotation = R.from_quat(final_rotation_new)
            # rotation_rotation = R.from_quat(q_new)
            # final_rotation_new = (original_rotation * rotation_rotation).as_quat()
            # import pdb; pdb.set_trace()
            self.move_obj_to_pose(place_position, final_rotation_new)
            ### place the target object to the predicted position
            # self.add_obj_to_env(grasp_obj_urdf, [place_position[0], place_position[1], place_position[2]+0.1], final_rotation_new)

            ### render new image with the new object added
            points_envs, colors_envs, rgb_envs, depth_envs, seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation()

            ### save new render results
            # self.save_render(rgb_envs=rgb_envs[0], depth_envs=depth_envs[0], ori_points_env=ori_points_envs[0], 
            #                 ori_colors_env=ori_colors_envs[0], points=points_envs[0], colors=colors_envs[0], 
            #                 save_dir = save_root, save_name = "after_plc")
            self.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, save_dir = save_root, save_name = "after_plc")
            
            ### get the final position of all the object
            self.refresh_observation(get_visual_obs=False)
        
            self.run_steps(pre_steps=30)
            
            ### render new image with the new object added
            points_envs, colors_envs, rgb_envs, depth_envs, seg_envs, ori_points_envs, ori_colors_envs, \
                pixel2pointid, pointid2pixel = self.refresh_observation()

            ### save new render results
            # self.save_render(rgb_envs=rgb_envs[0], depth_envs=depth_envs[0], ori_points_env=ori_points_envs[0], 
            #                 ori_colors_env=ori_colors_envs[0], points=points_envs[0], colors=colors_envs[0], 
            #                 save_dir = save_root, save_name = "after_plc")
            self.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, 
                    save_dir = save_root, save_name = "after_wait")
            
        final_obj_pos = []
        for obj_actor_idx in self.obj_actor_idxs[0]:
            final_obj_pos.append(self.rb_states[obj_actor_idx, :3].cpu().numpy())
        return final_obj_pos

    def add_obj_to_env(self, urdf_path, obj_pose_p, final_rotation):
        obj_asset_file = urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = False
        
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000
        self.num_asset_per_env+=1
        self.obj_assets.append(self.gym.load_asset(self.sim, self.asset_root, obj_asset_file, asset_options))
        self.obj_num_links+=1
        
        for i in range(self.num_envs):
            env = self.envs[i]
            initial_pose = gymapi.Transform()
            initial_pose.p.x = obj_pose_p[0]
            initial_pose.p.y = obj_pose_p[1]
            initial_pose.p.z = obj_pose_p[2]
            rotation_noise = 0.0
            initial_pose.r.x = final_rotation[0]
            initial_pose.r.y = final_rotation[1]
            initial_pose.r.z = final_rotation[2]
            initial_pose.r.w = final_rotation[3]

            self.init_obj_pos_list[i].append([initial_pose.p.x, initial_pose.p.y, initial_pose.p.z])
            self.init_obj_rot_list[i].append([initial_pose.r.x, initial_pose.r.y, initial_pose.r.z, initial_pose.r.w])
            
            added_obj_actor_handle = self.gym.create_actor(env, self.obj_assets[-1], initial_pose, 'added_actor', i, 1, self.asset_seg_ids[-1])
            
            obj_actor_idx = self.gym.get_actor_rigid_body_index(env, added_obj_actor_handle, 0, gymapi.DOMAIN_SIM)
            self.obj_actor_idxs[i].append(obj_actor_idx)
            self.gym.set_actor_scale(env, added_obj_actor_handle, self.cfgs["obj_scale"])
        self.gym.prepare_sim(self.sim)
        self.init_observation()

    def run_steps(self, pre_steps = 100, refresh_obs = True, refresh_visual_obs = False, print_step = False):
        # simulation loop
        for frame in range(pre_steps):
            if print_step:
                print("Step: ", frame)
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if refresh_obs:
                self.refresh_observation(get_visual_obs=refresh_visual_obs)
            
            # update viewer
            self.gym.step_graphics(self.sim)
            if not self.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        self.refresh_observation(get_visual_obs=refresh_visual_obs)
            
    def run_placement_pipeline(self, instruction, target_obj_path_relative, save_root):
  
        ### get observation
        points_envs, colors_envs, rgb_envs, depth_envs, seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = self.refresh_observation()

        ### save rendered data
        # self.save_render(rgb_envs=rgb_envs[0], depth_envs=depth_envs[0], ori_points_env=ori_points_envs[0], 
        #                 ori_colors_env=ori_colors_envs[0], points=points_envs[0], colors=colors_envs[0], 
        #                 save_dir = save_root, save_name = "before_plc")
        self.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, 
                save_dir = save_root, save_name = "before_plc")

        ### save to a tmp path for gpt4v inference
        tmp_img_path = save_root + "/rgb_tmp.png"
        imageio.imwrite(tmp_img_path, rgb_envs[0][0])

        ### prompt for gpt4v inference: find the object to query
        prompt = f"Now I want to control a robot to {instruction}. Now the robot is holding the target \
object but need to find a good position to place it.I have a detection model that can tell \
you the accurate position of any given object name. You can invoke this api now. Please give \
me the object you want to invoke the api and know the position.After that I will give you \
the accurate position for you to make further decision. Please list their names splitted by \
space without any other response and text. For example, if the instruction is \"Place the \
cup between the book and the pear\", you can output \"book,pear\". Please note that you should \
only provide the object names, separated by ',', and refrain from including any other information or responses. You should \
also not output \"cup\" because it is on the robot arm instead of the scene."

        ### gpt4v inference
        import pdb; pdb.set_trace()
        response = infer_path(prompt, tmp_img_path)
        print(response.json()['choices'][0]['message']['content'])

        ### get the object bbox information with gsam
        objs = response.json()['choices'][0]['message']['content'].split(",")
        position_info = "The position information is as follow. The axis-aligned bounding box information \
is given by two diagonal corner points"
        objs_str = ",".join(objs)
        for obj in objs:
            masks, bbox_axis_aligned_envs, bbox_center_envs = self.inference_gsam(rgb_envs[0][0], 
                ori_points_envs[0][0], ori_colors_envs[0][0], text_prompt = obj, save_dir = save_root, save_name = "gsam")
            if bbox_axis_aligned_envs == None:
                continue
            bbox = list(bbox_axis_aligned_envs[0])
            bbox_center = bbox_center_envs[0]
            position_info += f"The position information of {obj} is: the axis-aligned bounding box of the \
object is {bbox}, bounding box center is {bbox_center}."


        table_position_str = str(self.cfgs["asset"]["table_pose_p"])
        table_scale_str = str(self.cfgs["table_scale"])
        ### prompt for gpt4v inference: find the position to place the object with the given information
        prompt = f"Now I want to control a robot to {instruction}. Now the robot is holding the target object \
but need to find a good position to place it.I have a detection model that can tell you the \
accurate position of any given object name. You can invoke this api now. In the last query, \
you choose to check the position of {objs_str}: {position_info} Please output the position you want \
to target to be to following the given instruction. In this context, consider the Y-axis to represent left and \
right movements, and the X-axis to represent forward and backward movements. \
To clarify, +Y denotes moving to the right, -Y signifies moving to the left, \
+X corresponds to moving backward, and -X indicates moving forward. The table \
center is at position {table_position_str}. If an object is at [0.2, 0.0, 0.2], \
the back of this object can be [0.3, 0.0, 0.2], with larger x coordination. If an object is at [0.2, 0.0, 0.2], \
the front of this object can be [0.1, 0.0, 0.2], with smaller x coordination. Try not to make collision. The table is big and do not place the \
target object too near to the existing object. The X Y Z length of the table is \
{table_scale_str}. You need to place the target object on the table. \
Please output x y z position splitted by space without any other response and text. Important: \
Your answer will used in downstream api, so please only output 3 float numbers without any other response and text."

        ### gpt4v inference
        response = infer_path(prompt, tmp_img_path)
        print(response.json()['choices'][0]['message']['content'])
        try:
            position = [float(p) for p in response.json()['choices'][0]['message']['content'].split(" ")]
        except:
            response = infer_path(prompt, tmp_img_path)
            print(response.json()['choices'][0]['message']['content'])
            position = [float(p) for p in response.json()['choices'][0]['message']['content'].split(" ")]
        ### place the target object to the predicted position
        self.add_obj_to_env(target_obj_path_relative, [position[0], position[1], position[2]+0.2])

        ### render new image with the new object added
        points_envs, colors_envs, rgb_envs, depth_envs, seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = self.refresh_observation()

        ### save new render results
        # self.save_render(rgb_envs=rgb_envs[0], depth_envs=depth_envs[0], ori_points_env=ori_points_envs[0], 
        #                 ori_colors_env=ori_colors_envs[0], points=points_envs[0], colors=colors_envs[0], 
        #                 save_dir = save_root, save_name = "after_plc")
        self.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, 
                save_dir = save_root, save_name = "after_plc")
        
        ### get the final position of all the object
        self.refresh_observation(get_visual_obs=False)
        
        self.run_steps(pre_steps=30)
        
        ### render new image with the new object added
        points_envs, colors_envs, rgb_envs, depth_envs, seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = self.refresh_observation()

        ### save new render results
        # self.save_render(rgb_envs=rgb_envs[0], depth_envs=depth_envs[0], ori_points_env=ori_points_envs[0], 
        #                 ori_colors_env=ori_colors_envs[0], points=points_envs[0], colors=colors_envs[0], 
        #                 save_dir = save_root, save_name = "after_plc")
        self.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, 
                save_dir = save_root, save_name = "after_wait")
        
        final_obj_pos = []
        for obj_actor_idx in self.obj_actor_idxs[0]:
            final_obj_pos.append(self.rb_states[obj_actor_idx, :3].cpu().numpy())
        return final_obj_pos

    def clean_up(self):
        # cleanup
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
