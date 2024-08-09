
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import glob, math, json, time
from gym.utils import read_yaml_config, prepare_gsam_model
from gym.utils import images_to_video, quat_axis, orientation_error, cube_grasping_yaw, read_yaml_config, \
    get_downsampled_pc, get_point_cloud_from_rgbd, get_point_cloud_from_rgbd_GPU

class GetTask:
    def __init__(self, cfgs, task_cfgs):
        self.cfgs = cfgs
        self.task_cfgs = task_cfgs
        self.device = cfgs["DEVICE"]
    
        # headless
        self.headless = cfgs["HEADLESS"]

        # prepare gym cfgs
        self._prepare_gym_cfgs()
        
        # setup gym
        self._init_gym()
        
        # setup scene
        self._setup_scene()
        
        # init observation
        self._init_observation()
        

    def _prepare_gym_cfgs(self):
        # camera settings
        self._use_cam = self.cfgs["cam"]["use_cam"]
        if self._use_cam:
            self._cam_w = self.cfgs["cam"]["cam_w"]
            self._cam_h = self.cfgs["cam"]["cam_h"]
            self._cam_far_plane = self.cfgs["cam"]["cam_far_plane"]
            self._cam_near_plane = self.cfgs["cam"]["cam_near_plane"]
            self._horizontal_fov = self.cfgs["cam"]["cam_horizontal_fov"]
            self._cam_poss = self.cfgs["cam"]["cam_poss"]
            self._cam_targets = self.cfgs["cam"]["cam_targets"]
            self._num_cam_per_env = len(self._cam_poss)
            self._point_cloud_bound = self.cfgs["cam"]["point_cloud_bound"]
            
        # segmentation
        self._table_seg_id = self.cfgs["asset"]["table_seg_id"]    
        self._franka_seg_id = self.cfgs["asset"]["franka_seg_id"]
        self._asset_seg_ids = [str(id_) for id_ in range(2, 2 + len(self.task_cfgs["selected_urdfs"]))]
        # Add custom arguments
        self._args = gymutil.parse_arguments(description="Placement",
            custom_parameters=[
                {"name": "--device", "type": str, "default": "cuda"},
                ]
            )
        # configure sim
        self._sim_params = gymapi.SimParams()
        self._sim_params.up_axis = gymapi.UP_AXIS_Z
        self._sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self._sim_params.dt = 1.0 / 60.0
        self._sim_params.substeps = 2
        self._sim_params.use_gpu_pipeline = self._args.use_gpu_pipeline
        assert self._args.physics_engine == gymapi.SIM_PHYSX
        self._sim_params.physx.solver_type = 1
        self._sim_params.physx.num_position_iterations = 8
        self._sim_params.physx.num_velocity_iterations = 1
        self._sim_params.physx.rest_offset = 0.0
        self._sim_params.physx.contact_offset = 0.001
        self._sim_params.physx.friction_offset_threshold = 0.001
        self._sim_params.physx.friction_correlation_distance = 0.0005
        self._sim_params.physx.num_threads = self._args.num_threads
        self._sim_params.physx.use_gpu = self._args.use_gpu

    def _init_gym(self):
        self._gym = gymapi.acquire_gym()
        self._sim = self._gym.create_sim(self._args.compute_device_id, self._args.graphics_device_id, 
                                         self._args.physics_engine, self._sim_params)
        if self._sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        if not self.headless:
            self.viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        
    def _setup_scene(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self._gym.add_ground(self._sim, plane_params)
        
        self.asset_root = self.cfgs["asset"]["asset_root"]
        self._prepare_franka_asset()
        self._prepare_obj_assets()
        self._load_env(load_cam=self._use_cam)

    def _prepare_franka_asset(self):
        self._controller_name = cfgs["controller"]
        # load franka asset
        franka_asset_file = self.cfgs["asset"]["franka_asset_file"]
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self.franka_asset = self._gym.load_asset(self._sim, self.asset_root, franka_asset_file, asset_options)

        # configure franka dofs
        self.franka_dof_props = self._gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = self.franka_dof_props["lower"]
        franka_upper_limits = self.franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
        
        # Set controller parameters
        # IK params
        self.damping = 0.05

        # use position drive for all dofs
        if self._controller_name == "ik" or self._controller_name == "curobo":
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
        self.franka_num_dofs = self._gym.get_asset_dof_count(self.franka_asset)
        self.franka_default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        self.franka_default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        self.franka_default_dof_pos[7:] = franka_upper_limits[7:]

        self.franka_default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.franka_default_dof_state["pos"] = self.franka_default_dof_pos

        # send to torch
        self.default_dof_pos_tensor = to_torch(self.franka_default_dof_pos, device=self.device)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self._gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_num_links = len(franka_link_dict)
        # print("franka dof:", self.franka_num_dofs, "franka links:", self.franka_num_links)
        self.franka_hand_index = franka_link_dict["panda_hand"]

    def _prepare_obj_assets(self):
        self._table_pose_p = self.cfgs["asset"]["table_pose_p"]
        self._table_scale = self.cfgs["asset"]["table_scale"]
        table_dims = gymapi.Vec3(self._table_scale[0], self._table_scale[1], self._table_scale[2])
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(self._table_pose_p[0], self._table_pose_p[1], self._table_pose_p[2])

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self._gym.create_box(self._sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        obj_asset_files = self.task_cfgs["selected_urdfs"]
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000
        self.num_asset_per_env = len(obj_asset_files)
        self.obj_assets = [self._gym.load_asset(self._sim, self.asset_root, obj_asset_file, asset_options) for obj_asset_file in obj_asset_files]
        self.obj_num_links_dict = [self._gym.get_asset_rigid_body_dict(asset_i) for asset_i in self.obj_assets]
        self.obj_num_links = sum([len(obj_num_links) for obj_num_links in self.obj_num_links_dict])
        self.obj_num_dofs = sum([self._gym.get_asset_dof_count(asset_i) for asset_i in self.obj_assets])
        self.table_num_links = 1

    def _load_env(self, load_cam = True):
        self._num_envs = cfgs["num_envs"]
        self._num_per_row = int(math.sqrt(self._num_envs))
        self._spacing = cfgs["env_spacing"]
        self._env_lower = gymapi.Vec3(-self._spacing, -self._spacing, 0.0)
        self._env_upper = gymapi.Vec3(self._spacing, self._spacing, self._spacing)
        
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

        self._asset_files = self.task_cfgs["selected_urdfs"]
        self._asset_seg_ids = [2 + i for i in range(len(self._asset_files))]
        self._obj_pose_ps = self.task_cfgs["init_obj_pos"]
        self._obj_pose_rs = [pose[3:] for pose in self._obj_pose_ps]
        if self._obj_pose_rs is not None:
            obj_pose_rs = [self._obj_pose_rs[obj_i] for obj_i in range(self.num_asset_per_env)]
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
            

        for i in range(self._num_envs):
            # create env
            env = self._gym.create_env(self._sim, self._env_lower, self._env_upper, self._num_per_row)
            self.envs.append(env)
            origin = self._gym.get_env_origin(env)
            self.env_offsets.append([origin.x, origin.y, origin.z])

            # add franka
            franka_handle = self._gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 2, self._franka_seg_id)

            # set dof properties
            self._gym.set_actor_dof_properties(env, franka_handle, self.franka_dof_props)

            # set initial dof states
            self._gym.set_actor_dof_states(env, franka_handle, self.franka_default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self._gym.set_actor_dof_position_targets(env, franka_handle, self.franka_default_dof_pos)

            # get inital hand pose
            hand_handle = self._gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self._gym.get_rigid_transform(env, hand_handle)
            self.init_franka_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_franka_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self._gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)
            
            ### Table
            self.table_handle = self._gym.create_actor(env, self.table_asset, self.table_pose, "table", i, 0, self._table_seg_id)
            
            ## Object Assets
            self.init_obj_pos_list.append([])
            self.init_obj_rot_list.append([])
            self.obj_actor_idxs.append([])
            for asset_i in range(self.num_asset_per_env):
                initial_pose = gymapi.Transform()
                initial_pose.p.x = self._obj_pose_ps[asset_i][0] + np.random.uniform(-1.0, 1.0) * position_noise[0]
                initial_pose.p.y = self._obj_pose_ps[asset_i][1] + np.random.uniform(-1.0, 1.0) * position_noise[1]
                initial_pose.p.z = self._obj_pose_ps[asset_i][2]
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
                
                obj_actor_handle = self._gym.create_actor(env, self.obj_assets[asset_i], initial_pose, f'actor_{asset_i}', i, 0, self._asset_seg_ids[asset_i])
                
                obj_actor_idx = self._gym.get_actor_rigid_body_index(env, obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                self.obj_actor_idxs[i].append(obj_actor_idx)
                self._gym.set_actor_scale(env, obj_actor_handle, self.cfgs["asset"]["obj_scale"])
            
            if self.cfgs["USE_ARTI"]:
                ### Articulated Object
                arti_initial_pose = gymapi.Transform()
                arti_initial_pose.p.x = arti_obj_pose_p[0] + np.random.uniform(-1.0, 1.0) * arti_position_noise
                arti_initial_pose.p.y = arti_obj_pose_p[1] + np.random.uniform(-1.0, 1.0) * arti_position_noise
                arti_initial_pose.p.z = arti_obj_pose_p[2]
                arti_initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), arti_rotation/180.0*math.pi + arti_rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))
                self.arti_init_obj_pos_list.append([arti_initial_pose.p.x, arti_initial_pose.p.y, arti_initial_pose.p.z])
                self.arti_init_obj_rot_list.append([arti_initial_pose.r.x, arti_initial_pose.r.y, arti_initial_pose.r.z, arti_initial_pose.r.w])
                arti_obj_actor_handle = self._gym.create_actor(env, self.arti_obj_asset, arti_initial_pose, 'arti_actor', i, 1, self.asset_seg_ids[-1] + 1)
                self._gym.set_actor_dof_properties(env, arti_obj_actor_handle, self.arti_obj_dof_props)
                # set initial dof states
                ### TODO check
                # self.arti_obj_default_dof_state["pos"][:3] = 2 + np.random.uniform(-1.0, 1.0) * 0.5
                self._gym.set_actor_dof_states(env, arti_obj_actor_handle, self.arti_obj_default_dof_state, gymapi.STATE_ALL)
                # set initial position targets
                self._gym.set_actor_dof_position_targets(env, arti_obj_actor_handle, self.arti_obj_default_dof_state["pos"])
                arti_obj_actor_idx = self._gym.get_actor_rigid_body_index(env, arti_obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                self.arti_obj_actor_idxs.append(arti_obj_actor_idx)
                self._gym.set_actor_scale(env, arti_obj_actor_handle, self.cfgs["asset"]["arti_obj_scale"])
                
            
            if load_cam:
                # add camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self._cam_w
                cam_props.height = self._cam_h
                cam_props.far_plane = self._cam_far_plane
                cam_props.near_plane = self._cam_near_plane 
                cam_props.horizontal_fov = self._horizontal_fov
                cam_props.enable_tensors = True
                self.cams.append([])
                self.depth_tensors.append([])
                self.rgb_tensors.append([])
                self.seg_tensors.append([])
                self.cam_vinvs.append([])
                self.cam_projs.append([])
                for i in range(self._num_cam_per_env):
                    cam_handle = self._gym.create_camera_sensor(env, cam_props)
                    self._gym.set_camera_location(cam_handle, env, 
                        gymapi.Vec3(self._cam_poss[i][0], self._cam_poss[i][1], self._cam_poss[i][2]), 
                        gymapi.Vec3(self._cam_targets[i][0], self._cam_targets[i][1], self._cam_targets[i][2]))
                    self.cams[-1].append(cam_handle)
                
                    proj = self._gym.get_camera_proj_matrix(self._sim, env, cam_handle)
                    # view_matrix_inv = torch.inverse(torch.tensor(self._gym.get_camera_view_matrix(self._sim, env, cam_handle))).to(self.device)
                    vinv = np.linalg.inv(np.matrix(self._gym.get_camera_view_matrix(self._sim, env, cam_handle)))
                    self.cam_vinvs[-1].append(vinv)
                    self.cam_projs[-1].append(proj)

                    # obtain rgb tensor
                    rgb_tensor = self._gym.get_camera_image_gpu_tensor(
                        self._sim, env, cam_handle, gymapi.IMAGE_COLOR)
                    # wrap camera tensor in a pytorch tensor
                    torch_rgb_tensor = gymtorch.wrap_tensor(rgb_tensor)
                    self.rgb_tensors[-1].append(torch_rgb_tensor)
                    
                    # obtain depth tensor
                    depth_tensor = self._gym.get_camera_image_gpu_tensor(
                        self._sim, env, cam_handle, gymapi.IMAGE_DEPTH)
                    # wrap camera tensor in a pytorch tensor
                    torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor)
                    self.depth_tensors[-1].append(torch_depth_tensor)
        

                    # obtain depth tensor
                    seg_tensor = self._gym.get_camera_image_gpu_tensor(
                        self._sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    # wrap camera tensor in a pytorch tensor
                    torch_seg_tensor = gymtorch.wrap_tensor(seg_tensor)
                    self.seg_tensors[-1].append(torch_seg_tensor)
                    
                    
            
        self.env_offsets = np.array(self.env_offsets)
        
        # point camera at middle env
        if not self.headless:
            
            viewer_cam_pos = gymapi.Vec3(self._cam_poss[0][0], self._cam_poss[0][1], self._cam_poss[0][2])
            viewer_cam_target = gymapi.Vec3(self._cam_targets[0][0], self._cam_targets[0][1], self._cam_targets[0][2])
            middle_env = self.envs[self._num_envs // 2 + self._num_per_row // 2]
            self._gym.viewer_camera_look_at(self.viewer, middle_env, viewer_cam_pos, viewer_cam_target)

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self._gym.prepare_sim(self._sim)

    def _init_observation(self):
        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self._gym.acquire_jacobian_tensor(self._sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = self.jacobian[:, self.franka_hand_index - 1, :, :7]

        # get mass matrix tensor
        _massmatrix = self._gym.acquire_mass_matrix_tensor(self._sim, "franka")
        self.mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = self.mm[:, :7, :7]          # only need elements corresponding to the franka arm
        
        # get rigid body state tensor
        _rb_states = self._gym.acquire_rigid_body_state_tensor(self._sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        num_rb = int(self.rb_states.shape[0]/self._num_envs)
        
        self.root_states = gymtorch.wrap_tensor(self._gym.acquire_actor_root_state_tensor(self._sim))
        self._gym.refresh_actor_root_state_tensor(self._sim)
            
        
        if self.cfgs["USE_ARTI"]:
            assert num_rb == self.franka_num_links + self.obj_num_links + self.table_num_links + self.arti_obj_num_links, "Number of rigid bodies in tensor does not match franka & obj asset"
        else:
            assert num_rb == self.franka_num_links + self.obj_num_links + self.table_num_links, "Number of rigid bodies in tensor does not match franka & obj asset"
        
        # get dof state tensor
        _dof_states = self._gym.acquire_dof_state_tensor(self._sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        num_dof = int(self.dof_states.shape[0]/self._num_envs)
        if self.cfgs["USE_ARTI"]:
            assert num_dof == self.franka_num_dofs + self.obj_num_dofs + self.arti_obj_num_dofs, "Number of dofs in tensor does not match franka & obj asset"
        else:
            assert num_dof == self.franka_num_dofs + self.obj_num_dofs, "Number of dofs in tensor does not match franka & obj asset"

        self.dof_pos = self.dof_states[:, 0].view(self._num_envs, num_dof, 1)
        self.dof_vel = self.dof_states[:, 1].view(self._num_envs, num_dof, 1)

    def refresh_observation(self, get_visual_obs = True):
        # refresh tensors
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)
        self._gym.refresh_mass_matrix_tensors(self._sim)
        
        # state obs
        self.hand_pos = self.rb_states[self.hand_idxs, :3]
        self.hand_rot = self.rb_states[self.hand_idxs, 3:7]
        self.hand_vel = self.rb_states[self.hand_idxs, 7:]

        ### TODO: support different dof tensor shapes in different envs
        self.robot_dof_qpos_qvel = self.dof_states.reshape(self._num_envs,-1,2)[:,:self.franka_num_dofs, :].view(self._num_envs, self.franka_num_dofs, 2)
        
        # render sensors and refresh camera tensors
        if self._use_cam and get_visual_obs:
            self._gym.render_all_camera_sensors(self._sim)
            self._gym.start_access_image_tensors(self._sim)
            points_envs = []
            colors_envs = []
            ori_points_envs = []
            ori_colors_envs = []
            rgb_envs = []
            depth_envs = []
            seg_envs = []
            # bbox_axis_aligned_envs = []
            # bbox_center_envs = []
            for env_i in range(self._num_envs):
                points_env = []
                colors_env = []
                rgb_env = []
                depth_env = []
                seg_env = []
                for cam_i_per_env in range(self._num_cam_per_env):
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
                        self._cam_w, self._cam_h
                    )
                    points = pointclouds[:, :3].cpu().numpy()
                    colors = pointclouds[:, 3:6].cpu().numpy()
                    i_indices, j_indices = np.meshgrid(np.arange(self._cam_w), np.arange(self._cam_h), 
                            indexing='ij')
                    pointid2pixel = np.stack((i_indices, j_indices), axis=-1).reshape(-1, 2)
                    pixel2pointid = np.arange(self._cam_w * self._cam_h).reshape(self._cam_w, self._cam_h)
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

            self._gym.end_access_image_tensors(self._sim)

            obs_reture = {
                "points_envs": points_envs,
                "colors_envs": colors_envs,
                "rgb_envs": rgb_envs,
                "depth_envs": depth_envs,
                "seg_envs": seg_envs,
                "ori_points_envs": ori_points_envs,
                "ori_colors_envs": ori_colors_envs,
                "pixel2pointid": pixel2pointid,
                "pointid2pixel": pointid2pixel,
                "hand_pos": self.hand_pos,
                "hand_rot": self.hand_rot,
                "hand_vel": self.hand_vel,
                "robot_dof_qpos_qvel": self.robot_dof_qpos_qvel,
            }
        else:
            obs_reture = {
                "hand_pos": self.hand_pos,
                "hand_rot": self.hand_rot,
                "hand_vel": self.hand_vel,
                "robot_dof_qpos_qvel": self.robot_dof_qpos_qvel,
            }
        
        return obs_reture

    def clean_up(self):
        # cleanup
        if not self.headless:
            self._gym.destroy_viewer(self.viewer)
        self._gym.destroy_sim(self._sim)

if __name__  == "__main__":
    cfgs = read_yaml_config("config.yaml")
    task_cfgs_path = "/home/haoran/Projects/Rearrangement/Open6DOR/Method/tasks/6DoF/behind/Place_the_binder_behind_the_calculator_on_the_table.__lying_flat/20240704-160939_no_interaction/task_config_new2.json"
    with open(task_cfgs_path, "r") as f: task_cfgs = json.load(f)
    
    open6dor_task = GetTask(cfgs=cfgs, task_cfgs=task_cfgs)
    _ = open6dor_task.refresh_observation()
    open6dor_task.clean_up()