import isaacgym
from gym.object_gym import ObjectGym
import numpy as np
from gym.utils import read_yaml_config, prepare_gsam_model
import torch
import glob, time
import json
import open3d as o3d

import os
import sys
import tqdm
import os
os.environ['CURL_CA_BUNDLE'] = ''
# from rotation_ablate import RotationEngine
from isaacgym import gymutil
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation
if False:
    from clip_baseline.obj_pose_opt import sample_poses_grid
from rotation_for_overall import RotationEngine

sys.path.append(sys.path[-1]+"/gym")
# set printoptions
torch.set_printoptions(precision=4, sci_mode=False)
time_str = time.strftime("%Y%m%d-%H%M%S")
args = gymutil.parse_arguments(description="Placement",
    custom_parameters=[
        {"name": "--mode", "type": str, "default": ""},
        {"name": "--task_root", "type": str, "default": f"gym_outputs_task_gen_{time_str}"},
        {"name": "--config", "type": str, "default": "config"},
        {"name": "--device", "type": str, "default": "cuda"},
        ])

def init_gym(cfgs, task_cfg=None, random_task = True, index = 0):
    # init gsam
    if cfgs["INFERENCE_GSAM"]:
        grounded_dino_model, sam_predictor = prepare_gsam_model(device=args.device)
    else:
        grounded_dino_model, sam_predictor = None, None
        
    if random_task:
        # read json
        with open("benchmark/dictionary/category_dictionary.json", "r") as f: category_dictionary = json.load(f)
        with open("benchmark/dictionary/instruction_dictionary.json", "r") as f: instruction_dictionary = json.load(f)

        urdf_paths = []
        obj_name = []
        uuids = []

        if "ycb" in cfgs["dataset"]:
            # all the ycb urdf data
            json_dict = json.load(open("benchmark/dictionary/object_dictionary_complete.json"))
            all_uuid = json_dict.keys()
            
            #ycb_urdf_paths = glob.glob("assets/ycb_16k_backup/*/*.urdf")
            ycb_urdf_paths = glob.glob("benchmark/mesh/ycb/*/*.urdf")
            ycb_names = [urdf_path.split("/")[-2] for urdf_path in ycb_urdf_paths]
            ycb_obj_name = [" ".join(name.split("_")[1:-2]) for name in ycb_names]
            ycb_uuid = [urdf_path.split("/")[-2].split("_")[0] for urdf_path in ycb_urdf_paths]
            
            valid_idx = [i for i in range(len(ycb_uuid)) if ycb_uuid[i] in all_uuid]
            
            ycb_uuids = [ycb_uuid[i] for i in valid_idx]
            ycb_urdf_paths = [ycb_urdf_paths[i] for i in valid_idx]
            ycb_obj_name = [" ".join(json_dict[ycb_uuid[i]]['category'].split("_")) for i in valid_idx]
            urdf_paths+=ycb_urdf_paths
            obj_name+=ycb_obj_name
            uuids += ycb_uuids
        if "objaverse" in cfgs["dataset"]:
            json_dict = json.load(open("benchmark/dictionary/object_dictionary_complete.json"))
            
            all_uuid = json_dict.keys()
            # all the objaverse data
            objaverse_urdf_paths = glob.glob("assets/objaverse_final_norm/*/*_2.urdf")
            objaverse_obj_uuid = [path.split("/")[-2] for path in objaverse_urdf_paths]
            
            valid_idx = [i for i in range(len(objaverse_obj_uuid)) if objaverse_obj_uuid[i] in all_uuid]
            objaverse_obj_uuids = [objaverse_obj_uuid[i] for i in valid_idx]
            objaverse_urdf_paths = [objaverse_urdf_paths[i] for i in valid_idx]
            objaverse_obj_name = [" ".join(json_dict[objaverse_obj_uuid[i]]['category'].split("_")) for i in valid_idx]
            urdf_paths+=objaverse_urdf_paths
            obj_name+=objaverse_obj_name
            uuids+=objaverse_obj_uuids
        if "objaverse_old" in cfgs["dataset"]:
            json_dict = json.load(open("category_dictionary.json"))
            
            all_uuid = []
            for key in json_dict.keys(): all_uuid+=json_dict[key]["object_uuids"]
            # all the objaverse data
            objaverse_urdf_paths = glob.glob("benchmark/mesh/objaverse_final_norm/*/*_2.urdf")
            objaverse_names = [urdf_path.split("/")[-2] for urdf_path in objaverse_urdf_paths]
            objaverse_obj_name = [" ".join(name.split("_")[1:]) for name in objaverse_names]
            objaverse_obj_uuid = [name.split("_")[0] for name in objaverse_names]
            valid_idx = [i for i in range(len(objaverse_obj_uuid)) if objaverse_obj_uuid[i] in all_uuid]
            objaverse_urdf_paths = [objaverse_urdf_paths[i] for i in valid_idx]
            objaverse_obj_name = [objaverse_obj_name[i] for i in valid_idx]
            # import pdb; pdb.set_trace()
            urdf_paths+=objaverse_urdf_paths
            obj_name+=objaverse_obj_name

        # index
        # urdf_paths = [urdf_paths[index]]*6
        # obj_name = [obj_name[index]]*6
        orientation_id = np.random.randint(7)
        # orientation_id = np.random.randint(2) + 2
        orientation = ["left", "right", "front", "behind", "between", "center", "top"][orientation_id]
        
        if orientation == "center":
            selected_obj_num = np.random.randint(4, 5)
        elif orientation == "between":
            selected_obj_num = np.random.randint(3, 5)
        else:
            selected_obj_num = np.random.randint(2, 5)
        
        total_asset_num = len(urdf_paths)
        
        obj_idxs = np.random.choice(total_asset_num, selected_obj_num, replace=False)
        
        selected_obj_urdfs = ["/".join(urdf_paths[idx].split("/")[1:]) for idx in obj_idxs]
        selected_obj_names = [obj_name[idx] for idx in obj_idxs]
        selected_uuid = [uuids[idx] for idx in obj_idxs]

        target_obj_name = selected_obj_names[-1]
        target_obj_urdf = selected_obj_urdfs[-1]
        target_obj_uuid = selected_uuid[-1]
        
        init_pose_ = cfgs["asset"]["table_pose_p"].copy()
        init_pose_[2]+= cfgs["asset"]["table_scale"][2]/2+0.1
        
        selected_ob_poses = [init_pose_ for i in range(selected_obj_num)]

        instruction = f"place the {target_obj_name} at the center of all the objects on the table"

        if orientation == "between":
            instruction = f"Place the {target_obj_name} between the {selected_obj_names[0]} and the {selected_obj_names[1]} on the table. "
        elif orientation == "center":
            instruction = f"Place the {target_obj_name} at the center of all the objects on the table. "
        elif orientation == "top":
            instruction = f"Place the {target_obj_name} on top of the {selected_obj_names[0]} on the table. "
        elif orientation == "behind":
            instruction = f"Place the {target_obj_name} behind the {selected_obj_names[0]} on the table. "
        elif orientation == "front":
            instruction = f"Place the {target_obj_name} in front of {selected_obj_names[0]} on the table. "
        else:
            instruction = f"Place the {target_obj_name} to the {orientation} of the {selected_obj_names[0]} on the table. "

        if cfgs["WITH_ROTATION"]:
            if False:
                rotations = ["upright", "sideways", "upside_down"]
                rotation_id = np.random.randint(len(rotations))
                rotation = rotations[rotation_id]
                rotation_instruction = f"We also need to specify the rotation of the object: "
                if rotation == "upright":
                        rotation_instruction += "the object should be placed upright on the table. The best placement should corresponds with how human usually place the object, bottom down and top up."
                elif rotation == "sideways":
                        #"By analyzing the placement of the object in each of these images, which image contains the best placement of the object? Please tell me the index of the image. Best placement means the object should be upright.",
                        #"How many input images are there in total?"# By analyzing the placement of the object in each of these images, which image contains the best placement of the object? Please tell me the index of the image. Best placement means the object should be upright."
                        rotation_instruction += "the object is not in an upright or upside down position, rather, it is lying on its side."
                elif rotation == "upside_down":
                        rotation_instruction += "the top or cap of the object is likely in contact with the table while the bottom side is facing up. Also, if there's texture on the object, the text is likely to be in a flipped manner as well."
                else:
                    import pdb; pdb.set_trace()
                instruction_save = instruction + rotation
                instruction = instruction + " " + rotation_instruction
            else:
                position_instruction = instruction
                target_obj_uuid
                target_name = "_".join(target_obj_name.split(" "))
                instruction_labels = category_dictionary[target_name]["instruction_labels"]
                if len(instruction_labels) == 1:
                    instruction_label = instruction_labels[0]
                else:
                    instruction_label = instruction_labels[np.random.randint(1, len(instruction_labels))]
                rotation_instruction_partial = instruction_dictionary[instruction_label]['prompt']
                rotation_instruction = f"We also need to specify the rotation of the object after placement: {rotation_instruction_partial}"
                instruction = position_instruction + rotation_instruction
            instruction_save = position_instruction+"_"+instruction_label
        else:
            # pass
            instruction_save = instruction
        
        
        print("#" * 50)
        print("selected objects: ", selected_obj_names)
        print("target object: ", target_obj_name)
        print("instruction: ", instruction)
        print("#" * 50)
        
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_root = cfgs["SAVE_ROOT"] + "/" + orientation + "/" + instruction_save + "/" + time_str
        save_root = save_root.replace(" ", "_")
        if cfgs["USE_CUROBO"] == True: save_root = save_root + "_interaction"
        else: save_root = save_root + "_no_interaction"
        # import pdb; pdb.set_trace()
        os.makedirs(save_root, exist_ok=True)
        
        selected_ob_pose_rs = None
        
    else:
        # need to change
        selected_obj_names = task_cfg["selected_obj_names"]
        # all_urdf_paths = glob.glob("assets/ycb_16k_backup/*/*.urdf")+ glob.glob("assets/objaverse_final_norm/*/*.urdf")
        selected_obj_urdfs=task_cfg["selected_urdfs"]
        
        # for obj_name in selected_obj_names:
        #     obj_name = "_".join(obj_name.split(" "))
        #     print(len([all_urdf_path for all_urdf_path in all_urdf_paths if obj_name in all_urdf_path]))
        #     try:
        #         selected_obj_urdfs.append(["/".join(all_urdf_path.split("/")[1:]) for all_urdf_path in all_urdf_paths if obj_name in all_urdf_path][0])
        #     except:
        #         exit()
        # import pdb; pdb.set_trace()
        selected_obj_num = len(selected_obj_names)
        instruction = task_cfg["instruction"]
        selected_ob_poses = task_cfg["init_obj_pos"]
        selected_ob_pose_rs = [pose[3:] for pose in selected_ob_poses]
        save_root = task_cfg["save_root"]
        orientation = task_cfg["orientation"]
        cfgs["asset"]["position_noise"] = [0,0,0]
        cfgs["asset"]["rotation_noise"] = 0
        print(selected_obj_urdfs)
        target_obj_name = selected_obj_names[-1]
        target_obj_urdf = selected_obj_urdfs[-1]
        
        if cfgs["WITH_ROTATION"]:
            rotation_instruction = task_cfg["rotation_instruction"]
            position_instruction = task_cfg["position_instruction"]
            instruction_label = task_cfg["rotation_instruction_label"]
        
    cfgs["asset"]["asset_files"] = selected_obj_urdfs
    cfgs["asset"]["asset_seg_ids"] = [2 + i for i in range(selected_obj_num)]
    cfgs["asset"]["obj_pose_ps"] = selected_ob_poses
    cfgs["asset"]["obj_pose_rs"] = selected_ob_pose_rs
    
    if cfgs["WITH_ROTATION"]:
        engine = RotationEngine()
        asset_folder = target_obj_urdf.split("/")[0]
        obj_folder = target_obj_urdf.split("/")[1]
        
    if False: # whether to use rotation engine
        final_rotation = engine.get_final_rotation(
            # original_mesh_folder=args.original_mesh_folder,
            # instructions=args.instructions,
            # rendering_view=args.rendering_view
            original_mesh_folder=f"assets/{asset_folder}/{obj_folder}",
            asset_root=cfgs["asset_root"]+"/"+asset_folder,
            mesh_urdf_path="/".join(target_obj_urdf.split("/")[1:]),
            instructions=["upright"], #specified label
            prompts = [], # would paralyze 'instructions' if 'prompts' is not empty
            output_folder='output/rotation',
            rendering_view=30,
            multi_image=True
            # /home/haoran/Projects/ObjectPlacement/gym/assets/ycb_16k_backup/003_cracker_box_google_16k/003_cracker_box_google_16k.urdf
        )
    else:
        final_rotation = None
        object_label_prompt = "None"
    
    # orientation = "test"
    # instruction = "test_grasp_placement"
    # gym = ObjectGym(cfgs, grounded_dino_model, sam_predictor)
    gym = ObjectGym(cfgs, grounded_dino_model, sam_predictor)
    
    gym.refresh_observation(get_visual_obs=False)
    gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
    gym.refresh_observation(get_visual_obs=False)
    gym.save_root = save_root
    
    final_obj_pos = []
    for obj_actor_idx in gym.obj_actor_idxs[0]:
        final_obj_pos.append(gym.rb_states[obj_actor_idx].cpu().numpy())
    print([dd[0] for dd in final_obj_pos])
    print(gym.init_obj_pos_list)
    task_config_now = {
        "orientation": orientation,
        "rotation": object_label_prompt,
        "selected_obj_names": selected_obj_names,
        "selected_urdfs": selected_obj_urdfs,
        "target_obj_name": target_obj_name,
        "instruction": instruction,
        "init_obj_pos": [final_obj_pos_.tolist() for final_obj_pos_ in final_obj_pos],
    }
    task_config_now["position_instruction"] = instruction
    
    if cfgs["WITH_ROTATION"]:
        task_config_now["rotation_instruction"] = rotation_instruction
        task_config_now["position_instruction"] = position_instruction
        task_config_now["rotation_instruction_label"] = instruction_label
    if random_task:
        with open(f"{save_root}/task_config.json", "w") as f: json.dump(task_config_now, f)
    else:
        with open(f"{save_root}/task_config_test.json", "w") as f: json.dump(task_config_now, f)
    return gym, cfgs, task_config_now


if args.mode == "gen_task":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = False
    cfgs["SAVE_ROOT"] = f"output/{args.task_root}"
    for  i in range(10000):
        gym, cfgs, task_config_now= init_gym(cfgs, index=i, random_task=True)

        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        gym.save_render(rgb_envs=rgb_envs,
                        depth_envs=depth_envs,
                        ori_colors_env=ori_colors_envs,
                        ori_points_env=ori_points_envs,
                        points=points_envs,
                        colors=colors_envs,
                        save_dir=gym.save_root,
                        save_name=f"before")
        # gym.move_obj_to_pose([0.3, 0, 0.35], [0, 0, 0, 1])
        # gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
        # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
        #     pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)

        # gym.save_render(rgb_envs=rgb_envs,
        #                 depth_envs=depth_envs,
        #                 ori_colors_env=ori_colors_envs,
        #                 ori_points_env=ori_points_envs,
        #                 points=points_envs,
        #                 colors=colors_envs,
        #                 save_dir=gym.save_root,
        #                 save_name="after_movement")
        gym.clean_up()
        del gym
elif args.mode == "gen_task_rot":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = False
    cfgs["WITH_ROTATION"] = True
    cfgs["SAVE_ROOT"] = f"output/{args.task_root}"
    for  i in range(10000):
        gym, cfgs, task_config_now= init_gym(cfgs, index=i, random_task=True)

        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        gym.save_render(rgb_envs=rgb_envs,
                        depth_envs=depth_envs,
                        ori_colors_env=ori_colors_envs,
                        ori_points_env=ori_points_envs,
                        points=points_envs,
                        colors=colors_envs,
                        save_dir=gym.save_root,
                        save_name=f"before")
        # gym.move_obj_to_pose([0.3, 0, 0.35], [0, 0, 0, 1])
        # gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
        # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
        #     pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)

        # gym.save_render(rgb_envs=rgb_envs,
        #                 depth_envs=depth_envs,
        #                 ori_colors_env=ori_colors_envs,
        #                 ori_points_env=ori_points_envs,
        #                 points=points_envs,
        #                 colors=colors_envs,
        #                 save_dir=gym.save_root,
        #                 save_name="after_movement")
        gym.clean_up()
        del gym
elif args.mode == "interaction":
    raise NotImplementedError
elif args.mode == "move_obj":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = True
    task_root = args.task_root
    task_cfgs_paths = glob.glob(f"output/{task_root}/*/*/*/task_config.json")
    # random shuffle
    import random
    random.shuffle(task_cfgs_paths)
    for task_i, task_cfgs_path in enumerate(task_cfgs_paths):
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        if os.path.exists(task_cfg["save_root"]+"/result.json"):
            print("result.json exists, skip")
            continue
        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        
        print(gym.save_root)
        
        final_obj_pos = gym.interaction(
            instruction = task_config_now["instruction"],
            grasp_obj_urdf = task_config_now["selected_urdfs"][-1],
            final_rotation = None,
            save_root=gym.save_root, save_video=cfgs["SAVE_VIDEO"], 
            use_3d=cfgs["USE_3D"])

        results = {
            "selected_obj_urdf": task_config_now["selected_urdfs"],
            "target_obj_urdf": task_config_now["selected_urdfs"][-1],
            "instruction": task_config_now["instruction"],
            "final_obj_pos": [final_obj_pos_.tolist() for final_obj_pos_ in final_obj_pos],
        }
        
        # print basic info
        print("#" * 50)
        print("instruction: ", task_config_now["instruction"])
        print("selected objects: ", task_config_now["selected_obj_names"])
        print("target object: ", task_config_now["target_obj_name"])
        print("#" * 50)
        
        with open(f"{gym.save_root}/result.json", "w") as f:
            json.dump(results, f)
        
        gym.clean_up()
        del gym
elif args.mode == "render_clip":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = False
    cfgs["cam"]["cam_w"] = int(2160/4)
    cfgs["cam"]["cam_h"] = int(1440/4)
    task_root = args.task_root
    task_cfgs_paths = glob.glob(f"output/{task_root}/*/*/*/task_config.json")
    # random shuffle
    import random
    random.shuffle(task_cfgs_paths)
    for task_i, task_cfgs_path in enumerate(tqdm.tqdm(task_cfgs_paths)):
    # for task_i, task_cfgs_path in enumerate(task_cfgs_paths):
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        save_root=task_cfg["save_root"]
        if len(glob.glob(f"{save_root}/2499_*.png")) >= 1:
            print("render finished")
            continue
        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        
        print("outputs saved to:", gym.save_root)
        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
        pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
    
        gym.save_render(rgb_envs=rgb_envs,
                depth_envs=None,
                ori_colors_env=colors_envs,
                ori_points_env=None,
                points=None,
                colors=colors_envs,
                save_dir=gym.save_root,
                save_name="before_movement",
                save_single=True)
        ## TODO: pose_batch define
        ######################################################
        device = torch.device("cuda:0")
        x_range = (0.25, 0.75)
        y_range = (-0.35, 0.35)
        z_range = (0.35, 0.4)
        sample_res=[5, 4, 1, 5, 5, 5]

        pose_batch = sample_poses_grid(device, x_range, y_range, z_range, sample_res)

        np.savetxt(os.path.join(gym.save_root, "pose_batch.txt"), pose_batch)

        for pose_i, pose in enumerate(tqdm.tqdm(pose_batch)):
            object_position = pose[:3]

            object_rotation_euler = pose[3:]
            
            object_rotation_quat = euler2quat(object_rotation_euler[0], object_rotation_euler[1], object_rotation_euler[2], 'sxyz') 
            object_rotation_quat = [object_rotation_quat[1], object_rotation_quat[2], object_rotation_quat[3], object_rotation_quat[0]]

            #print(object_rotation_quat)

            gym.move_obj_to_pose(object_position, object_rotation_quat) # object_position = [0.3, 0, 0.35], object_rotation = [0, 0, 0, 1]
            gym.run_steps(pre_steps = 1, refresh_obs=False, print_step=False)
            points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)

            gym.save_render(rgb_envs=rgb_envs,
                            depth_envs=None,
                            ori_colors_env=colors_envs,
                            ori_points_env=None,
                            points=None,
                            colors=colors_envs,
                            save_dir=gym.save_root,
                            save_name=f"{pose_i}_position:{object_position}_rotation:{object_rotation_quat}",
                            save_single=True)
        output_image_path = gym.save_root
    
        # final_obj_pos = gym.interaction(
        #     instruction = task_config_now["instruction"],
        #     grasp_obj_urdf = task_config_now["selected_urdfs"][-1],
        #     final_rotation = None,
        #     save_root=gym.save_root, save_video=cfgs["SAVE_VIDEO"], 
        #     use_3d=cfgs["USE_3D"])

        # results = {
        #     "selected_obj_urdf": task_config_now["selected_urdfs"],
        #     "target_obj_urdf": task_config_now["selected_urdfs"][-1],
        #     "instruction": task_config_now["instruction"],
        #     "final_obj_pos": [final_obj_pos_.tolist() for final_obj_pos_ in final_obj_pos],
        # }
        
        # # print basic info
        # print("#" * 50)
        # print("instruction: ", task_config_now["instruction"])
        # print("selected objects: ", task_config_now["selected_obj_names"])
        # print("target object: ", task_config_now["target_obj_name"])
        # print("#" * 50)
        
        # with open(f"{gym.save_root}/result.json", "w") as f:
        #     json.dump(results, f)
        
        gym.clean_up()
        del gym
elif args.mode == "render_top_down":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = True
    cfgs["WITH_ROTATION"] = True
    cfgs["cam"]["cam_poss"] = [[0.5, 0, 1.1]]
    cfgs["cam"]["cam_targets"] = [[0.51, 0, 0.3]]
    cfgs["asset"]["franka_pose_p"] = [-1, 0, 0]
    for  i in range(10000):
        gym, cfgs, task_config_now= init_gym(cfgs, index=i, random_task=True)

        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        # import pdb; pdb.set_trace()
        gym.save_render(rgb_envs=rgb_envs,
                        depth_envs=depth_envs,
                        ori_colors_env=ori_colors_envs,
                        ori_points_env=ori_points_envs,
                        points=points_envs,
                        colors=colors_envs,
                        save_dir=gym.save_root,
                        save_name=f"before")
        masks, bbox_axis_aligned_envs, bbox_center_envs = gym.inference_gsam(rgb_envs[0][0], ori_points_envs[0][0], ori_colors_envs[0][0], text_prompt = task_config_now["selected_obj_names"][-1], save_dir = gym.save_root, save_name = "gsam")
        image_raw = rgb_envs[0][0]
        masked_img = image_raw.copy()
        masked_img[~masks[0][0].cpu().numpy()] = 255
        cropped_img_range_xmin = max(0, np.where(masks[0][0].cpu().numpy())[0].min()-10)
        cropped_img_range_xmax = min(image_raw.shape[0], np.where(masks[0][0].cpu().numpy())[0].max()+10)
        cropped_img_range_ymin = max(0, np.where(masks[0][0].cpu().numpy())[1].min()-10)
        cropped_img_range_ymax = min(image_raw.shape[1], np.where(masks[0][0].cpu().numpy())[1].max()+10)
        masked_img_cropped = masked_img[cropped_img_range_xmin:cropped_img_range_xmax,cropped_img_range_ymin:cropped_img_range_ymax]
        import imageio
        imageio.imwrite(f"{gym.save_root}/masked_img.png", masked_img[...,:3])
        imageio.imwrite(f"{gym.save_root}/masked_img_cropped.png", masked_img_cropped[...,:3])
        # import pdb; pdb.set_trace()
        # gym.move_obj_to_pose([0.3, 0, 0.35], [0, 0, 0, 1])
        # gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
        # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
        #     pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)

        # gym.save_render(rgb_envs=rgb_envs,
        #                 depth_envs=depth_envs,
        #                 ori_colors_env=ori_colors_envs,
        #                 ori_points_env=ori_points_envs,
        #                 points=points_envs,
        #                 colors=colors_envs,
        #                 save_dir=gym.save_root,
        #                 save_name="after_movement")
        gym.clean_up()
        del gym
elif args.mode == "count_img_gen":
    paths = glob.glob(f"output/{args.task_root}/*/*/*/2499_*.png")
    total = len(glob.glob(f"output/{args.task_root}/*/*/*/task_config.json"))
    print(len(paths),total)
elif args.mode == "for_clip":
    
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = False
    cfgs["cam"]["cam_w"] = int(2160/4)
    cfgs["cam"]["cam_h"] = int(1440/4)
    task_root = args.task_root
    task_cfgs_paths = glob.glob(f"output/{task_root}/*/*/*/task_config.json")
    for task_cfgs_path in task_cfgs_paths:
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        instruction = task_cfg["position_instruction"] + " " + task_cfg["rotation_instruction"]
        # import pdb; pdb.set_trace()
        sampled_image_folder = "/".join(task_cfgs_path.split("/")[:-1])
        # import pdb; pdb.set_trace()
        output_root = "output/baseline_overall/objaverse0"
        device = torch.device("cuda:0")
        # instructions don't overlap?
        output_dir = os.path.join(output_root, instruction)
        get_clip_best_rotation(sampled_image_folder, instruction, output_dir, device)
elif args.mode == "move_obj_position":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = True
    cfgs["WITH_ROTATION"] = False
    task_root = args.task_root
    task_cfgs_paths = glob.glob(f"output/{task_root}/*/*/*/task_config.json")
    # random shuffle
    import random
    random.shuffle(task_cfgs_paths)
    for task_i, task_cfgs_path in enumerate(task_cfgs_paths):
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        if os.path.exists(task_cfg["save_root"]+"/result.json"):
            print("result.json exists, skip")
            continue
        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        
        print(gym.save_root)
        
        final_obj_pos = gym.interaction(
            instruction = task_config_now["position_instruction"],
            grasp_obj_urdf = task_config_now["selected_urdfs"][-1],
            final_rotation = None,
            save_root=gym.save_root, save_video=cfgs["SAVE_VIDEO"], 
            use_3d=cfgs["USE_3D"])

        results = {
            "selected_obj_urdf": task_config_now["selected_urdfs"],
            "target_obj_urdf": task_config_now["selected_urdfs"][-1],
            "position_instruction": task_config_now["position_instruction"],
            "final_obj_pos": [final_obj_pos_.tolist() for final_obj_pos_ in final_obj_pos],
        }
        
        # print basic info
        print("#" * 50)
        print("instruction: ", task_config_now["position_instruction"])
        print("selected objects: ", task_config_now["selected_obj_names"])
        print("target object: ", task_config_now["target_obj_name"])
        print("#" * 50)
        
        with open(f"{gym.save_root}/result_position.json", "w") as f:
            json.dump(results, f)
        
        gym.clean_up()
        del gym
elif args.mode == "move_obj_position_rotation":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = True
    cfgs["WITH_ROTATION"] = True
    cfgs["cam"]["cam_poss"] = [[1, 0, 1.2], [0.5, 0, 1.1]]
    cfgs["cam"]["cam_targets"] = [[0.5, 0, 0.15], [0.501, 0, 0.3]]
    cfgs["asset"]["franka_pose_p"] = [-1, 0, 0]
    task_root = args.task_root
    task_cfgs_paths = glob.glob(f"output/{task_root}/*/*/*/task_config.json")
    # random shuffle
    import random
    random.shuffle(task_cfgs_paths)
    # import pdb; pdb.set_trace()
    for task_i, task_cfgs_path in enumerate(task_cfgs_paths):
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        if os.path.exists(task_cfg["save_root"]+"/result.json"):
            print("result.json exists, skip")
            continue
        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        save_root = gym.save_root
        print(gym.save_root)
        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        
        masks, bbox_axis_aligned_envs, bbox_center_envs = gym.inference_gsam(rgb_envs[0][1], ori_points_envs[0][1], ori_colors_envs[0][1], text_prompt = task_config_now["selected_obj_names"][-1], save_dir = gym.save_root, save_name = "gsam")
        image_raw = rgb_envs[0][1]
        masked_img = image_raw.copy()
        masked_img[~masks[0][0].cpu().numpy()] = 255
        cropped_img_range_xmin = max(0, np.where(masks[0][0].cpu().numpy())[0].min()-10)
        cropped_img_range_xmax = min(image_raw.shape[0], np.where(masks[0][0].cpu().numpy())[0].max()+10)
        cropped_img_range_ymin = max(0, np.where(masks[0][0].cpu().numpy())[1].min()-10)
        cropped_img_range_ymax = min(image_raw.shape[1], np.where(masks[0][0].cpu().numpy())[1].max()+10)
        masked_img_cropped = masked_img[cropped_img_range_xmin:cropped_img_range_xmax,cropped_img_range_ymin:cropped_img_range_ymax]
        import imageio
        imageio.imwrite(f"{gym.save_root}/masked_img.png", masked_img[...,:3])
        imageio.imwrite(f"{gym.save_root}/masked_img_cropped.png", masked_img_cropped[...,:3])
        
        gym.clean_up()
        del gym
        continue

        ###
 
        _object_uuid = task_config_now["selected_urdfs"][-1].split("/")[1].split("_")[0]
        _initial_scene_image = f"{save_root}/masked_img_cropped.png"
        _rotation_instructions = [task_config_now["rotation_instruction"].split(": ")[1]]
       
        _instruct_dict = "/data/yufei/object_rearrangement/benchmark/dictionary/instruction_dictionary.json"
        print("=====================================================")
        print(_object_uuid, "starts rotation proposal")

        _exp_name = "overall_ours_test"
        _output_folder = f"output_new/{_exp_name}"

        _result_root = "results"
        _result_dict = os.path.join(_result_root, f"{_exp_name}.json")

        os.makedirs(_output_folder, exist_ok=True)
        os.makedirs(_result_root, exist_ok=True)
        import pdb; pdb.set_trace()
        with RotationEngine() as engine:
            quat_from_engine = engine.get_final_rotation(

                object_name=_object_uuid,
                instructions=_rotation_instructions,
                phase2_ablate=False,
                output_folder=_output_folder,
                instruction_dictionary=_instruct_dict,
                input_image_path=_initial_scene_image,
                visualize=False,
                multi_image=True,
                experiment=True,
                ablate_engine=False,
        
                reconstruct=True,
                result_dictionary=_result_dict,
                asset_root="/data/yufei/projects/ObjectPlacement/assets/reconstructed_for_overall/obj",
        


            )
        init_quat = task_config_now["init_obj_pos"][-1][3:7]
        init_rotation = init_quat.as_quat()

        sim_quat = np.array(quat_from_engine[0])#np.quaternion(float(_r_x), float(_r_y),float(_r_z),float(_r_w))
        R = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 1, 1]])
        r = Rotation.from_matrix(R)
        trans_quat = r.as_quat()
        #final_rotation = Rotation.from_quat(sim_quat) * Rotation.from_quat(trans_quat).inv()
        final_rotation =  Rotation.from_quat(trans_quat) *  Rotation.from_quat(sim_quat) * init_rotation
        #import pdb; pdb.set_trace()
        #final_rotation = R * Rotation.from_quat(sim_quat)
        final_quaternion = final_rotation.as_quat()

        print(_object_uuid, "ends rotation proposal")
        print("Final Rotation:", final_quaternion)
        print("=====================================================")



        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        


  
        final_obj_pos = gym.interaction(
            instruction = task_config_now["position_instruction"],
            grasp_obj_urdf = task_config_now["selected_urdfs"][-1],
            final_rotation = final_quaternion,
            save_root=gym.save_root, save_video=cfgs["SAVE_VIDEO"], 
            use_3d=cfgs["USE_3D"])
        results = {
            "selected_obj_urdf": task_config_now["selected_urdfs"],
            "target_obj_urdf": task_config_now["selected_urdfs"][-1],
            "position_instruction": task_config_now["position_instruction"],
            "final_obj_pos": [final_obj_pos_.tolist() for final_obj_pos_ in final_obj_pos],
        }
        import pdb; pdb.set_trace()
        # print basic info
        print("#" * 50)
        print("instruction: ", task_config_now["position_instruction"])
        print("selected objects: ", task_config_now["selected_obj_names"])
        print("target object: ", task_config_now["target_obj_name"])
        print("#" * 50)
        
        with open(f"{gym.save_root}/result_position.json", "w") as f:
            json.dump(results, f)
        
        gym.clean_up()
        del gym
elif args.mode == "render_move_obj_position_rotation":
    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = True
    cfgs["WITH_ROTATION"] = False
    cfgs["cam"]["cam_poss"] = [[1, 0, 1.2], [0.5, 0, 1.1]]
    cfgs["cam"]["cam_targets"] = [[0.5, 0, 0.15], [0.501, 0, 0.3]]
    cfgs["asset"]["franka_pose_p"] = [-1, 0, 0]
    task_root = args.task_root
    task_cfgs_paths = glob.glob(f"output/{task_root}/*/*/*/task_config.json")
    # /home/haoran/Project/ObjectPlacement/output/position/*/*/task_config.json
    # import pdb; pdb.set_trace()
    print("len(task_cfgs_paths)", len(task_cfgs_paths))
    # random shuffle
    import random
    random.shuffle(task_cfgs_paths)
    for task_i, task_cfgs_path in enumerate(task_cfgs_paths):
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        save_root = task_cfg["save_root"]
        if os.path.exists(f"{save_root}/result.json"):
            print("result.json exists, skip")
            continue
        # len(glob.glob("/home/haoran/Project/ObjectPlacement/output/*/*/*/masked_img_cropped.png"))
        # import pdb; pdb.set_trace()
        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        # import pdb; pdb.set_trace()
        print(gym.save_root)
        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        gym.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, 
                        save_dir = save_root, save_name = "before_new")
        masks, bbox_axis_aligned_envs, bbox_center_envs = gym.inference_gsam(rgb_envs[0][1], ori_points_envs[0][1], ori_colors_envs[0][1], text_prompt = task_config_now["selected_obj_names"][-1], save_dir = gym.save_root, save_name = "gsam")
        image_raw = rgb_envs[0][1]
        masked_img = image_raw.copy()
        masked_img[~masks[0][0].cpu().numpy()] = 255
        cropped_img_range_xmin = max(0, np.where(masks[0][0].cpu().numpy())[0].min()-10)
        cropped_img_range_xmax = min(image_raw.shape[0], np.where(masks[0][0].cpu().numpy())[0].max()+10)
        cropped_img_range_ymin = max(0, np.where(masks[0][0].cpu().numpy())[1].min()-10)
        cropped_img_range_ymax = min(image_raw.shape[1], np.where(masks[0][0].cpu().numpy())[1].max()+10)
        masked_img_cropped = masked_img[cropped_img_range_xmin:cropped_img_range_xmax,cropped_img_range_ymin:cropped_img_range_ymax]
        import imageio
        imageio.imwrite(f"{gym.save_root}/masked_img.png", masked_img[...,:3])
        imageio.imwrite(f"{gym.save_root}/masked_img_cropped.png", masked_img_cropped[...,:3])
        
        # gym.clean_up()
        # del gym
        # continue
    
        from scipy.spatial.transform import Rotation

        ###
        _r_x = input("relative rotation x:")
        _r_y = input("relative rotation y:")
        _r_z = input("relative rotation z:")
        _r_w = input("relative rotation w:")
        sim_quat = np.array([_r_x, _r_y, _r_z, _r_w])#np.quaternion(float(_r_x), float(_r_y),float(_r_z),float(_r_w))
        R = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 1, 1]])
        r = Rotation.from_matrix(R)
        trans_quat = r.as_quat()
        #final_rotation = Rotation.from_quat(sim_quat) * Rotation.from_quat(trans_quat).inv()
        final_rotation =  Rotation.from_quat(trans_quat) *  Rotation.from_quat(sim_quat)
        # import pdb; pdb.set_trace()
        #final_rotation = R * Rotation.from_quat(sim_quat)
        final_quaternion = final_rotation.as_quat()


     

  
        final_obj_pos = gym.interaction(
            instruction = task_config_now["position_instruction"],
            grasp_obj_urdf = task_config_now["selected_urdfs"][-1],
            final_rotation = final_quaternion,
            save_root=gym.save_root, save_video=cfgs["SAVE_VIDEO"], 
            use_3d=cfgs["USE_3D"])

        results = {
            "selected_obj_urdf": task_config_now["selected_urdfs"],
            "target_obj_urdf": task_config_now["selected_urdfs"][-1],
            "position_instruction": task_config_now["position_instruction"],
            "final_obj_pos": [final_obj_pos_.tolist() for final_obj_pos_ in final_obj_pos],
        }
        import pdb; pdb.set_trace()
        # print basic info
        print("#" * 50)
        print("instruction: ", task_config_now["position_instruction"])
        print("selected objects: ", task_config_now["selected_obj_names"])
        print("target object: ", task_config_now["target_obj_name"])
        print("#" * 50)
        
        with open(f"{gym.save_root}/result_position.json", "w") as f:
            json.dump(results, f)
        
        gym.clean_up()
        del gym
elif args.mode == "move_obj_position_rotation_pre_render":
    # import glob
    # len(glob.glob(f"/home/haoran/Project/ObjectPlacement/output/gym_outputs_task_gen_obja_0304_rot_new_filter_not_finished/*/*/*/*crop*.png"))
  

    cfgs = read_yaml_config(f"{args.config}.yaml")
    cfgs["INFERENCE_GSAM"] = False
    cfgs["WITH_ROTATION"] = True
    cfgs["cam"]["cam_poss"] = [[1, 0, 1.2], [0.5, 0, 1.1]]
    cfgs["cam"]["cam_targets"] = [[0.5, 0, 0.15], [0.501, 0, 0.3]]
    cfgs["asset"]["franka_pose_p"] = [-1, 0, 0]
    task_root = args.task_root
    task_cfgs_paths = glob.glob(f"output/{task_root}/*/*/*/task_config.json")
    # random shuffle
    import random
    random.shuffle(task_cfgs_paths)
    # import pdb; pdb.set_trace()
    import shutil
    for task_i, task_cfgs_path in enumerate(task_cfgs_paths):
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        save_root = task_cfg["save_root"]
        task_root = save_root.split("/")[-2]


        _exp_name = "overall_ours_0312"
        _output_folder = f"output_new/{_exp_name}"
        done = False
        for root, dirs, files in os.walk(_output_folder):
            for dir in dirs:
                if task_root == dir:
                    dir_path = os.path.join(root, dir)
                    subdirs = os.listdir(dir_path)
                    if len(subdirs) >= 4 and 'rendered_image' in subdirs:
                        done = True
                        
                        break
                    else:
                        
                        shutil.rmtree(dir_path)

            break
        
        if done:
            continue
        

        # import pdb; pdb.set_trace()

        # if os.path.exists(task_cfg["save_root"]+"/result.json"):
        #     print("result.json exists, skip")
        #     continue

        # if os.path.exists(task_cfg["save_root"]+"/result_rotation.json"):
        #     print("result.json exists, skip")
        #     continue


      
        if not os.path.exists(f"{save_root}/masked_img_cropped.png"):
            print("cropped image not exist, skip")
            continue
   
        # gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        
        # print(gym.save_root)
        # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
        #     pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
        
        # masks, bbox_axis_aligned_envs, bbox_center_envs = gym.inference_gsam(rgb_envs[0][1], ori_points_envs[0][1], ori_colors_envs[0][1], text_prompt = task_config_now["selected_obj_names"][-1], save_dir = gym.save_root, save_name = "gsam")
        # image_raw = rgb_envs[0][1]
        # masked_img = image_raw.copy()
        # masked_img[~masks[0][0].cpu().numpy()] = 255
        # cropped_img_range_xmin = max(0, np.where(masks[0][0].cpu().numpy())[0].min()-10)
        # cropped_img_range_xmax = min(image_raw.shape[0], np.where(masks[0][0].cpu().numpy())[0].max()+10)
        # cropped_img_range_ymin = max(0, np.where(masks[0][0].cpu().numpy())[1].min()-10)
        # cropped_img_range_ymax = min(image_raw.shape[1], np.where(masks[0][0].cpu().numpy())[1].max()+10)
        # masked_img_cropped = masked_img[cropped_img_range_xmin:cropped_img_range_xmax,cropped_img_range_ymin:cropped_img_range_ymax]
        # import imageio
        # imageio.imwrite(f"{gym.save_root}/masked_img.png", masked_img[...,:3])
        # imageio.imwrite(f"{gym.save_root}/masked_img_cropped.png", masked_img_cropped[...,:3])
        
        # from scipy.spatial.transform import Rotation

        ###

        _object_uuid = task_cfg["selected_urdfs"][-1].split("/")[1].split("_")[0]
        _initial_scene_image = f"{save_root}/masked_img_cropped.png"
        if os.path.exists(_initial_scene_image) == False:
            continue
        else:
            print("========================================")
            print("generating goal rotation for:", save_root.split("/")[-2])

        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)

        _rotation_instructions = [task_cfg["rotation_instruction"].split(":", 1)[1]]
        _instruct_dict = "/data/yufei/object_rearrangement/benchmark/dictionary/instruction_dictionary.json"



        _exp_name = "overall_ours_0312"
        _output_folder = f"output_new/{_exp_name}"

        _result_root = "results_overall"
        _result_dict = os.path.join(_result_root, f"{_exp_name}.json")

        os.makedirs(_output_folder, exist_ok=True)
        os.makedirs(_result_root, exist_ok=True)

        init_quat = task_config_now["init_obj_pos"][-1][3:7]
       
        gym.clean_up()
        del gym
        


       
        with RotationEngine() as engine:
            quat_from_engine = engine.get_final_rotation(
                task_id = task_root,
        

                object_name=_object_uuid, 
                instructions=_rotation_instructions,
                phase2_ablate=False,
                output_folder=_output_folder,
                instruction_dictionary=_instruct_dict,
                input_image_path=_initial_scene_image,
                visualize=False,
                multi_image=True,
                experiment=True,
                ablate_engine=False,
        
                reconstruct=True,
                result_dictionary=_result_dict,
                #asset_root="/data/yufei/projects/ObjectPlacement/assets/reconstructed_for_overall/obj",
            )

        with open(f"{save_root}/result_rotation.json", "w") as f:
            json.dump(quat_from_engine, f)
  

        continue
        # init_quat = np.array(init_quat)
        # init_quat = Rotation.from_quat(init_quat)  
        # init_rotation = init_quat.as_matrix()
        # init_rotation = Rotation.from_matrix(init_rotation)
        # init_quat1 = init_rotation.as_quat()

        #quat_from_engine = [[1,2,3,4]]
        sim_quat = np.array(quat_from_engine[0])#np.quaternion(float(_r_x), float(_r_y),float(_r_z),float(_r_w))
        R = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 1, 1]])
        r = Rotation.from_matrix(R)
        trans_quat = r.as_quat()
        #final_rotation = Rotation.from_quat(sim_quat) * Rotation.from_quat(trans_quat).inv()
    
        final_rotation =  Rotation.from_quat(trans_quat) *  Rotation.from_quat(sim_quat) * Rotation.from_quat(init_quat)


        #final_rotation = R * Rotation.from_quat(sim_quat)
        final_quaternion = final_rotation.as_quat()
        import pdb; pdb.set_trace()
        gym, cfgs, task_config_now = init_gym(cfgs, task_cfg=task_cfg, random_task=False)
        
        final_obj_pos = gym.interaction(
            instruction = task_config_now["position_instruction"],
            grasp_obj_urdf = task_config_now["selected_urdfs"][-1],
            final_rotation = final_quaternion,
            save_root=gym.save_root, save_video=cfgs["SAVE_VIDEO"], 
            use_3d=cfgs["USE_3D"])
        results = {
            "selected_obj_urdf": task_config_now["selected_urdfs"],
            "target_obj_urdf": task_config_now["selected_urdfs"][-1],
            "position_instruction": task_config_now["position_instruction"],
            "final_obj_pos": [final_obj_pos_.tolist() for final_obj_pos_ in final_obj_pos],
        }
        # import pdb; pdb.set_trace()
        # print basic info
        print("#" * 50)
        print("instruction: ", task_config_now["position_instruction"])
        print("selected objects: ", task_config_now["selected_obj_names"])
        print("target object: ", task_config_now["target_obj_name"])
        print("#" * 50)
        
        with open(f"{gym.save_root}/result_position.json", "w") as f:
            json.dump(results, f)
        
        gym.clean_up()
        del gym

else:
    raise NotImplementedError   
