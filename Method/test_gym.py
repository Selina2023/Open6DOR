# exit()
import sys
import os
sys.path = [os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))] +  sys.path
sys.path = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))] + sys.path

# import gym
from gym.object_gym import ObjectGym
from gym.utils import read_yaml_config

import json, glob, random

tag = "handle_right"
# tag = "upside"
anno_path = f"/home/haoran/Projects/Rearrangement/Open6DOR/Benchmark/benchmark_catalogue/annotation/annotation_{tag}.json"
# anno_path = f"/home/haoran/Projects/Rearrangement/ObjectPlacement/rotation_anno/annotation_upright_1_.json"
save_root_ = f"/home/haoran/Projects/Rearrangement/Open6DOR/anno_test/anno_images-final_-{tag}"
anno_data = json.load(open(anno_path, 'r'))
anno_keys = list(anno_data.keys())
# import pdb; pdb.set_trace()
random.shuffle(anno_keys)
for anno in anno_keys:
    # print(anno["object_name"], anno["upright"])
    anno_data_i = anno_data[anno]['annotation']
    obj_id = anno
    save_root = f"{save_root_}/{tag}-{obj_id}"
    # if os.path.exists(f"/home/haoran/Projects/Rearrangement/ObjectPlacement/rotation_anno/anno_images/upright-{obj_id}/task_config-rgb-0-0.png"):
    if os.path.exists(f"{save_root_}/{tag}-{obj_id}/task_config-rgb-0-0.png"):
        continue
    cfgs = read_yaml_config("config.yaml")
    
    if len(obj_id) > 10: # objaverse
        cfgs["asset"]["asset_files"] = [f"objaverse_final_norm/{obj_id}/material_2.urdf"]
    else:
        path = glob.glob(f"assets/ycb_16k_backup/{obj_id}*/{obj_id}*.urdf")[0]
        path_r = "/".join(path.split("/")[-3:])
        cfgs["asset"]["asset_files"] = [path_r]
    if len(list(anno_data_i.keys())) > 1:
        import pdb; pdb.set_trace()
    try:
        quat_anno = anno_data_i[list(anno_data_i.keys())[0]]["quat"]
    except:
        continue
    if anno_data_i[list(anno_data_i.keys())[0]]["stage"] != 1 and anno_data_i[list(anno_data_i.keys())[0]]["stage"] != 2:
        import pdb; pdb.set_trace()
    
    cfgs["asset"]["obj_pose_ps"] = [[0.5, 0, 0.4]]
    try:
        cfgs["asset"]["obj_pose_rs"]  = [[quat_anno[0][0], quat_anno[0][1], quat_anno[0][2],quat_anno[0][3],]]
    except:
        cfgs["asset"]["obj_pose_rs"]  = [[quat_anno[0], quat_anno[1], quat_anno[2],quat_anno[3],]]
        
    cfgs["asset"]["position_noise"] = [0, 0]
    cfgs["asset"]["rotation_noise"] = 0
        # cfgs["asset"]["asset_files"] = [obj_id]
    # cfgs["asset"]["asset_files"] = anno["object_name"]
    gym = ObjectGym(cfgs, None, None, pre_steps = 0)
    
    print(list(anno_data_i.keys())[0])
    gym.refresh_observation(get_visual_obs=False)
    # save_root = f"/home/haoran/Projects/Rearrangement/ObjectPlacement/rotation_anno/anno_images2/upright-{obj_id}"
    
    os.makedirs(save_root, exist_ok=True)
    points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)
    gym.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, save_dir = save_root, save_name = "task_config")
    
    # gym.run_steps(1000)
    # import pdb; pdb.set_trace()
    gym.clean_up()
