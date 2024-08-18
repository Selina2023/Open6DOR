import glob
import json
import imageio.v2 as imageio
import os
import argparse
from evaluation import evaluator
import yaml

mesh_root = "meshes"
def load_task(task_path, bench_config):
    # task_config
    task_config = json.load(open(task_path, 'r'))
    
    # task_instruction
    task_instruction = task_config["instruction"]
    print("instruction:", task_instruction)

    # task_image
    if bench_config["image_mode"] == "GIVEN_IMAGE_ISAACGYM":
        image_path = task_path.replace("task_config.json", "before-rgb-0-0.png")
        task_image = imageio.imread(image_path)

    elif bench_config["image_mode"] == "GIVEN_IMAGE_BLENDER":
        pass
    
    elif bench_config["image_mode"] == "RENDER_IMAGE_ISAACGYM":
        from ..Method.interaction import init_gym
        gym, cfgs, task_config_now= init_gym(task_config, index=i, random_task=True, no_position = True)

        points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, \
            pixel2pointid, pointid2pixel = gym.refresh_observation(get_visual_obs=True)

        task_image = colors_envs[0]
    
    elif bench_config["image_mode"] == "RENDER_IMAGE_BLENDER":
        from renderer import open6dor_renderer
        task_image = None
        output_root_path = bench_config["output_path"]
        obj_paths = task_config["selected_urdfs"]
        obj_ids = [path.split("/")[-2] for path in obj_paths]

        init_poses = task_config["init_obj_pos"]
        obj_poses = {}

        for i in range(len(obj_ids)):
            pos = init_poses[i]
            id = obj_ids[i]
            position = pos[:3]
            quaternion = pos[3:7] 
            transformation_matrix = open6dor_renderer.create_transformation_matrix(position, quaternion)
            obj_poses[id] = transformation_matrix
        task_id = "my_test"
        script = generate_shell_script(output_root_path, task_id, obj_paths, init_poses, 
            bench_config["background_material_id"], bench_config["env_map_id"], 
            bench_config["cam_quaternion"], bench_config["cam_translation"])
        # run shell script
        os.system(f"bash {script}")
        
    return task_config, task_instruction, task_image

def generate_shell_script(output_root_path, task_id, obj_paths, init_poses,
                          background_material_id, env_map_id, cam_quaternion, cam_translation):
    script_name = "renderer/run_renderer.sh"
    command = "cd renderer\n"
    command += f"./blender-2.93.3-linux-x64/blender material_lib_v2.blend --background --python open6dor_renderer.py -- \\\n"
    command += f"    --output_root_path {output_root_path} \\\n"
    command += f"    --task_id {task_id} \\\n"
    command += f"    --obj_paths {' '.join(obj_paths)} \\\n"
    init_obj_pos_flat = ' '.join(map(str, [item for sublist in init_poses for item in sublist]))
    command += f"    --init_obj_pos {init_obj_pos_flat} \\\n"
    command += f"    --background_material_id {background_material_id} \\\n"
    command += f"    --env_map_id {env_map_id} \\\n"
    command += f"    --cam_quaternion {' '.join(map(str, cam_quaternion))} \\\n"
    command += f"    --cam_translation {' '.join(map(str, cam_translation))}\n"

    shell_file_content = f"#!/bin/bash\n\n{command}"

    with open(script_name, "w") as shell_file:
        shell_file.write(shell_file_content)

    print(f"Shell script {script_name} generated successfully.")
    print("=============================================")

    return script_name

def eval_task(cfgs, pred_pose, use_rot = False):
    if use_rot:
        pred_rot = pred_pose["rotation"]
        rot_gt = list(cfgs['anno_target']['annotation'].values())[0]["quat"]
        rot_deviation = evaluator.evaluate_rot(rot_gt, pred_rot)
        print(f"Rotation deviation: {rot_deviation} degrees")

    pos_bases = cfgs['init_obj_pos']
    pred_pos = pred_pose["position"]
    pos_eval = evaluator.evaluate_posi(pred_pos, pos_bases, "behind")

    return rot_deviation, pos_eval

def method_template(cfgs, task_instruction, task_image):
    pred_pose = {
        "position": [0,0,0],
        "rotation": [0,0,0,0]
    }
    return pred_pose

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Benchmarking script for task evaluation")
    parser.add_argument("--mode", type=str, choices=["load_test", "eval"], help="Path to the task configuration file")
    parser.add_argument("--task_data", type=str, default="6dof", help="path set or single path to the task configuration file")
    parser.add_argument("--image_mode", type=str, default="GIVEN_IMAGE_ISAACGYM", help="Image mode")
    parser.add_argument("--output_path", type=str, default="../output/test", help="Path to the output directory")

    _args = parser.parse_args()
    
    render_configs = yaml.load(open("bench_config.yaml", 'r'), Loader=yaml.FullLoader)
    import pdb; pdb.set_trace()
    # merge the two configs
    bench_config = {**_args.__dict__, **render_configs}
    if bench_config["task_data"] == "6dof":
        task_paths = glob.glob('tasks/6DoF/*/*/*/task_config_new2.json')
    elif bench_config["task_data"] == "position":
        task_paths = glob.glob('tasks/position/*/*/*/task_config_new2.json')
    elif bench_config["task_data"] == "rotation":
        task_paths = glob.glob('tasks/rotation/*/*/*/task_config_new2.json')
    else:
        task_paths = [bench_config["task_data"]]

    if bench_config["mode"] == "load_test":
        for task_path in task_paths:
            task_config, task_instruction, task_image = load_task(task_path, bench_config)

    elif bench_config["mode"] == "eval":
        USE_ROT = False if bench_config["task_data"] == "position" else True
        for task_path in task_paths:
            task_config = json.load(open(task_path, 'r'))
            task_config, task_instruction, task_image = load_task(task_path, bench_config)
            pred_pose = method_template(task_config, task_instruction, task_image)
            eval_task(task_config, pred_pose, use_rot = USE_ROT)
