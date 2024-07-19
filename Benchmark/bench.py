import glob
import json
import imageio
import os
import argparse
from renderer import open6dor_renderer
from evaluation import evaluator

overall_paths = glob.glob('task_examples/overall/*/*/*/task_config.json')

pos_paths = glob.glob('task_examples/position/*/*/*/task_config.json')

rot_paths = glob.glob('task_examples/rotation/*/*/*/task_config.json')

mesh_root = "meshes"


    


def load_task(task_path, image_mode = "RENDER_IMAGE_BLENDER", output_path = "../output/test", cam_quaternion = [0, 0, 0.0, 1.0], cam_translation = [0.0, 0.0, 4], background_material_id = 44, env_map_id = 25):
    # task_config
    # task_path = "/Users/selina/Desktop/projects/Open6DOR/Benchmark/task_examples/overall/behind/Place_the_apple_behind_the_box_on_the_table.__upright/20240704-145831_no_interaction/task_config.json"#TODO 1: load config path from dataset using task_id
    task_config = json.load(open(task_path, 'r'))
    
    # task_instruction
    task_instruction = task_config["instruction"]
    print("instruction:", task_instruction)
    
    # task_image
    if image_mode == "GIVEN_IMAGE_ISAACGYM":
        image_path = task_path.replace("task_config.json", "before-rgb-0-0.png")
        task_image = imageio.imread(image_path)
      

    elif image_mode == "GIVEN_IMAGE_BLENDER":
        pass
    elif image_mode == "RENDER_IMAGE_ISAACGYM":
        pass
    elif image_mode == "RENDER_IMAGE_BLENDER":
        output_root_path = output_path
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
        script = generate_shell_script(output_root_path, task_id, obj_paths, init_poses, background_material_id, env_map_id, cam_quaternion, cam_translation)
        # run shell script
        os.system(f"bash {script}")



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

load_task("./task_examples/overall/behind/Place_the_apple_behind_the_box_on_the_table.__upright/20240704-145831_no_interaction/task_config.json")
def eval_task(task_id, pred_pose):
    pred_rot = [0,0,0,0]#TODO 2: extract rotation from pred_pose
    pred_pos = 0#TODO 3: extract position from pred_pose

    rot_gt = [0,0,0,0]#TODO 4: load ground truth rotation from annot
    pos_gt = []#TODO 5: load ground truth position from annot


    rot_deviation = evaluator.evaluate_rot(rot_gt, pred_rot)  #TODO 6: click into evaluate_rot

    print(f"Rotation deviation: {rot_deviation} degrees")

    pos_eval = -1#TODO 7: evaluate position


if __name__ == "__main__":
    # read args and call apis

    parser = argparse.ArgumentParser(description="Benchmarking script for task evaluation")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for load_task
    parser_load = subparsers.add_parser("load_task", help="Load a task")
    parser_load.add_argument("--task_path", type=str, required=True, help="Path to the task configuration file")
    parser_load.add_argument("--image_mode", type=str, default="GIVEN_IMAGE_ISAACGYM", help="Image mode")
    parser_load.add_argument("--output_path", type=str, default="../output/test", help="Path to the output directory")
    parser_load.add_argument("--cam_quaternion", type=float, nargs=4, default=[0, 0, 0.0, 1.0], help="Camera quaternion")
    parser_load.add_argument("--cam_translation", type=float, nargs=3, default=[0.0, 0.0, 4], help="Camera translation")
    parser_load.add_argument("--background_material_id", type=int, default=44, help="Background material ID")
    parser_load.add_argument("--env_map_id", type=int, default=25, help="Environment map ID")

    # Subparser for eval_task
    parser_eval = subparsers.add_parser("eval_task", help="Evaluate a task")
    parser_eval.add_argument("--task_id", type=str, required=True, help="Task ID")
    parser_eval.add_argument("--pred_pose", type=str, required=True, help="Predicted pose")

    args = parser.parse_args()

    if args.command == "load_task":
        load_task(args.task_path, args.image_mode, args.output_path, args.cam_quaternion, args.cam_translation, args.background_material_id, args.env_map_id)
    elif args.command == "eval_task":
        eval_task(args.task_id, args.pred_pose)
    else:
        parser.print_help()

# print(len(overall_paths), len(pos_paths), len(rot_paths))
# for overall_path in overall_paths:
#     task_image, task_instruction, task_config = load_task(overall_path)
#     print(task_instruction)
#     print(task_config)
#     break