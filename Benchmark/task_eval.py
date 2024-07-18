import glob
import json
import imageio
import os
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
    script_name = "run_renderer.sh"
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

# print(len(overall_paths), len(pos_paths), len(rot_paths))
# for overall_path in overall_paths:
#     task_image, task_instruction, task_config = load_task(overall_path)
#     print(task_instruction)
#     print(task_config)
#     break