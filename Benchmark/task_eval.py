import glob
import json
import imageio
from renderer import open6dor_renderer
from Open6DOR.Benchmark.evaluation import evaluator

overall_paths = glob.glob('task_examples/overall/*/*/*/task_config.json')

pos_paths = glob.glob('task_examples/position/*/*/*/task_config.json')

rot_paths = glob.glob('task_examples/rotation/*/*/*/task_config.json')

mesh_root = "meshes"


    


def load_task(task_id, image_mode = "GIVEN_IMAGE_ISAACGYM", output_path = "../output/test", cam_quaternion = [0, 0, 0.0, 1.0], cam_translation = [0.0, 0.0, 4], background_material_id = 44, env_map_id = 25):
    # task_config
    path = #TODO 1: load config path from dataset using task_id
    task_config = json.load(open(path, 'r'))
    
    # task_instruction
    task_instruction = task_config["instruction"]
    
    # task_image
    if image_mode == "GIVEN_IMAGE_ISAACGYM":
        image_path = path.replace("task_config.json", "before-rgb-0-0.png")
        task_image = imageio.imread(image_path)
    elif image_mode == "GIVEN_IMAGE_BLENDER":
        pass
    elif image_mode == "RENDER_IMAGE_ISAACGYM":
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

        open6dor_renderer.rendering(output_root_path, task_id, mesh_root, obj_ids, obj_poses, background_material_id, env_map_id, cam_quaternion, cam_translation)
    elif image_mode == "RENDER_IMAGE_BLENDER":
        pass
    
    return task_image, task_instruction, task_config


def eval_task(task_id, pred_pose):
    pred_rot = [0,0,0,0]1#TODO 2: extract rotation from pred_pose
    pred_pos = 0#TODO 3: extract position from pred_pose

    rot_gt = [0,0,0,0]1#TODO 4: load ground truth rotation from annot
    pos_gt = []1#TODO 5: load ground truth position from annot


    rot_deviation = evaluator.evaluate_rot(rot_gt, pred_rot)

    print(f"Rotation deviation: {rot_deviation} degrees")

    pos_eval = #TODO 6: evaluate position

print(len(overall_paths), len(pos_paths), len(rot_paths))
for overall_path in overall_paths:
    task_image, task_instruction, task_config = load_task(overall_path)
    print(task_instruction)
    print(task_config)
    break