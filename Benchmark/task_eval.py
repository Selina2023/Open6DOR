import glob
import json
import imageio


overall_paths = glob.glob('tasks/rot_banch_0704_overall/*/*/*/task_config.json')

pos_paths = glob.glob('tasks/rot_banch_0717_pos/*/*/*/task_config.json')

rot_paths = glob.glob('tasks/rot_banch_0717_rot/*/*/*/task_config.json')


def load_task(path, image_mode = "GIVEN_IMAGE_ISAACGYM"):
    # task_config
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
        pass
    elif image_mode == "RENDER_IMAGE_BLENDER":
        pass
    
    return task_image, task_instruction, task_config


def eval_task(pred_quat, gt_quat):
    pass

print(len(overall_paths), len(pos_paths), len(rot_paths))
for overall_path in overall_paths:
    task_image, task_instruction, task_config = load_task(overall_path)
    print(task_instruction)
    print(task_config)
    break