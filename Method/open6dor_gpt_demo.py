import json, imageio
import argparse
# from gym.utils import read_yaml_config, prepare_gsam_model
import numpy as np, os
import yaml
import glob
import torch
import open3d as o3d
from position.vlm_utils import infer_path

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file into a Python dictionary
        config = yaml.safe_load(file)
    return config
class Open6DOR_GPT:
    def __init__(self, cfgs, mode):
        self.cfgs = cfgs
        self.device = cfgs["DEVICE"]
        self._prepare_ckpts()
        if mode == "position":
            self.cam_vinv = self.cfgs["cam"]["vinv"]
            self.cam_proj = self.cfgs["cam"]["proj"]
            pass
            # position_pred = self._inference_position_engine()
        elif mode == "rotation":
            self._prepare_rotation_engine()
        elif mode == "6dof":
            # self._inference_position_engine()
            self._prepare_rotation_engine()

    def inference_position_engine(self, image_path, depth_path, instruction):
        
        # get point cloud
        depth = np.load(depth_path)
        rgb = imageio.imread(image_path)
        cam_h, cam_w = rgb.shape[:2]
        from position.utils import get_point_cloud_from_rgbd_GPU
        pointclouds = get_point_cloud_from_rgbd_GPU(torch.tensor(depth), torch.tensor(rgb),None,self.cam_vinv, self.cam_proj, cam_w, cam_h)
        pc_xyz = pointclouds[...,:3].cpu().numpy()
        pc_colors = pointclouds[...,3:].cpu().numpy()
        
        if self.cfgs["VISUALIZE"]:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pc_xyz)

            point_cloud.colors = o3d.utility.Vector3dVector(pc_colors/255)

            o3d.visualization.draw_geometries([point_cloud])
        prompt = f"Now, I'd like to guide a robot to perform a specific task: {instruction}. \
At this stage you don't need to think about rotation, just 3d position. All the objects are placed on the table. \
You have access to a precise object detection model capable of determining the exact position \
of objects by name. Please specify the object you'd like to identify using this API. Once you \
provide the object name, we will retrieve its position, enabling you to make informed decisions. \
Kindly list the names of the objects you want to identify, separating them with ',', without \
including any additional responses or text. For example, if the instruction is 'Place the cup \
between the book and the pear,' you should output 'cup,book,pear' Please note that you should \
only provide the object names, separated by ',', and refrain from including any other information or responses."
        
        response = infer_path(prompt, image_path)
        print(response.json()['choices'][0]['message']['content'])

        ### get the object bbox information with gsam
        objs = response.json()['choices'][0]['message']['content'].split(",")
        position_info = "The position information is as follow. The axis-aligned bounding box information \
is given by two diagonal corner points"
        
        print(objs)
        for obj in objs:
            masks, bbox_axis_aligned_envs, bbox_center_envs = self._inference_gsam(
                rgb, pc_xyz, pc_colors, 
                text_prompt = obj, save_dir = os.path.dirname(image_path), save_name = "gsam")
            
            
            if bbox_center_envs is None:
                position_info += f"The position information of {obj} is unknown."
            else:
                bbox = list(bbox_axis_aligned_envs[0])
                bbox_center = bbox_center_envs[0]
                position_info += f"The position information of {obj} is: the axis-aligned bounding box of {obj} is {bbox},\
the bounding box of {obj} center is {bbox_center}."

            objs_str = ",".join(objs)
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
    response will be utilized in downstream API processes, so ensure in your response, the last line only contains \
    one string and three float values."

        ### gpt4v inference
        response = infer_path(prompt, image_path)
        print(response.json()['choices'][0]['message']['content'])
        try:
            infer_data = response.json()['choices'][0]['message']['content'].split(",")
            grasp_obj_name = infer_data[0]
            place_position = [float(p) for p in infer_data[1:]]
        except:
            response = infer_path(prompt, image_path)
            print(response.json()['choices'][0]['message']['content'])
            infer_data = response.json()['choices'][0]['message']['content'].split(",")
            grasp_obj_name = infer_data[0]
            place_position = [float(p) for p in infer_data[1:]]
            
        # TODO
        if self.cfgs["VISUALIZE"]:

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pc_xyz)

            point_cloud.colors = o3d.utility.Vector3dVector(pc_colors/255)

            point_target = o3d.geometry.PointCloud()
            point_target.points = o3d.utility.Vector3dVector([place_position])
            point_target.colors = o3d.utility.Vector3dVector([[0,1,0]])
            o3d.visualization.draw_geometries([point_cloud, point_target])

        print("#"*30)
        print("Instruction: ", instruction)
        print("Grasp object name: ", grasp_obj_name)
        print("Place position: ", place_position)
        print("#"*30)
        return grasp_obj_name, place_position

    def _prepare_ckpts(self):
        # prepare gsam model
        if self.cfgs["INFERENCE_GSAM"]:
            self._grounded_dino_model, self._sam_predictor = self._prepare_groundedsam()

            self._box_threshold = 0.3
            self._text_threshold = 0.25
        else:
            self._grounded_dino_model, self._sam_predictor = None, None

    def _prepare_groundedsam(self):
        
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        sam_version = "vit_h"
        sam_checkpoint = "../assets/ckpts/sam_vit_h_4b8939.pth"
        grounded_checkpoint = "../assets/ckpts/groundingdino_swint_ogc.pth"
        config = "vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        from vision.grounded_sam_demo import prepare_GroundedSAM_for_inference
        grounded_dino_model, sam_predictor = prepare_GroundedSAM_for_inference(sam_version=sam_version, sam_checkpoint=sam_checkpoint,
                grounded_checkpoint=grounded_checkpoint, config=config, device=self.device)
        return grounded_dino_model, sam_predictor

    def _inference_gsam(self, rgb_img, points, colors, text_prompt, save_dir, save_name = "gsam"):
        
        bbox_axis_aligned_envs = []
        bbox_center_envs = []
        
        assert self.cfgs["INFERENCE_GSAM"]
        from vision.grounded_sam_demo import prepare_GroundedSAM_for_inference, inference_one_image
        masks = inference_one_image(rgb_img[..., :3], self._grounded_dino_model, self._sam_predictor, box_threshold=self.box_threshold, text_threshold=self.text_threshold, text_prompt=text_prompt, device=self.device)

        if masks is None:
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


    ######################### TODO BELOW #########################

    def segment_image(self, image_path, output_folder, instruction):
        #TODO
        segment_prompt = self.cfgs["segment_prompt"]
        target_name = self.inference_vlm(segment_prompt, image_path)
        masks, _image = self.inference_gsam(image_path = image_path, prompt=target_name)
        rounds = 0
        while masks is None and rounds < 3:
            masks, _image = self.inference_gsam(image_path = image_path, prompt=target_name)
            rounds += 1
        if masks is None:
            return -1
        _image[~masks[0][0].cpu().numpy().astype(bool)] = 255
        # find the bounding box of the object:
        leftmost = np.min(np.where(masks[0][0].cpu().numpy().astype(bool))[1])
        upmost = np.min(np.where(masks[0][0].cpu().numpy().astype(bool))[0])
        rightmost = np.max(np.where(masks[0][0].cpu().numpy().astype(bool))[1])
        bottommost = np.max(np.where(masks[0][0].cpu().numpy().astype(bool))[0])
        # crop the image according to the bounding box:
        _image = _image[upmost:bottommost, leftmost:rightmost]
        # save the cropped image:
        cropped_image_path = os.path.join(output_folder, "cropped.png")
        imageio.imwrite(cropped_image_path, _image)
        return cropped_image_path

    def _prepare_rotation_engine(self):
        from rotation.rotation_engine import RotationEngine
        # TODO: add gt / reconstruction, prompts in cfgs
        self.rotation_engine = RotationEngine(self.cfgs) 



    def inference_rotation_engine(self, input_folder, output_folder, cfgs, visualize=False):
        
        # segment target object
        image_file = os.path.join(input_folder, "color.png")
        instruction = os.path.join(input_folder, "instruction.txt")
        cropped_image = self.segment_image(image_path=image_file, output_folder=output_folder, instruction=instruction)

        self.rotation_engine.get_final_rotation(
            instruction,
            input_image=cropped_image,
            output_folder=output_folder,
            cfgs=cfgs
        )
        if visualize:
            pass

        return #proposed_rotation, proposed_rotation_ind

    def inference_vlm(self, prompt, image_path, print_ans = False):
        from gym.vlm_utils import infer_path
        # prepare vlm model
        response = infer_path(prompt, image_path)
        while 'choices' not in response.json():
            response = infer_path(prompt, image_path)
        ans = response.json()['choices'][0]['message']['content']
        if print_ans:
            print(ans)
        return ans


    def inference_gsam(self, image: np.ndarray = None, image_path: str = None, prompt = None):
        from vision.grounded_sam_demo import prepare_GroundedSAM_for_inference, inference_one_image
        if image is not None:
            masks = inference_one_image(image[..., :3], self._grounded_dino_model, self._sam_predictor, box_threshold=self._box_threshold, text_threshold=self._text_threshold, text_prompt=prompt, device=self.device)
        elif image_path is not None:
            image = imageio.imread(image_path)
            masks = inference_one_image(image[..., :3], self._grounded_dino_model, self._sam_predictor, box_threshold=self._box_threshold, text_threshold=self._text_threshold, text_prompt=prompt, device=self.device)
        return masks, image

    def inference_task(self, task_cfgs):
        # prepare task data
        task_data = self.prepare_task_data(task_cfgs)

        # inference
        pred_pose = self.inference(task_data, self._grounded_dino_model, self._sam_predictor)

        return pred_pose
    
    def run_rotation_exp(self, task_folder, output_folder, image_path=None, mode=None):
        task_cfgs_path = os.path.join(task_folder, "task_config_new3.json")
        with open(task_cfgs_path, "r") as f: task_cfgs = json.load(f)   
        object_name = task_cfgs["target_obj_name"]
        object_code = task_cfgs["target_obj_code"]
        task_name = task_folder.split("/")[-1]

        instruction = task_cfgs["rotation_instruction"][68:]
        init_rotation = task_cfgs["init_obj_pos"][-1][3:7]
        if image_path is None:

            image_folder = os.path.join(output_folder, task_name)
            image_path = os.path.join(image_folder, "cropped.png")
        import time
        start_t = time.time()
        # proposed_rotation, proposed_rotation_ind = 
        if len(object_code) < 6:
            return
        self.inference_rotation_engine(task_name, instruction, init_rotation, image_path, output_folder, object_name, object_code, mode=mode)
        end_t = time.time()
        time_cost = end_t - start_t
        record_file = os.path.join(output_folder, "record.txt")
        with open(record_file, "a") as f:
            f.write(f"{task_name}:{time_cost},{mode}\n")

            # f.write(task_name + ":" + str(time_cost) + "\n")

        print("rotation experiment done!")


def test_vlm():
    cfgs = read_yaml_config("config.yaml")

    image_path = "test_image.png"
    image = imageio.imread(image_path)
    open6dor_gpt = Open6DOR_GPT(cfgs=cfgs)
    prompt = "hello gpt, describe the image"
    print("The ans is: ", open6dor_gpt.inference_vlm(prompt, image_path, print_ans=True))
    print("vlm test passed!")
    import pdb; pdb.set_trace()

def test_gsam():
    image_path = "test_image.png"

    image = imageio.imread(image_path)
    cfgs = read_yaml_config("config.yaml")
    open6dor_gpt = Open6DOR_GPT(cfgs=cfgs)
    masks, _image = open6dor_gpt.inference_gsam(image_path = image_path, prompt="calculator")

    _image[~masks[0][0].cpu().numpy().astype(bool)] = 0
    # find the leftemost, upmost, rightmost, bottommost pixel in masks[0][0] with the value of True:
    leftmost = np.min(np.where(masks[0][0].cpu().numpy().astype(bool))[1])
    upmost = np.min(np.where(masks[0][0].cpu().numpy().astype(bool))[0])
    rightmost = np.max(np.where(masks[0][0].cpu().numpy().astype(bool))[1])
    bottommost = np.max(np.where(masks[0][0].cpu().numpy().astype(bool))[0])

    # crop the image according to the bounding box
    _image = _image[upmost:bottommost, leftmost:rightmost]

    # _image[masks[0][0].cpu().numpy().astype(bool)] = 0
    imageio.imwrite("test_mask.png", _image)

    print("The mask is saved as test_mask.png, check it!")
    import pdb; pdb.set_trace()

def test_rotation_engine():
    cfgs = read_yaml_config("config.yaml")
    open6dor_gpt = Open6DOR_GPT(cfgs=cfgs)
    # instruction = "The bottle is upright on the table."
    task_folder = "../Benchmark/dataset/tasks/task0824_rot/left/Place_the_apple_to_the_left_of_the_calipers_on_the_table.__upright/20240824-161453_no_interaction"
    image_path = os.path.join(task_folder, "color.png")
    task_cfgs_path = os.path.join(task_folder, "task_config.json")
    # object_name = "test_engine_trans"
    output_folder = "./output/0826"
    # task_cfgs_path = "./reconstruction/examples/Place_the_glasses_at_the_center_of_all_the_objects_on_the_table.__lower_rim/20240704-151909_no_interaction/task_config_new.json"
    with open(task_cfgs_path, "r") as f: task_cfgs = json.load(f)   
    import pdb; pdb.set_trace() 
    object_name = task_cfgs["target_obj_name"]
    task_name = task_folder.split("/")[-1]
    if os.path.exists(os.path.join(output_folder, task_name)):
        task_name = task_name + object_name
    instruction = task_cfgs["instruction"]
    init_rotation = task_cfgs["init_obj_pos"][-1][3:7]
    
    open6dor_gpt.inference_rotation_engine(task_name, instruction, init_rotation, image_path, output_folder)

    print("rotation engine test passed!")

if __name__  == "__main__":
    ## parse arguments
    parser = argparse.ArgumentParser()
    # choose one mode from ["6dof", "position", "rotation"]
    parser.add_argument("--mode", type=str, default="position")
    # specify the folder containing inputs of the task
    parser.add_argument("--task_dir", type=str, default="task_refine_6dof_example")
    
    args = parser.parse_args()
    mode = args.mode
    task_dir = args.task_dir
    
    method_cfgs = read_yaml_config("./method_cfg.yaml")
    open6dor_engine = Open6DOR_GPT(method_cfgs, mode)

    # start running
    task_cfg_paths = glob.glob(f"../assets/tasks/{task_dir}/*/*/task_config_new5.json")
    for task_cfg_path in task_cfg_paths:
        ## input information
        task_root = os.path.dirname(task_cfg_path)
        rgb_file = os.path.join(task_root, "isaac_render-rgb-0-0.png")
        depth_file = os.path.join(task_root, "isaac_render-depth-0-0.npy")
        task_cfgs = json.load(open(task_cfg_path))
        position_instruction = task_cfgs["position_instruction"]
        rotation_instruction = task_cfgs["rotation_instruction"]

        # ensure all inputs are available
        assert os.path.exists(rgb_file), f"rgb file {rgb_file} does not exist"
        # assert os.path.exists(depth_file), f"depth file {depth_file} does not exist"
        assert position_instruction != "" and rotation_instruction != "", f"instruction file does not exist"

        grasp_obj_name, place_position = open6dor_engine.inference_position_engine(rgb_file,depth_file, position_instruction)
        

        ############### TODO BELOW ################
        # rotation