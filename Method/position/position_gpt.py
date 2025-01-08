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
import sys,os
# parent of current
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from position.utils import images_to_video, quat_axis, orientation_error, cube_grasping_yaw, read_yaml_config, \
    get_downsampled_pc, get_point_cloud_from_rgbd, get_point_cloud_from_rgbd_GPU
import os, json
import yaml
from scipy.spatial.transform import Rotation as R
import sys
import trimesh
import plotly.graph_objects as go
import os
from vision.grounded_sam_demo import prepare_GroundedSAM_for_inference, inference_one_image
from position.vlm_utils import infer_path

class PositionEngine():
    def __init__(
        self,
        cfgs,
        device = "cuda",
    ):
        self.device = device
        self.cfgs = cfgs

        self._prepare_groundedsam()
        
    def _prepare_groundedsam(self, 
            sam_checkpoint_path="../assets/ckpts/sam_vit_h_4b8939.pth", 
            grounded_checkpoint_path="../assets/ckpts/groundingdino_swint_ogc.pth", 
            config_path="vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            box_threshold=0.3, 
            text_threshold=0.25,
            sam_version = "vit_h",
        ):
        """
        sam_version = "vit_h"
        sam_checkpoint = "../assets/ckpts/sam_vit_h_4b8939.pth"
        grounded_checkpoint = "../assets/ckpts/groundingdino_swint_ogc.pth"
        config = "../vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        """

        self.grounded_dino_model, self.sam_predictor = prepare_GroundedSAM_for_inference(
            sam_version=sam_version, sam_checkpoint=sam_checkpoint_path,
            grounded_checkpoint=grounded_checkpoint_path, config=config_path, device=self.device)

    def processing_data(self, image, depth):
        depth_tentor = torch.from_numpy(depth).to(self.device)
        image_tensor = torch.from_numpy(image).to(self.device)
        cam_w, cam_h = image_tensor.shape[1], image_tensor.shape[0]
        pointclouds = get_point_cloud_from_rgbd_GPU(
            depth_tentor,
            image_tensor,
            None,
            self.cfgs["cam"]["vinv"], 
            self.cfgs["cam"]["proj"], 
            cam_w, cam_h
        )
        points = pointclouds[:, :3].cpu().numpy()
        colors = pointclouds[:, 3:6].cpu().numpy()
        i_indices, j_indices = np.meshgrid(np.arange(cam_w), np.arange(cam_h), 
                indexing='ij')
        pointid2pixel = np.stack((i_indices, j_indices), axis=-1).reshape(-1, 2)
        pixel2pointid = np.arange(cam_w * cam_h).reshape(cam_w, cam_h)
        pointid2pixel = None
        pixel2pointid = None
        # save points
        if False:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors/255.0)
            o3d.io.write_point_cloud("point.ply", point_cloud)
        # points_env.append(points)
        # colors_env.append(colors)
        return points, colors, image, depth, pointid2pixel, pixel2pointid
                    
    def inference_gsam(self, rgb_img, points, colors, text_prompt, save_dir, save_name = "gsam"):
        bbox_axis_aligned_envs = []
        bbox_center_envs = []
        
        masks = inference_one_image(
            rgb_img[..., :3], self.grounded_dino_model, 
            self.sam_predictor, 
            box_threshold=0.3, 
            text_threshold=0.25, 
            text_prompt=text_prompt, 
            device=self.device)

        if masks is None:
            # import pdb; pdb.set_trace()
            return None, None, None
        if True:
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
            
            if True:
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
      
    def run_position_gpt_inference(self, task_root):
        image_path = imageio.imread(task_root + "/before-rgb-0-1.png")
        depth_path = np.load(task_root + "/before-depth-0-1.npy")
        task_cfg = json.load(open(task_root + "/task_config.json"))
        instruction = task_cfg["instruction"]
        # import pdb; pdb.set_trace()
        points, colors, image, depth, pointid2pixel, pixel2pointid = self.processing_data(image_path, depth_path)
        
        # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)
        ### save rendered data
        # self.save_render(rgb_envs=rgb_envs[0], depth_envs=depth_envs[0], ori_points_env=ori_points_envs[0], 
        #                 ori_colors_env=ori_colors_envs[0], points=points_envs[0], colors=colors_envs[0], 
        #                 save_dir = save_root, save_name = "before_plc")
        # self.save_render(rgb_envs=rgb_envs, depth_envs=None, ori_points_env=None, ori_colors_env=None, points=None, colors=None, 
        #                 save_dir = save_root, save_name = "before_plc")


        ### save to a tmp path for gpt4v inference
        tmp_img_path = task_root + "/rgb_tmp.png"
        imageio.imwrite(tmp_img_path, image)
        
        if False:
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
                    image, points, colors, 
                    text_prompt = obj, save_dir = task_root, save_name = "gsam")
                
                
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
            
        masks, bbox_axis_aligned_envs, bbox_center_envs = self.inference_gsam(image, points, colors, text_prompt = grasp_obj_name, save_dir = task_root, save_name = "gsam")
        
        if False:
            mask = masks[0][0].cpu().numpy()
            image_masked = image.copy()
            image_masked[~mask] = 255
            mask_pixels = np.where(mask)
            mask_bbox = np.array([[mask_pixels[0].min() - 50, mask_pixels[1].min()- 50], [mask_pixels[0].max()+50, mask_pixels[1].max()+50]])
            mask_bbox[:,0] = np.clip(mask_bbox[:,0], 0, 1439)
            mask_bbox[:,1] = np.clip(mask_bbox[:,1], 0, 2159)
            cropped_image = image_masked[mask_bbox[0,0]:mask_bbox[1,0], mask_bbox[0,1]:mask_bbox[1,1]]
            
            # save
            fname = os.path.join(task_root, "gsam-mask.png")
            imageio.imwrite(fname, image_masked)
            fname = os.path.join(task_root, "gsam-mask-crop.png")
            imageio.imwrite(fname, cropped_image)
            
            # self.inference_sudo_ai(fname)
        
        if False:
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

if __name__ == "__main__":
    cfgs = read_yaml_config("/home/ghr/Projects/Rearrangement/Open6DOR_internal/Method/method_cfg.yaml")
    position_gpt = PositionEngine(cfgs=cfgs)
    # position_gpt.prepare_groundedsam()
    image_path = "/home/ghr/Projects/Rearrangement/Open6DOR_internal/Method/output/pos_tasks_1018/behind/Place_the_hammer_behind_the_hard_drive_on_the_table._/20241018-130717_no_interaction/before-rgb-0-1.png"
    depth_path = "/home/ghr/Projects/Rearrangement/Open6DOR_internal/Method/output/pos_tasks_1018/behind/Place_the_hammer_behind_the_hard_drive_on_the_table._/20241018-130717_no_interaction/before-depth-0-1.npy"
    image = imageio.imread(image_path)
    depth = np.load(depth_path)
    position_gpt.run_position_gpt_inference("/home/ghr/Projects/Rearrangement/Open6DOR_internal/Method/output/pos_tasks_1018/behind/Place_the_hammer_behind_the_hard_drive_on_the_table._/20241018-130717_no_interaction")
    