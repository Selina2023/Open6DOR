import cv2
import numpy as np
import os, glob
import argparse
import imageio
from PIL import Image
from isaacgym.torch_utils import *
import torch
import math
import yaml

def images_to_video(image_folder, video_path, frame_size=(1920, 1080), fps=30):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")])

    if not images:
        print("No images found in the specified directory!")
        return
    
    writer = imageio.get_writer(video_path, fps=fps)
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = imageio.imread(img_path)

        if img.shape[1] > frame_size[0] or img.shape[0] > frame_size[1]:
            print("Warning: frame size is smaller than the one of the images.")
            print("Images will be resized to match frame size.")
            img = np.array(Image.fromarray(img).resize(frame_size))
        
        writer.append_data(img)
    
    writer.close()
    print("Video created successfully!")
    
def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file into a Python dictionary
        config = yaml.safe_load(file)
    return config