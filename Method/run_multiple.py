import sys, os
import argparse

# add args
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
#parser.add_argument('--f', type=str, default="python reconstruction/mesh_reconstruction.py")
parser.add_argument('--f', type=str, default="python interaction.py --mode gen_task_rot --task_root rot_banch_0704 ")
#parser.add_argument('--f', type=str, default="python overall_clip.py")



args = parser.parse_args()

for i in range(args.n):
    os.system(args.f)
    