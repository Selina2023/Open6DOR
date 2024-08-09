import glob
import json
from gym.utils import read_yaml_config, prepare_gsam_model

class Open6DOR_GPT:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = cfgs["DEVICE"]
        
    def prepare_ckpts(self):
        # prepare sam model
        if self.cfgs["INFERENCE_GSAM"]:
            grounded_dino_model, sam_predictor = prepare_gsam_model(device=self.device)
        else:
            grounded_dino_model, sam_predictor = None, None
        
    def inference_task(self, task_cfgs):
        # prepare task data
        task_data = self.prepare_task_data(task_cfgs)
        
        # inference
        pred_pose = self.inference(task_data, grounded_dino_model, sam_predictor)
        
        return pred_pose
  
  
 
if __name__  == "__main__":
    cfgs = read_yaml_config("config.yaml")
    task_cfgs_path = "/home/haoran/Projects/Rearrangement/Open6DOR/Method/tasks/6DoF/behind/Place_the_apple_behind_the_box_on_the_table.__upright/20240704-145831_no_interaction/task_config_new2.json"
    with open(task_cfgs_path, "r") as f: task_cfgs = json.load(f)
    
    open6dor_gpt = Open6DOR_GPT(cfgs=cfgs, task_cfgs=task_cfgs)