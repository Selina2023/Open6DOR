import json, imageio
from gym.utils import read_yaml_config, prepare_gsam_model
import numpy as np

class Open6DOR_GPT:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = cfgs["DEVICE"]
        self._prepare_ckpts()

    def _prepare_ckpts(self):
        # prepare gsam model
        if self.cfgs["INFERENCE_GSAM"]:
            self._grounded_dino_model, self._sam_predictor = prepare_gsam_model(device=self.device)

            self._box_threshold = 0.3
            self._text_threshold = 0.25
        else:
            self._grounded_dino_model, self._sam_predictor = None, None

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

def test_vlm():
    cfgs = read_yaml_config("config.yaml")
    open6dor_gpt = Open6DOR_GPT(cfgs=cfgs)
    prompt = "hello gpt, describe the image"
    image_path = "test_image.png"
    print("The ans is: ", open6dor_gpt.inference_vlm(prompt, image_path, print_ans=True))
    print("vlm test passed!")
    import pdb; pdb.set_trace()

def test_gsam():
    image_path = "test_image.png"
    cfgs = read_yaml_config("config.yaml")
    open6dor_gpt = Open6DOR_GPT(cfgs=cfgs)
    masks, _image = open6dor_gpt.inference_gsam(image_path = image_path, prompt="calculator")
    _image[masks[0][0].cpu().numpy().astype(bool)] = 0
    imageio.imwrite("test_mask.png", _image)
    print("The mask is saved as test_mask.png, check it!")
    import pdb; pdb.set_trace()

if __name__  == "__main__":
    # test_gsam()

    test_vlm()

    cfgs = read_yaml_config("config.yaml")
    task_cfgs_path = "/home/haoran/Projects/Rearrangement/Open6DOR/Method/tasks/6DoF/behind/Place_the_apple_behind_the_box_on_the_table.__upright/20240704-145831_no_interaction/task_config_new2.json"
    with open(task_cfgs_path, "r") as f: task_cfgs = json.load(f)
    
    open6dor_gpt = Open6DOR_GPT(cfgs=cfgs, task_cfgs=task_cfgs)
    