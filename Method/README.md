# Method Introduction



## Get Task
A class to get a task through configuration file, which can be used to load simulation env in IsaacGym and get the task information, render and control robot, .etc

- _prepare_task: Load simulation env and get task information

- _init_gym: Initialize gym env

- _setup_scene: Set up scene

- prepare_franka_asset: from `self.cfgs["asset"]["franka_asset_file"]` to load franka asset

- _prepare_obj_assets: Load object assets: table, objects

- _load_env: load all assets to env and set up scene

- _init_observation: Initialize observation space and corresponding observation functions

- refresh_observation: get observation dict from env

- clean_up: clean up env

## Open6DOR-GPT

GroundedSAM:
```
cd Method/vision/GroundedSAM/GroundingDINO
pip install -e .
cd ../../../..
cd Method/vision/GroundedSAM/segment_anything
pip install -e .
cd ../../../..
```
Extensions:
```
sudo apt update
sudo apt install fonts-dejavu
```

if meet error:
```
cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub'
```
try:
```
pip install --upgrade huggingface_hub
```

SAM checkpoint is [here](https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth)

GroundingDINO checkpoint is [here](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

## Task Generation

The core code for task generation is in `Method/interaction.py`. The task generator is responsible for generating tasks for Open6DOR. 

#### Position Track
```bash
python interaction.py --mode gen_task --task_root debug_gen_task_pos
```


#### Rotation Track
```bash
python interaction.py --mode gen_task_pure_rot --task_root debug_gen_task_rot
```

#### 6DoF Track
```bash
python interaction.py --mode gen_task_rot --task_root debug_gen_task_6dof
```

#### Large Dataset Generation
If you want to generate a large dataset, you can use the following command:
```bash
python run_multiple.py --f "YOUR COMMAND" --n YOUR_RUN_TIMES
```

#### Change Parameters
You can change the parameters in `Method/interaction.py` to generate different tasks.

##### Object Number
```python
    if orientation == "center":
        selected_obj_num = np.random.randint(4, 5)
    elif orientation == "between":
        selected_obj_num = np.random.randint(3, 5)
    else:
        selected_obj_num = np.random.randint(2, 5)
```

##### Object Position
In config.yaml, you can change the object position range:
```yaml
assets:
    position_noise: [0.2, 0.25] # x and y position random range, depends on the table size
```

