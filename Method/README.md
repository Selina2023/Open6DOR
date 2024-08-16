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

