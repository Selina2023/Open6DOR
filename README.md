# Open6DOR
![Teaser](./images/teaser_final1.jpg)
This is the official repository of [Open6DOR: Benchmarking Open-instruction 6-DoF Object Rearrangement and A VLM-based Approach](https://pku-epic.github.io/Open6DOR/). In this paper, we propel the pioneer construction of the benchmark and approach for table-top Open-instruction 6-DoF Object Rearrangement (Open6DOR). Specifically, we collect a synthetic dataset of 200+ objects and carefully design 2400+ Open6DOR tasks. These tasks are divided into the Position-track, Rotation-track, and 6-DoF-track for evaluating different embodied agents in predicting the positions and rotations of target objects. Besides, we also propose a VLM-based approach for Open6DOR, named Open6DOR-GPT, which empowers GPT-4V with 3D-awareness and simulation-assistance while exploiting its strengths in generalizability and instruction-following for this task. We compare the existing embodied agents with our Open6DOR-GPT on the proposed Open6DOR benchmark and find that Open6DOR-GPT achieves the state-of-the-art performance. We further show the impressive performance of Open6DOR-GPT in diverse real-world experiments.

## Benchmark
The Open6DOR Benchmark is specifically designed for table-top Open6DOR tasks within a simulation environment. Our dataset encompasses 200+ high-quality objects, forming diverse scenes and totaling 2400+ diverse tasks. All tasks are carefully configured and accompanied by detailed annotations. To ensure comprehensive evaluation, we provide three specialized tracks of benchmark: the Rotation-track Benchmark ($\mathcal{B}_r$), the Position-track benchmark ($\mathcal{B}_p$), and the 6-DoF-track Benchmark ($\mathcal{B}_{\text{6DOR}}$).
In this repository, we provide:
- A dataset of diverse objects
- 2400+ Open6DOR tasks with detailed annotations
- A set of evaluation metrics for each track of tasks

### Installation
**Environment Setup**
We recommend using Linux system for better compatability with our modules (including Blender and Isaacgym).
```
# Clone the repository
git clone git@github.com:Selina2023/Open6DOR.git
cd Open6DOR
# Create an environment
conda create -n Open6DOR python=3.9
# Install dependencies
pip install -r requirements.txt
```
***TODO: add dependencies***


**Dataset Downloads**
Refer to the next section for specific file locations.
- Download the [object datasets](https://drive.google.com/drive/folders/1Gm30OtQWRb5NitIdnLSJlfLdAG_rWHQX?usp=sharing) and uncompress.
- Download the ***TODO: [task datasets](???)*** and uncompress.

**Rendering Dependencies**
- Download [Blender 2.93.3 (Linux x64)](https://download.blender.org/release/Blender2.93/blender-2.93.3-linux-x64.tar.xz) and uncompress.
- Download the [environment map asset](/envmap_lib.tar.gz) and uncompress.
- Download the [blend file](/material_lib_v2.blend).
- Install the Python packages (Numpy, etc.) into the Blender built-in Python environment. 
```
cd Benchmark/renderer/blender-2.93.3-linux-x64/2.93/python/bin
./python3.9 -m ensurepip
./python3.9 -m pip install --upgrade pip --user
./python3.9 -m pip install numpy --user
```
**Simulator Dependencies**

**File Structure**
After downloading the datasets, organize the file structure as follows:

```
Benchmark
├── benchmark_catalogue                              
│   ├── annotation
│   │   └── ...
│   ├── category_dictionary.json
│   └── ...
├── dataset
│   ├── objects
│   │   ├── objaverse_rescale
│   │   └── ycb
│   └── tasks
│       ├── 6DoF_track
│       ├── position_track
│       └── rotation_track
├── evaluation
│   └── evaluator.py
├── renderer
│   ├── blender-2.93.3-linux-x64
│   ├── envmap_lib                                
│   │   ├── abandoned_factory_canteen_01_1k.hdr
│   │   └── ...
│   ├── texture
│   │   └── texture0.jpg
│   ├── material_lib_v2.blend
│   ├── modify_material.py
│   └── open6dor_renderer.py
├── task_examples
│   ├── 6DoF
│   ├── position
│   └── rotation
└── bench.py

```

### Usage
Along with the dataset, we provide several functions to enable visualization and evaluation of the tasks:
- To load a task example, run the following command (you may change the image_mode to GIVEN_IMAGE_ISAACGYM or others):
```
cd Benchmark
python bench.py load_task --task_path ./task_examples/6DoF/behind/Place_the_apple_behind_the_box_on_the_table.__upright/20240704-145831_no_interaction/task_config.json --image_mode RENDER_IMAGE_BLENDER --output_path ./output/test 

```
- To evaluate the task, run the following command:
#TODO: where to store the annotations & waht is the predicted pose format?

```
python bench.py eval_task --task_id my_test --pred_pose path/to/pred_pose.json

```

- Besides evaluating the numerical results of the pose prediction directly, we provide another set of metrics where users are allowed to control the robot arm and interact with the simulation environment. Such evaluation is soely based on the final pose of the target object after execution. To do this, run the following command:

```
python interaction.py --mode XXX
```

### Format Conventions
```
rotation matrix?
13-float list?
```

## Method
![Method](./images/overall_pipeline_final1.jpg)
By incorporating 3D awareness and simulation assistance, we effectively tackle the Open6DOR task through a decomposed approach. 
Specifically, Open6DOR-GPT takes the RGB-D image and instruction as input and outputs the corresponding robot motion trajectory. Firstly, the preprocessing module extracts the object names and masks. Then, the two modules simultaneously predict the position and rotation of the target object in a decoupled way. Finally, the planning module generates a trajectory for execution.

Code coming soon... (We are currently trying to update our method to attain better real-time performance)


```bash
cd vision/GroundedSAM/GroundingDINO
pip install -e .
cd ../segment_anything
pip install -e .
cd ../../..
```




## Troubleshooting

- requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /bert-base-uncased/resolve/main/tf_model.h5 (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f4769a3cc40>: Failed to establish a new connection: [Errno 101] Network is unreachable'))
    - Solution: Network error, In China, try global proxy.

## Contact
For further details or questions, please feel free to contact us:
- [Yufei Ding](https://selina2023.github.io/): selina@stu.pku.edu.cn
- [Haoran Geng](https://geng-haoran.github.io/): ghr@berkeley.edu
- [He Wang](https://hughw19.github.io/): hewang@pku.edu.cn