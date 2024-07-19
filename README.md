# Open6DOR
![teaser](./images/teaser_final1.pdf)
This is the official repository of [Open6DOR: Benchmarking Open-instruction 6-DoF Object Rearrangement and A VLM-based Approach](https://pku-epic.github.io/Open6DOR/). In this paper, we propel the pioneer construction of the benchmark and approach for table-top Open-instruction 6-DoF Object Rearrangement (Open6DOR). Specifically, we collect a synthetic dataset of 200+ objects and carefully design 2400+ Open6DOR tasks. These tasks are divided into the Position-track, Rotation-track, and 6-DoF-track for evaluating different embodied agents in predicting the positions and rotations of target objects. Besides, we also propose a VLM-based approach for Open6DOR, named Open6DOR-GPT, which empowers GPT-4V with 3D-awareness and simulation-assistance while exploiting its strengths in generalizability and instruction-following for this task. We compare the existing embodied agents with our Open6DOR-GPT on the proposed Open6DOR benchmark and find that Open6DOR-GPT achieves the state-of-the-art performance. We further show the impressive performance of Open6DOR-GPT in diverse real-world experiments.

# Benchmark
The Open6DOR Benchmark is specifically designed for table-top Open6DOR tasks within a simulation environment. Our dataset encompasses 200+ high-quality objects, forming diverse scenes and totaling 2400+ diverse tasks. All tasks are carefully configured and accompanied by detailed annotations. To ensure comprehensive evaluation, we provide three specialized tracks of benchmark: the Rotation-track Benchmark \(\mathcal{B}_r\), the Position-track benchmark \(\mathcal{B}_p\), and the 6-DoF-track Benchmark 
\(\mathcal{B}_{6DOR}\). 
In this repository, we provide:
- A dataset of diverse objects
- 2400+ Open6DOR tasks with detailed annotations
- A set of evaluation metrics for each track of tasks

## Installation
```
# Clone the repository
git clone git@github.com:Selina2023/Open6DOR.git
cd Open6DOR
# Create an environment
conda create -n Open6DOR python=3.9?
# Install dependencies
pip install -r requirements.txt
# Download datasets
- Download [Blender 2.93.3 (Linux x64)](https://download.blender.org/release/Blender2.93/blender-2.93.3-linux-x64.tar.xz) and uncompress.
- Download the [environment map asset](/envmap_lib.tar.gz) and uncompress.
- Download the [blend file](/material_lib_v2.blend).


```
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
│   └── texture
│       └── texture0.jpg
├── material_lib_v2.blend
├── modify_material.py
├── task_examples
│   ├── 6DoF
│   ├── position
│   └── rotation
└── bench.py



```

## Usage



```
python interaction.py --mode XXX
```

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
