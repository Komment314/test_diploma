# Opt-informed-RRT*
Repository of thesis "Path Planning for High-DOF manipulator" and algorithm "Opt-informed-RRT*"

# Structure of repository

### 1. diploma 
Text of diploma in pdf format in russian language

### 2. basics 
Base classes and functions for running algorithms and benchmarking

### 3. algos 
Algorithms for single run and viewing

### 4. benchmarking 
Benchmarking set-up, its saved results and visualisation

# Installation

### Clone repository

```bash
git clone git@github.com:Komment314/opt-informed-rrt-star.git
```

### Install requirements

```bash
python install -r requirements.txt
```

### Install CoppeliaSim

Install CoppeliaSim edu from original site: https://coppeliarobotics.com

# Launching single runs

### Setting CoppeliaSim Scene

Start the CoppeliaSim and via `File -> Open scene...` load scene `UR10_rack_scene.ttt`, which is located in the folder `/scenes`

### Algorithms

To launch Informed-RRT* use [Example_default.ipynb](./algos/Example_default.ipynb)

To launch Opt-Informed-RRT* [Example_modified.ipynb](./algos/Example_modified.ipynb)

### Examples of runs



https://github.com/Komment314/test_diploma/blob/main/videos/informed-rrt-star-demo.mp4

https://github.com/Komment314/test_diploma/blob/main/videos/opt-informed-rrt-star-demo.mp4


### Examples of runs 2



https://github.com/Komment314/test_diploma/assets/71181179/d0379a3d-48d5-413c-94f3-06e245735d15



https://github.com/Komment314/test_diploma/assets/71181179/57743c24-6622-4641-a843-31d1f6316309




# Benchmarking

1. Open [pre_trees.ipynb](./benchmarking/path_finding_for_benckmarking/pre_trees.ipynb), set `n_paths_per_target`, remember this value and run. It will create `n_paths_per_target` pre-RRT* trees for each target. It will take a couple of hours to complete.

2. Open [bench.ipynb](./benchmarking/bench.ipynb), set `n_paths_per_target` to remembered value and set `n_tests_per_path` and remember it too for future visualization. You can set `current_target_indexes` with a list of index/-es of targets to run the tests one at a time, not all at once. Set `max_iters` and `max_time`.

3. If you dont want to save stats then just run the code. Otherwise uncomment rows with `np.save...` and run. Saved files will be saved to `benchmarking_stats/` folder so if you want to start over delete or move previous stats.

4. For results visualization use [vis.ipynb](./benchmarking/vis/vis.ipynb). Set values `n_targets`, `n_paths`, `max_iters`, `max_time` and `timestep` and run.
