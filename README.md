![](https://img.shields.io/badge/release-TODO-brightgreen.svg?style=popout-square)
![](https://img.shields.io/badge/CARLA-0.9.6+-blue.svg?style=popout-square)
![](https://img.shields.io/badge/python-3.6%20|%203.7%20|3.8-blue.svg?style=popout-square)
![](https://img.shields.io/badge/license-MIT-orange.svg?style=popout-square)

CARLA real traffic scenarios
========================

![I80 demo](I80_demo.gif)

### TL;DR

- Custom CARLA maps that mimic real-world roads and human driver behaviors from NGSim dataset (I80 and US101)
- We've trained and benchmarked policies on real-world lane change maneuvers from NGSim dataset
- We provide the source code for running NGSim-based scenarios in CARLA. Scenario interface is similar to [openai gym](https://gym.openai.com/) interface

### Prerequisites
1. Download and extract CARLA ([0.9.6 download link](https://github.com/carla-simulator/carla/releases/tag/0.9.6)). Then, add PythonAPI wheel to your `PYTHONPATH`:
    ```bash
    export CARLA_ROOT=/path/to/your/carla/release-folder
    export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.6-linux-x86_64.egg:$PYTHONPATH
    ```
2. You also need to install our asset packages with two new CARLA maps
    - Download archives - **TODO LINK**
    - Copy archives to `$CARLA_ROOT/Import`
    - Import `cd Import && ./ImportAssets.sh`

### Quickstart

##### Terminal I
```bash
./CarlaUE4.sh -benchmark -fps=10
```

##### Terminal II
```bash
# (wait until server loads)
python example/example_roundabout_scenario_usage.py
```
##### Terminal III
```bash
# (wait until scenario script connects successfully, map rendering may tak a while)
python example/manual_driving.py --res 640x480
```
Code tested with CARLA 0.9.6.

### Real-traffic scenarios
* `python example/example_replay_ngsim_in_carla.py` - shows how to run scenario in training loop.
* `python example/example_replay_ngsim_in_carla.py` - shows how to replay NGSim dataset in CARLA. It was used to generated GIF in this README file.

### Credits

Code for interfacing with NGSim dataset was based on https://github.com/Atcold/pytorch-PPUU