CARLA real traffic scenarios
========================

![I80 demo](I80_demo.gif)

### TL:DR

- Custom CARLA maps that mimic real-world roads from NGSim dataset (I80 and US101)
- We've trained and benchmarked policies on real-world lane change maneuvers from NGSim dataset
- We provide code for running NGSim-based scenarios in CARLA. Scenario interface is similar to [openai gym](https://gym.openai.com/) interface

### How to run examples?

1) You need to add CARLA python client wheel to your `PYTHONPATH`
```
export CARLA_ROOT=/path/to/your/carla/release
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.6-linux-x86_64.egg:$PYTHONPATH
```
2) You need to install our asset packs with two new CARLA maps - DOWNLOAD_LINK_TODO

### Examples:

* `example/example_replay_ngsim_in_carla.py` - shows how to run scenario in training loop.
* `example/example_replay_ngsim_in_carla.py` - shows how to replay NGSim dataset in CARLA. It was used to generated GIF in this README file.
* `example/example_roundabout_scenario_usage.py` - Town03 with artificial CARLA traffic (actors driving on autopilot)

Code was tested on CARLA 0.9.6.

### Credits

* Code for interfacing with NGSim dataset based on https://github.com/Atcold/pytorch-PPUU