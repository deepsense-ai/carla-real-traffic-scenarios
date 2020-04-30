CARLA real traffic scenarios
========================

![I80 demo](I80_demo.gif)

# TL:DR

- We prepared CARLA maps that mimic real-world roads from NGSim dataset (I80 and US101),
- we train and benchmark policies on real-world lane-change manuveurs from NGSim dataset,
- we share code for running NGSim-based scenarios in CARLA. 

# How to run example?

You need to add CARLA python client whl to your pythonpath:
```
export CARLA_ROOT=/path/to/your/carla/release
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.6-linux-x86_64.egg:$PYTHONPATH
```

Code is tested only against CARLA 0.9.6.

## Credits

* Code for interfacing with NGSim dataset based on https://github.com/Atcold/pytorch-PPUU