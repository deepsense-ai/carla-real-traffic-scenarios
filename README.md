![](https://img.shields.io/badge/release-1.0.0-brightgreen.svg?style=popout-square)
![](https://img.shields.io/badge/CARLA-0.9.6+-blue.svg?style=popout-square)
![](https://img.shields.io/badge/python-3.6+-blue.svg?style=popout-square)
![](https://img.shields.io/badge/license-MIT-orange.svg?style=popout-square)

CARLA real traffic scenarios
========================
![readme-main-cropped](https://user-images.githubusercontent.com/64484917/90624607-7f3cb980-e218-11ea-8877-30c599e76f4f.gif)

NGSIM freeway             |  openDD roundabout
:-------------------------:|:-------------------------:
![readme-ngsim](https://user-images.githubusercontent.com/64484917/90623562-f07b6d00-e216-11ea-872f-a00000b75c14.gif) | ![readme-opendd](https://user-images.githubusercontent.com/64484917/90628518-0cced800-e21e-11ea-96fe-a30e3762ab1a.gif)

### Features

1. Handmade maps for [CARLA](https://carla.org/) that mimic real-world roads 
    - 7 roundabouts (https://arxiv.org/abs/2007.08463)
    - 2 freeways ([US101](https://www.fhwa.dot.gov/publications/research/operations/07030/index.cfm), [I-80](https://www.fhwa.dot.gov/publications/research/operations/06137/))
2. Code that transfers real-world traffic from datasets into CARLA
3. Scenario API similar to [OpenAI Gym](https://gym.openai.com/)

We've also trained and benchmarked policies on lane change maneuvers and roundabout navigation.
More details, our article and videos of trained policies are available on [our website](https://sites.google.com/view/carla-real-traffic-scenarios/home).

### Prerequisites

`pip install -r requirements.txt`

#### CARLA
Download and extract CARLA ([0.9.9.4 download link](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.4.tar.gz)). Then, add PythonAPI wheel to your `PYTHONPATH`:
```bash
export CARLA_ROOT=/path/to/your/carla/release-folder
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg:$PYTHONPATH
```
#### New maps
Install our CARLA package with new maps: 7 roundabout, 2 freeways
    - Download our archive: [Google Drive download link](https://drive.google.com/file/d/1FCHL7YJk12AwfxuMPmwXPJj71n3mwSxE/view?usp=sharing)
    - Move the archive to: `$CARLA_ROOT/Import`
    - Ingest into CARLA release: `cd Import && ./ImportAssets.sh`
    
If working on remote servers with no desktop, simply use [gdown](https://pypi.org/project/gdown/).
#### Datasets
1. Download: [NGSIM](http://bit.ly/PPUU-data), [opendDD](https://drive.google.com/file/d/12laSzLCaJQa-09sXOnbwaR2INzHurVq1/view?usp=sharing)
2. Unpack NGSIM archive: `tar xf xy-trajectories.tgz`
3. Unpack openDD archive 

### Quickstart

##### Terminal I
```bash
cd $CARLA_ROOT
./CarlaUE4.sh -benchmark -fps=10
```

##### Terminal II
Wait until server boots up
```bash
python example/example_roundabout_scenario_usage.py
```
##### Terminal III
```bash
# (wait until scenario script connects successfully, map rendering may tak a while)
python example/manual_driving.py --res 900x500
```
Code tested with CARLA 0.9.6.

### Real-traffic scenarios

1. Download dataset: [Google Drive download link](http://bit.ly/PPUU-data)
2. openDD
    https://drive.google.com/file/d/12laSzLCaJQa-09sXOnbwaR2INzHurVq1/view?usp=sharing
2. Unpack: `tar xf xy-trajectories.tgz`
3. Set environment variables:
#### TODO
The directory must contain **sqlite file** and subdirectory `image_georeferenced`
```bash
export OPENDD_DIR=.
export NGSIM_DIR=~/Downloads/xy-trajectories

```
3. Run the example:
```bash
python examples/runnable_template.py --dataset opendd --num-episodes 5
```
* `python example/example_scenario_usage.py` - shows how to run scenario in training loop
* `python example/example_replay_ngsim_in_carla.py` - shows how to replay NGSim dataset in CARLA. It was used to generated GIF in this README file


### Feedback
We'd be happy to get any kind of feedback on what should be improved, what's not working etc.


### Credits
Code for interfacing with NGSIM dataset was based on https://github.com/Atcold/pytorch-PPUU
