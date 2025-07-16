![](https://img.shields.io/badge/release-1.0.0-brightgreen.svg?style=popout-square)
![](https://img.shields.io/badge/CARLA-0.9.6+-blue.svg?style=popout-square)
![](https://img.shields.io/badge/python-3.6+-blue.svg?style=popout-square)
![](https://img.shields.io/badge/license-MIT-orange.svg?style=popout-square)

CARLA real traffic scenarios
========================
<p align="center">
  <img width="100%" 
       height="auto" 
       alt="readme-main"
       src="https://user-images.githubusercontent.com/64484917/90624607-7f3cb980-e218-11ea-8877-30c599e76f4f.gif">
</p>

NGSIM freeway             |  openDD roundabout
:-------------------------:|:-------------------------:
![readme-ngsim](https://user-images.githubusercontent.com/64484917/90623562-f07b6d00-e216-11ea-872f-a00000b75c14.gif) | ![readme-opendd](https://user-images.githubusercontent.com/64484917/90628518-0cced800-e21e-11ea-96fe-a30e3762ab1a.gif)

### Features

1. Handmade maps for [CARLA](https://carla.org/) that mimic real-world roads 
    - 7 roundabouts (https://arxiv.org/abs/2007.08463)
    - 2 freeways ([US101](https://www.fhwa.dot.gov/publications/research/operations/07030/index.cfm), [I-80](https://www.fhwa.dot.gov/publications/research/operations/06137/))
2. Code that transfers real-world traffic from datasets into CARLA
3. Scenario API similar to [OpenAI Gym](https://gym.openai.com/)

We've also trained and benchmarked policies on lane change maneuvers and roundabout navigation tasks.

More details, our article and videos of trained policies are **published on [our website](https://sites.google.com/view/carla-real-traffic-scenarios/home).**

### Prerequisites
```bash
git clone https://github.com/deepsense-ai/carla-real-traffic-scenarios.git && cd carla-real-traffic-scenarios
pip install -r requirements.txt
```

If working on remote servers with no desktop, use [gdown](https://pypi.org/project/gdown/) to download from Google Drive links, e.g.
```bash
pip install gdown
gdown --id 1FCHL7YJk12AwfxuMPmwXPJj71n3mwSxE
```

#### 1. CARLA
Download and extract CARLA ([0.9.9.4 download link](https://carla-releases.b-cdn.net/Linux/CARLA_0.9.9.4.tar.gz)). Then, add PythonAPI wheel to your `PYTHONPATH`:
```bash
cd ~/Downloads
wget https://carla-releases.b-cdn.net/Linux/CARLA_0.9.9.4.tar.gz

export CARLA_ROOT=/home/$USER/Downloads/CARLA_0.9.9.4
mkdir -p $CARLA_ROOT;
tar zxvf CARLA_0.9.9.4.tar.gz --directory $CARLA_ROOT

export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg:$PYTHONPATH
```
#### 2. Maps
Download and import our CARLA package with new maps: 7 roundabout, 2 freeways: [Download link](https://deepsense.ai/data/carla-package-NGSIM-openDD.tar.gz)
```bash
mv ~/Downloads/carla-package-NGSIM-openDD.tar.gz $CARLA_ROOT/Import/
$CARLA_ROOT/ImportAssets.sh
```
    
#### 3. Datasets
Download and extract: [NGSIM](https://drive.usercontent.google.com/download?id=126SkGEa0eVk4GGsw7emYuyDhZI6qM4MN&export=download), [opendDD](hhttps://drive.usercontent.google.com/download?id=1qfW5YdosTKD-GlZj5OjJ5Ag77Vv1oPoT&export=download&authuser=0)
```
sudo apt install p7zip-full

# no space after -o
7z x ~/Downloads/openDD.7z -oopenDD

mkdir NGSIM && tar zxvf ~/Downloads/NGSIM.tgz --strip-components 1 --directory NGSIM
```
### Quickstart

##### Terminal I
```bash
cd $CARLA_ROOT
./CarlaUE4.sh -benchmark -fps=10
```

##### Terminal II
Wait until server boots up. Feel free to play with the code.
```bash

# Directory which contains "rdb1to7.sqlite" and "image_georeferenced/"
export OPENDD_DIR=~/Downloads/openDD

# Directory which contains "i80/" and "us101/"
export NGSIM_DIR=~/Downloads/NGSIM
python examples/runnable_template.py --dataset opendd --num-episodes 5
```

### Feedback
We encourage you send us any kind of feedback on what should be improved, what's not working etc.

### Credits
Code for interfacing with NGSIM dataset was based on https://github.com/Atcold/pytorch-PPUU

Authors (cannot be disclosed yet):
- Anonoymous 1 (anonoymous@email.com)
- Anonoymous 2 (anonoymous@email.com)
- Anonoymous 3 (anonoymous@email.com)
- Anonoymous 4 (anonoymous@email.com)
- Anonoymous 5 (anonoymous@email.com)
- Anonoymous 6 (anonoymous@email.com)
