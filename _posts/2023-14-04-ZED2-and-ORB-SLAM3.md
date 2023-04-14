---
title: ZED2 with ORB-SLAM3 (Stereo-IMU mode) step-by-step
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-14-04 11:11:14 +0700
categories: [Computer Vision]
tags: [Tutorial]
img_path: /assets/img/post_assest/
render_with_liquid: false
---


## Introduction

It is also a project requirement, and needs to use visual inertial navigation to do some development, so the first step is to do some algorithm testing-simulation and physical testing. After passing the simulation test results, it is finally decided to useORB-SLAM3 to complete the task.

- System version:<b> Ubuntu 20.04, ROS Noetic </b>

## Contents

- Introduction
- Cuda and CuDNN installation
- ZED SDK installation
- Binocular camera, IMU calibration and joint calibration
- Algorithm operation results display
- Discussion

## Cuda and CuDNN installation

- CUDA Toolkit Install
- CuDNN install

## ZED SDK

- ZED SDK
- ZED Python API
- ZED ROS Wrapper

## Sensor Calibration

### Install Calibration tool

- Install dependencies

```bash
sudo apt-get install python-setuptools python rosinstall ipython libeigen3-dev libboost-all-dev doxygen libopencv-dev ros-noetic-vision-opencv ros-noetic-image-transport-plugins ros-noetic-cmake-modules software-properties-common libpoco-dev python-matplotlib python-scipy python-git python-pip libtbb-dev liblapack-dev python-catkin-tool libv4l-dev

sudo add-apt-repository ppa:igraph/ppa
sudo apt-get update
sudo apt-get install python-igraph

```

- Create a new workspace, remember to add the environment variable source ~/kalibr_workspace/devel/setup.bash in bashrc

```bash
mkdir -p ~/kalibr_ws/src
cd ~/kalibr_ws
catkin_make
```

- Download the source code and compile

```bash

cd ~/kalibr_ws/src

git clone -b 20 https://github.com/ethz-asl/kalibr.git

catkin build -DCMAKE_BUILD_TYPE=Release -j4

```

- Download and compile code_utils (here to compile under the workspace created by catkin_make, mine is catkin_ws)

```bash

cd ~/catkin_ws/src

git clone https://github.com/gaowenliang/code_utilsarget_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.088
tagSpacing: 0.3

sudo apt-get install libw-dev

cd ../

catkin_make

```

- Download and compile imu_utils

```bash
git clone https://github.com/geowenliang/imu_utils
cd ../
catkin_make
```

### Download the calibration board and yaml file

Enter GitHub to download relevant parameter files https://github.com/ethz-asl/kalibr/wiki/downloads#calibration-targets
Select Aprilgrid 6x6 0.8x0.8 m (A0 page), download its pdf and yaml files

![Fig.1](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/tag.png?raw=true)
