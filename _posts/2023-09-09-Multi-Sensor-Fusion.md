---
title: Paper note 1 - Multi-Sensor Fusion and Cooperative Perception for Autonomous Driving
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-09 11:11:14 +0700
categories: [Computer Vision]
tags: [Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# [Multi-Sensor Fusion and Cooperative Perception for Autonomous Driving: A Review](https://ieeexplore.ieee.org/abstract/document/10208208)

# 1. Introduction

Currently, there are two technical routes for autonomous driving (AD), namely, 
single-vehicle intelligent AD and vehicle–infrastructure cooperative AD
(VICAD). Among them, environment perception,
as an important component of the whole AD system,
has become a hot research topic in industry and academia. 

## The environmental perception information of AD mainly relies on sensors, such as cameras, radar, and lidar.
 
- <b> Image data: </b> color information and detailed
semantic information at the texture level but not depth information, which makes it difficult to obtain accurate position
information of objects in the world coordinate system and
leads to vulnerability to extreme weather conditions, such as
bright light.

- <b> Lidar data: </b> 3D information and are easy to
process to obtain spatial location information, but the data
themselves do not have detailed information, such as color,
and they are sparse and have a limited ability to discriminate
objects.

=> <i> The perception effect of a single data type
of image and point cloud is limited, but the fusion of the two
perceptions can achieve complementary advantages and obtain high-dimensional data with both detailed semantic and
spatial depth information, providing the possibility of generating better perception results </i>


## Perception systems in AD need to meet the following conditions:

- <b> High accuracy: </b> Providing accurate information about
the driving environment is the basis for planning decisions and control systems to work properly.

- <b> High robustness: </b> They should work properly in bad weather, situations not covered during training (open conditions), and when some sensors degrade or even fail.

- <b> Rapid real-time processing: </b> Data acceptance, algorithm processing, and actuator actions take time, especially when the
self-driving car is traveling at high speed, and a fast sensory processing system is a prerequisite to ensure safety.

=> To achieve these goals, self-driving vehicles are often
equipped with different types of sensors (e.g., cameras, lidar,
and millimeter-wave radar) and fuse different sensor modalities to exploit their complementary properties. 

However, the implementation of fused sensing of different sensors also
faces the problem of <b> temporal synchronization spatial alignment </b> , due to different acquisition frequencies of different
sensors and different coordinate systems for sensory data
acquisition, as well as differences in environmental perception, due to different <b> fused sensing strategies </b>.

# 2. Multi-Sensor Fusion Perception

## 2.1. Taxonomy of Fusion Strategies

In multi-sensor application scenarios, the fusion of image
and point cloud data is a common perception scheme. Image data have fine-grained 2D texture information, and
their missing depth information can be supplemented by
3D information of point cloud data, which also makes up
for the latter’s weak discriminative power and sparse data
granularity. The single-modal perception algorithms for
images and point clouds have been well researched, and
plenty of research concentrates on how to effectively represent the features of each modal branch and perform proper fusion. Therefore, effectively summarizing the fusion
strategies used by related algorithms is helpful for further
research of fusion algorithms design.

![The fusion methodology](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-1.png?raw=true)

We first summarize the fusion strategies into two types,
as symmetrical fusion and asymmetric fusion, according to
whether the data form of each branch is matched or not and
further subdivide the two types of strategies according to
the different data forms used by each modal branch. There
are three minor classes in symmetric fusion: data–data fusion, feature–feature fusion, and result–result fusion. And
there are four minor classes in asymmetric fusion: data–
feature fusion, feature–result fusion, data–result fusion,
and result selection fusion. In the following, the main idea
of the proposed taxonomy and definition of categories is
clarified in detail with corresponding fusion algorithms.

## 2.2. Symnetric Fusion Startegies

Symmetric fusion methods fuse data of different modalities
at the same level, including three classes: data–data fusion, feature–feature fusion, and result–result fusion. Methods
using the symmetric fusion strategy usually adopt a symmetrical structure design; branches of different modalities
have equal importance, and data processing is relatively
synchronized so that the data for fusion are produced at
almost the same stage.

### 2.2.1. Data–Data Fusion

The data–data fusion is carried out <b> based on the spatial correspondence among the multimodal raw data </b>. Multi-sensor
data are <b> spatially transformed into the same coordinate system for fusion </b>, and then single-modal data are augmented
and further processed as input to subsequent modules.

![ A data–data fusion strategy architecture. RGB: red–green–blue; FOV: field of view](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-2.png?raw=true)


We believe that data–data fusion is easy to implement
and can effectively improve the data dimension. Therefore,
this kind of fusion provides algorithms the possibility to
extract more discriminative semantic information after a
further encoding process.

![An example data–data fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-3.png?raw=true)

### 2.2.2. Feature–Feature Fusion

The feature–feature fusion also <b> relies on the corresponding relationship established among the multi-sensor data spaces . The features encoded by each sensor model branch
are projected into the same coordinate system for fusion </b> ,
such as projecting the point cloud features into the image
space, and fused with image features through simply addition or concatenation.

![A feature–feature fusion strategy architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-4.png?raw=true)

We believe that feature–feature fusion can realize the
combination of different modal semantic information at the
feature level, which helps to extract more robust perceptual
information. Compared with the combination at the data
level, it has the possibility to produce better perceptual accuracy. But at the same time, this fusion method leads to
a high degree of coupling of multi-sensor branches; thus,
all branches need to be trained and tuned together, which
makes the modification of the model more difficult.

![An example feature–feature fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-5.png?raw=true)


### 2.2.3. Result–Result Fusion

The result–result fusion <b> uses an additional model or rulebased strategy to fuse the independent perceptual results
of each modal branch and gets the final result after an integration process </b>.

![ A result–result fusion strategy architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-6.png?raw=true)

We believe that the fusion step of result–result fusion
is independent of the encoding process of multimodal
branches and that it is performed in the form of multibranch results integration, which is convenient to change
the number of branches and optimize the model of the
branches. But at the same time, this fusion method does
not fully utilize the opportunity to perform a deeper fusion
of various types of data at the feature level and generate
higher-level features.

![An example result–result fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-7.png?raw=true)


## 2.3. Asymmetric Fusion Strategies

In asymmetric fusion (that is, <b> data of different levels are involved in the fusion process </b>), there are four classes, including data–feature fusion, feature–result fusion, data–result
fusion, and result selection fusion. This kind of algorithm
adopts an asymmetric structure design; it can be observed
that a dominant modality provides primary information,
while other modalities provide auxiliary information. The perception task is mainly based on the dominant mode, and
the fused data are produced at different stages of each branch.

### 2.3.1. Data–Feature Fusion

The data–feature fusion uses the encoded features of one
modality to enhance the raw data of other modalities, and
the augmented data are processed to obtain the final perceptual result, which is actually the data of the dominant modality. 

![A data–feature fusion strategy architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-8.png?raw=true)

We believe that the advantage of data–feature fusion
is that the encoding process of different modalities is arranged sequentially; the coupling
is low so that the model of each modality can be separately optimized
and conveniently combined as a
whole. Under multitask joint supervision, each branch can be more
effectively trained, and the model
itself has strong interpretability.
The degree of such fusion is slightly
lower than that of feature–feature
fusion, and the final effect of the algorithm depends more
on the performance of the latter enhanced branch.

![An example data–feature fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-9.png?raw=true)

### 2.3.2. Feature–Result Fusion

Unlike result–result fusion methods, which get proposals
from all branches, the feature–result fusion methods have
<b> a dominant modality branch to generate the first-stage results like proposals, which are used as an ROI of features
from all branches. After feature selection, features are
fused and input to the second-stage modules for further
perception refinement </b>.

![A feature–result fusion strategy architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-10.png?raw=true)

We believe that in the feature–result fusion, the branch
that provides the one-stage resultant data can be partially
separated and optimized. The subsequent feature-selective
operation, such as region pooling, can be further promoted for
providing better features and the possibility to obtain higherprecision perceptual results. This fusion strategy relies on the
accuracy of the initial prediction results and the validity of the
selected features. To conduct results refinement in the second
stage often requires multiple branches to be trained together,
and thus, the model becomes more complex. 

![An example feature–result fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-11.png?raw=true)

### 2.3.3. Data–Result Fusion

Similar to feature–result fusion, <b> data–result fusion projects
proposals from the dominant modality branch to itself and
other branches for feature selection or raw data selection
based on the corresponding spatial region. The enhanced
raw data or feature is then processed for final perception </b>.

![A data–result fusion strategy architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-12.png?raw=true)

We argue that the pros and cons of data–result fusion are
similar to feature–result fusion, and the branch that provides raw data is more efficient since the feature encoding
has not yet been performed. But intuitively, the feature–result fusion has the possibility to obtain a better fusion effect
because the encoded features tend to be rich in information
with wider receptive fields, and the selection of data from
raw data relatively filters out more information that might be useful for subsequent processing.

![An example data–result fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-13.png?raw=true)

### 2.3.4. Result-Selecttive Fusion

The result-selective fusion strategy is <b> to get first-stage proposals from one of the branches, and the spatial regions
represented by the proposals are used as the data range of
other modalities for raw data selection. The filtered data
are then input to the latter network for efficient processing </b>.

![A result-selective fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-14.png?raw=true)

We believe that the coupling of each modal branch in
the result-selective fusion is low, and due to the raw data
selection, the latter branch becomes more efficient but might lose some vital information at the same time.

![An example result-selective fusion architecture](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-15.png?raw=true)





