---
title: Paper note 1 - Multi-Sensor Fusion and Cooperative Perception for Autonomous Driving
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-09 11:11:14 +0700
categories: [Computer Vision]
tags: [Paper]
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

## 2.4. Summary

![ A timeline of image and point cloud fusion methods and their corresponding fusion levels](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-16.png?raw=true)

# 3. Multi-Sensor Fusion for Cooperative Perception

The rapid development of multi-sensor fusion perception
technology has accelerated the implementation of AD while
also facing many opportunities and challenges. Due to
the lack of a global view and the remote sensing capability
of single-vehicle perception, great safety challenges remain
to be solved. With the research progress of 5G/vehicleto-everything (V2X) technology in the field of intelligent transportation systems, V2V and V2I collaborative perception expand the perceived range of self-driving vehicles and avoid traffic accidents caused by the inability
to perceive blind areas. Cooperative perception is of great
necessity for effectively reducing the incidence of traffic accidents and improving road traffic safety.

The current two collaborative perception infrastructures are summarized, with the overall system logic schematic, in Figure below. 

![The infrastructure of the two types of collaborative perception of V2V/V2I based on multi-sensor fusion, including various types of sensor devices, roadside computing units, roadside signal lights, and communication protocols.](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-17.png?raw=true)

We draw the pipeline structure of
multi-sensor fusion collaborative perception,
including multiple sensors for coperception (cameras, lidar, millimeter-wave radar, and so on), the modules of temporal synchronization and spatial alignment (sensors have
different acquisition frequencies and different coordinate
systems), projective transformation for different fusion
types of 2D image and 3D point cloud data, and multisensor coperception fusion strategies. The following introduces the current research status of multi-sensor fusion collaborative sensing methods and the classification
based on the multi-sensor fusion classification strategy
proposed in this article.

![A typical multi-sensor cooperative perception pipeline structure](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-18.png?raw=true)

## 3.1. V2V Collaborative Perception

[Chen et al.](https://arxiv.org/pdf/1905.05265) propose a method, Cooper, that <b> fuses lidar
sensor data collected from different locations and angles
of connected vehicles </b>, and they conduct collaborative
perception research from the raw lidar at the data level
to improve the detection capabilities of AD systems. This
method aligns various point clouds to enhance perception capabilities and verifies that it is possible to transmit
point cloud data for collaborative perception through existing in-vehicle network technologies.

[Xu et al.](https://arxiv.org/abs/2109.07644) present the <b> first large-scale open simulated dataset for V2V perception </b>. It contains over 70 interesting scenes, 11,464 frames, and 232,913 annotated
3D vehicle bounding boxes, some of which are included, collected from eight towns in CARLA and
the digital town of Culver City, CA, USA. Furthermore,
the authors <b> build a comprehensive benchmark implementing multiple models to evaluate multiple information fusion strategies (i.e., early, late, and intermediate
fusion) with state-of-the-art lidar detection algorithms </b>,
and they propose a <b> new attentive intermediate fusion
pipeline for the aggregation of information from multiple
connected vehicles </b>. Experiments show that the proposed
pipeline can be easily integrated with existing 3D lidar
detectors and achieve outstanding performance even
with large compression rates.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-19.png?raw=true)

[Chen et al.](https://arxiv.org/abs/1909.06459) propose
a <b> point cloud feature-based collaborative perception
framework (F-Cooper) for networked autonomous vehicles
to achieve better object detection accuracy </b>. Feature-based
data not only satisfy the collaborative perception training
process but also leverage the inherently small size of features to enable real-time edge computing without the risk
of network congestion. Experimental results show that
better detection results can be achieved by fusing features.
Specifically, the accuracy within 20 m and at long distances is improved by about 10% and 30%, and the edge computing communication delay at lower speeds is shorter.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-21.png?raw=true)

[Wang et al.](https://arxiv.org/abs/2008.07519) adopt V2V communication to improve
the perception and motion prediction performance of
autonomous vehicles. By <b> intelligently aggregating the
information received from multiple nearby vehicles to
observe the same scene from different perspectives, it
is possible to see through the obstructions in the blind
spot and detect very sparse obstacles at a long distance </b>.
The paper <b> proposes the V2VNet network structure, which utilizes a spatially aware graph neural network to aggregate information received from
all nearby autonomous vehicles and intelligently combine information from different time points and viewpoints in the scene </b>. In addition, this paper <b> proposes a new simulated V2V collaborative sensing dataset, V2VSim </b>. Experiments show that the method of sending compressed deep feature map activations through the V2VNet
network structure achieves high accuracy while meeting
the communication bandwidth requirements.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-20.png?raw=true)

V2VNet has three main stages: 

1. A convolutional network block that processes raw sensor data and creates a
compressible intermediate representation; 

2. A cross-vehicle aggregation stage, which aggregates information received
from multiple vehicles with the vehicle’s internal state (computed from its own sensor data) to compute an updated intermediate representation; and 

3. An output network that computes the final perception and prediction outputs.

[Yuan et al.](https://arxiv.org/abs/2109.11615) propose <b> an efficient key point-based deep
feature fusion framework based on the 3D object detector
PV-RCNN, Fusion PV-RCNN (FPV-RCNN) </b>, for collective perception. The paper mainly studies <b> the intervehicle collective perception messages (CPMs) to reduce occlusion </b>, thereby
improving the perception accuracy and safety of AD. The
FPV-RCNN method not only outperforms the perceptual performance of the method using BEV feature fusion but also reduces the data size of CPM communication transmission.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-22.png?raw=true)

[Marvasti et al.](https://arxiv.org/abs/2002.08440) propose a method for <b> collaborative object detection using lidar sensor data by introducing the concept of feature sharing for cooperative object detection </b>. By sharing part of the processed data, the main data part refers to features derived from the middle layers of deep neural networks to better understand the data among environmental cooperative vehicles while maintaining a balance between
the computational and communication loads.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-23.png?raw=true)

[Xu et al.](https://arxiv.org/abs/2203.10638) propose <b> a unified transformer architecture (V2X-ViT) for V2X collaborative perception, which efficiently fuses information from agents (i.e., vehicles and infrastructure) </b>. V2X-ViT consists of alternating layers
of heterogeneous multiagent self-attention and multiscale
windowed self-attention, which captures interagent interactions and spatial relationships of each agent to address
common V2X system challenges, including asynchronous information sharing, pose errors, and heterogeneity of V2X
components. Meanwhile, the proposed V2X-ViT network
structure creates a large-scale OPV2V collaborative dataset
with CARLA and OpenCDA for robust 3D object detection
even in harsh noisy environments.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-24.png?raw=true)

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-25.png?raw=true)

## 3.2. V2I Collaborative Perception

[Zhang et al.](https://ieeexplore.ieee.org/document/9789286) propose <b> a lifelong roadside learning framework for infrastructure-augmented AD (IAAD), r-Learner </b>.
They develop an IAAD system based on this learning framework and verify that the IAAD system significantly improves
the safety of autonomous vehicles. They believe that the
roadside perception unit can collect a large amount of local
sensor environment perception data, and the roadside unit
is equipped with sufficient perception and computing power through an industrial PC equipped with high-end GPUs, which can be used to train deep learning perception models
and for fine-tuning for better perceptual results.

The authors propose a multilevel fusion strategy in the
r-Learner learning framework, as described in Figure below,
including <b> sensor-level fusion, semantic-level fusion, and
cognitive-level fusion </b>. 

1. Sensor-level fusion leverages heterogeneous sensors to improve perception robustness by
fusing raw data transmissions from different sensors in a
system on road. 

2. Semantic-level fusion solves the problem
of continuous tracking across domains. Through semantic fusion, continuous tracking from one region to another,
or continuous reidentification from one region to another,
can be achieved. 

3. Cognitive-level fusion associates regional
results through an intelligent transportation cloud system
and returns global results to each regional server, which
sends environmental perception information to systems
on vehicle through 5G/V2X. 

In addition, global results not
only provide high-level reasoning for improving safety, such as abnormal behavior detection, dangerous driving recognition, and accident risk prediction,
but also contribute to global planning and control, such as traffic
flow prediction and adaptive traffic signal control.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-26.png?raw=true)

[Duan et al.](https://ieeexplore.ieee.org/document/9495350) propose a framework, <b> RGB-PVRCNN, to improve
the environment perception capability of autonomous vehicles at intersections by utilizing
V2I communication technology </b>. The framework <b> integrates
PVRCNN-based visual features </b>. The normal distribution
transform point cloud registration algorithm is applied to
obtain the position of the self-driving vehicle and build a
local perception map detected by the roadside multi-sensor
system. The local perception map is then fed back to the
self-driving vehicle to improve the perception of the selfdriving vehicle, which is beneficial to the path planning of
the intersection and improves the traffic efficiency.

To overcome the technical bottlenecks of autonomous
vehicles and improve the perception range and performance of vehicles in the driving environment, [Mo et al.](https://link.springer.com/article/10.1007/s11042-020-10488-2)
propose a <b> V2I collaborative perception framework </b> for AD
systems. <b> Considering that the intelligent roadside equipment may have short-term perception faults, an improved
Kalman filter algorithm is used, which can also output location information when roadside faults occur </b>. Compared
with pure vehicle perception, simulation experiments verify that the average localization accuracy of this method is
improved by 18% and 19% under normal conditions and
when intelligent roadside equipment fails, respectively.

[Arnold et al.](https://arxiv.org/abs/1912.12147) propose  <b> two cooperative 3D object detection schemes based on single-modal sensors </b>. The early
fusion schemes combine point cloud data from multiple spatially different sensing points before detection, and the later
fusion schemes fuse independently detected bounding boxes
from multiple spatially different sensors. The experimental dataset of the paper is generated by the simulation tool
CARLA, which enables simulating complex driving scenarios and obtains accurate ground truth data for training
and evaluation. The evaluation results show that the performance of the early fusion method is significantly better than
that of the late fusion method but at the expense of higher
communication bandwidth. Meanwhile, in the most challenging scenarios (T-junction and roundabout scenarios),
cooperative perception can recall more than 95% of objects,
while single-point perception can recall only 30% of objects.

[Xiang et al.](https://journals.sagepub.com/doi/full/10.1177/15501329221100412) adopt a multi-sensor fusion algorithm in
a cooperative vehicle–infrastructure system for blind spot
warning, in which the semantic information from the camera
and the range information from the lidar are fused at the data
layer. The lidar point cloud containing semantic information
is then clustered to obtain the type and location information
of the objects. Based on the sensor equipment deployed on
the roadside, the sensing information processed by the fusion
method is sent to the nearby vehicles in real time through 5G
and V2X technology for blind spot early warning, and its feasibility is verified by experiments and simulations.

The Institute for AI Industry Research (AIR), at Tsinghua University, and Baidu released the first vehicle–infrastructure cooperation-related 3D object detection dataset for
large-scale real scenes, DAIR-V2X. DAIR-V2X data are the
first large-scale, multimodal, and multiview vehicle–road
collaboration dataset for vehicle–infrastructure. As the main
section, its VICAD consists of 71,254 lidar frames and 71,254
camera frames, which are captured from real scenes for research of the vehicle–road collaborative global perspective
and long-range perception capability technology. In addition
to being used for vehicle–infrastructure cooperative 3D object detection, these data enable studying the time synchronization problem between vehicles and infrastructure sensors
as well as the cost of data transmission between them.

## 3.3. Summary

# 4. Challenges and Discussion

## 4.1. Relationship Between Multimodal Fusion and Cooperative Perception

Multimodal fusion is the basis of collaborative perception, and collaborative perception is an extended application of multimodal fusion. <b> The essence of multimodal
fusion based on multi-sensors lies in the way of combining data information to meet the requirements of the
timeliness, accuracy, and robustness of the perception
system </b>. Previous research on multimodal perception
strategies mainly includes three types of fusion at the
data level, feature level, and result level. Single-vehicle
multi-sensor data are easy to use to achieve data alignment because the data perception spaces are basically coincident. Therefore, a variety of fusion strategies can be used to compensate for the disadvantages of a single sensor and improve perceptual performance. However, the
perception range of single vehicle is limited and cannot
be used to assist decision making with the information
outside the perception range. The information within the
limited view is easily blocked by other participants and
environmental objects, resulting in information loss and
data quality degradation.
Collaborative perception, including V2V and V2I scenarios, aims to combine the perception information of multiple
traffic participants or vehicle–infrastructure to achieve a
larger perception range and obtain better quality perception information than single view. The effective combination of multimodal data from different traffic participants
or vehicle–infrastructure enriches and enhances the available data with information sharing and breaks the bottleneck of the limited visual field of single-vehicle perception.
However, the technology used in the underlying layer of
collaborative perception is still multimodal fusion, with
further extensions only on the data sources. Therefore, the
fusion strategies for collaborative perception also follow the
taxonomy of fusion strategies proposed in this article.

## 4.2. Datasets of Multimodal Fusion and Cooperative Perception

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/sensor-27.png?raw=true)

## 4.3. Challenges of Cooperative Perception

### 4.3.1. Communication and Computing Efficiency

The perception processes of V2V and V2I achieve perception beyond the visual field by sharing reliable information
of each traffic participant, which requires the realization
of three steps: 

1. Single-vehicle perception. 

2. Information transmission.

3. Collaborative perception.

The communication capability determines the time consumption of
information transmission, while the amount of transmitted data affects the overall time performance.

Most existing works assume complete data and controlled temporal relationships under good communication
conditions. However, practical communication faces many
challenges: 

1. A large number of vehicles in the same scene
makes it difficult to communicate perfectly with limited
bandwidth. 
2. Unavoidable communication delay and highspeed motion make it difficult to get real-time shared information.
3. Environmental factors lead to communication
interruptions and information loss. Future work needs to address these issues and achieve a more robust cooperative
perceptual system based on various fusion strategies.

To reduce the overhead of data computing and save the
total amount of data transmitted by communication, it is
necessary to choose a reasonable collaborative approach
as well as appropriate data processing. The synergistic
approach of prevalent feature fusion uses encoded features
as shared data, which provides more impact than the raw
data required for data-level fusion and has better representation capability and application flexibility than resultlevel fusion. In addition, effective key data selection and
feature compression can further reduce the transmission
overhead. Data selection, such as foreground and background separation, can remove redundant information as
well as emphasize key information.

### 4.3.2. Complex Scenarios and Multimodality

Most works assume that sensor data are collected normally
and with high quality, while data loss and low data quality
exist in actual scenes. In practice, key information occlusion
exists between vehicles and buildings, data quality decreases as distance increases and is greatly affected by extreme weather, and the high-speed movement of objects causes
difficulties in spatiotemporal alignment of data. It is difficult to face the changing and harsh environment by using a
single type of sensor, while the combination of multiple sensors can not only enrich the available environmental data
but also improve the robustness of algorithmic strategies to
the issues of weather, distance, and data quality.
However, naive data fusion strategies are still not sufficient
for applications. In practical scenarios, different vehicles
may be equipped with various numbers and types of sensors
with different sensor characteristics and data quality, which
leads to serious long tail phenomena. Hence, a generic cooperative perception framework with variable modal types
and numbers of traffic participants and robust fusion strategies needs to be thoroughly explored in future work.

### 4.3.3. Large-Scale Datasets and Domain Adaption

The deep learning approach adopted for collaborative scenarios is driven by massive data, especially data from real
complex scenes with high latency, low quality, and severe
weather conditions. However, limited by the expensive cost
of data acquisition and data labeling, the current collaborative data are small in size or generated from simulators.

### 4.3.4. Security and Privacy

With the development of 5G/V2X communication technology,
extensive research and application of multi-sensor fusion in
the field of V2X collaborative perception is enabled. However,
there are security attacks and data privacy risks in the process of collaborative perception through shared data transmission. Through multi-sensor data sharing among
vehicles, V2V cooperative perception increases the acquisition of multiview surrounding environment information,
expands its own perception field of view, improves its own
environmental perception ability, and contributes to subsequent planning and decision making. However,
when vehicles share data collaboratively, there can be malicious attacks on the shared data or unreliable tampered information as well as the sending of malicious instructions
to other vehicles, including sending a large amount of computing task information to occupy the computing resources
and network bandwidth of the vehicle system. What is more,
the technology sends some command information to take
over and control vehicles, and this shared information increases the security risks of coordinated vehicles.
In addition, during the transmission of V2V and V2I shared
data information to nearby peer vehicles or the cloud, due
to anonymous or pseudonymous attack access by attackers,
privacy-related user data may be stolen, mainly including a
large amount of sensitive data generated by vehicles, including location privacy data and user vehicle identity information. Therefore, security measures need to be taken
at the end, the link, and the cloud simultaneously. End security deals with vulnerabilities of in-vehicle networks, such
as engine control units, and distributed denial-of-service
attacks on controller area networks. Link security prevents
the data transmission from being tampered with and eaves dropped on. Cloud security protects the cloud from being accessed by unauthorized users and also keeps the data stored
on the cloud from being abused.

# 5. Conclusion

Multimodal perception technology is still booming, and
the future research and application scope of multi-sensor fusion-based cooperative perception technology will be more
extensive and deeper because of its over-the-horizon and
global perception capabilities. At the same time, combining
social signals in the field of intelligent transportation (including weather information, road condition information,
road environment information, etc.), further integrating
multi-source information, providing comprehensive driving assistance decision-making information. This emerging
and promising multi-source heterogeneous information fusion intelligent transportation technology has great potential to improve the efficiency and safety of road traffic.









