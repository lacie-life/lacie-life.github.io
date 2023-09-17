---
title: 3D Object Detection for Autonomous Driving - A Comprehensive Survey
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-16 11:11:14 +0700
categories: [Computer Vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

[3D Object Detection for Autonomous Driving: A Comprehensive Survey](https://arxiv.org/abs/2206.09474)

# 1. Introduction

## 1.1. 3D Object Detection

<b> Problem definition: </b> 3D object detection aims to predict bounding boxes of 3D objects in driving scenarios from sensory inputs.
A general formula of 3D object detection can be represented as

$$ B = f_{det}(I_{sensor}) $$

Where $B = \{B_1, ... , B_N \}$ is a set of 3D object in a scence, $I_{sensor}$ is a set of sensory inputs, and $f_{det}$ is a 3D detection model.

<b> How to represent 3D object? </b> 3D object can be represented in different forms, such as 3D bounding box, 3D mesh, 3D point cloud, etc. In this survey, we focus on 3D bounding box representation.

$$B = [x_c, y_c, z_c, l, w, \theta, class]$$

where $(x_c, y_c, z_c)$ is the 3D center of the object, $(l, w, h)$ is the length, width, and height of the object, $\theta$ is the orientation of the object, and $class$ is the object category. In some case, additional information such as velocity, acceleration, and pose can be added to the representation $(v_x, v_y)$.

<b> Sensor inputs: </b> There are many types of sensors that can provide raw data for 3D object detection. Among the sensors, radars,
cameras, and LiDAR (Light Detection And Ranging) sensors are
the three most widely adopted sensory types. Radars have long
detection range and are robust to different weather conditions.
Due to the Doppler effect, radars could provide additional velocity measurements. Cameras are cheap and easily accessible,
and can be crucial for understanding semantics, e.g. the type of
traffic sign. Cameras produce images $I_{cam} ∈ R^{W×H×3}$
for 3D
object detection, where $W$, $H$ are the width and height of an image, and each pixel has 3 RGB channels. Albeit cheap, cameras
have intrinsic limitations to be utilized for 3D object detection.
First, cameras only capture appearance information, and are not
capable of directly obtaining 3D structural information about a
scene. On the other hand, 3D object detection normally requires
accurate localization in the 3D space, while the 3D information,
e.g. depth, estimated from images normally has large errors. In
addition, detection from images is generally vulnerable to extreme weather and time conditions. Detecting objects from images at night or on foggy days is much harder than detection on
sunny days, which leads to the challenge of attaining sufficient
robustness for autonomous driving.

![An illustration of 3D object detection in autonomous driving scenarios](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-1.png?raw=true)

<b> Note: Comparisons with 2D object detection: </b> 2D object
detection, which aims to generate 2D bounding boxes on images,
is a fundamental problem in computer vision. 3D object detection methods have borrowed many design paradigms from the
2D counterparts: proposals generation and refinement, anchors,
non maximum suppression, etc. However, from many aspects,
3D object detection is not a naive adaptation of 2D object detection methods to the 3D space. 

(1) 3D object detection methods
have to deal with heterogeneous data representations. Detection
from point clouds requires novel operators and networks to handle irregular point data, and detection from both point clouds and
images needs special fusion mechanisms. 

(2) 3D object detection
methods normally leverage distinct projected views to generate
object predictions. As opposed to 2D object detection methods
that detect objects from the perspective view, 3D methods have
to consider different views to detect 3D objects, e.g. from the
bird’s-eye view, point view, and cylindrical view. 

(3) 3D object
detection has a high demand for accurate localization of objects
in the 3D space. A decimeter-level localization error can lead to
a detection failure of small objects such as pedestrians and cyclists, while in 2D object detection, a localization error of several
pixels may still maintain a high Intersection over Union (IoU)
between predicted and ground truth bounding boxes. Hence accurate 3D geometric information is indispensable for 3D object
detection from either point clouds or images.

<b> Note: Comparisons with indoor 3D object detection </b>

compared to indoor 3D object detection, there are unique challenges of detection in driving scenarios: 

(1) Point cloud distributions from LiDAR and RGB-D sensors are different. In indoor scenes, points
are relatively uniformly distributed on the scanned surfaces and
most 3D objects receive a sufficient number of points on their
surfaces. However, in driving scenes most points fall in a near
neighborhood of the LiDAR sensor, and those 3D objects that
are far away from the sensor will receive only a few points. Thus
methods in driving scenarios are specially required to handle
various point cloud densities of 3D objects and accurately detect those faraway and sparse objects. 

(2) Detection in driving
scenarios has a special demand for inference latency. Perception
in driving scenes has to be real-time to avoid accidents. Hence
those methods are required to be computationally efficient, otherwise they will not be applied in real-world applications

## 1.2. Datasets

![Datasets for 3D object detection in driving scenarios](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-2.png?raw=true)

<b> Note: Future prospects of driving datasets: </b> The research
community has witnessed an explosion of datasets for 3D object
detection in autonomous driving scenarios. A subsequent question may be asked: what will the next-generation autonomous
driving datasets look like? Considering the fact that 3D object
detection is not an independent task but a component in driving
systems, we propose that future datasets will include all important tasks in autonomous driving: perception, prediction, planning, and mapping, as a whole and in an end-to-end manner,
so that the development and evaluation of 3D object detection
methods will be considered from an overall and systematic view.

## 1.3. Evaluation Metrics

Various evaluation metrics have been proposed to measure the
performance of 3D object detection methods. Those evaluation
metrics can be divided into two categories. 

1. The first category
tries to extend the Average Precision (AP) metric in 2D
object detection to the 3D space:

![Metric](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-3.png?raw=true)

where p(r) is the precision-recall curve. The major difference with the 2D AP metric lies in the matching criterion between ground truths and predictions when calculating
precision and recall. 

- KITTI proposes two widely-used AP
metrics: $AP_{3D}$ and $AP_{BEV}$ , where $AP_{3D}$ matches the predicted
objects to the respective ground truths if the 3D Intersection over
Union (3D IoU) of two cuboids is above a certain threshold, and
$AP_{BEV}$ is based on the IoU of two cuboids from the bird’s-eye
view (BEV IoU). 

- NuScenes proposes $AP_{center}$ where a predicted object is matched to a ground truth object if the distance of
their center locations is below a certain threshold, and NuScenes
Detection Score (NDS) is further proposed to take both $AP_{center}$
and the error of other parameters, i.e. size, heading, velocity, into
consideration. 

- Waymo proposes $AP_{hungarian}$ that applies
the Hungarian algorithm to match the ground truths and predictions, and AP weighted by Heading ($AP_H$) is proposed to incorporate heading errors as a coefficient into the AP calculation.

2. The other category of approaches tries to resolve the evaluation problem from a more practical perspective. <b> The idea is
that the quality of 3D object detection should be relevant to the
downstream task, i.e. motion planning, so that the best detection
methods should be most helpful to planners to ensure the safety
of driving in practical applications </b>. Toward this goal, PKL
measures the detection quality using the KL-divergence of the
ego vehicle’s future planned states based on the predicted and
ground truth detections respectively. SDE leverages the minimal distance from the object boundary to the ego vehicle as the
support distance and measures the support distance error.

<b> Analysis: Pros and Cons of different evaluation metrics </b>

- <b> AP-based metrics </b> are widely used in 3D object detection.  However, those metrics overlook
the influence of detection on safety issues, which are also critical in real-world applications. For instance, a misdetection of an
object near the ego vehicle and far away from the ego vehicle
may receive a similar level of punishment in AP calculation, but
a misdetection of nearby objects is substantially more dangerous
than a misdetection of faraway objects in practical applications.

- <b> Metrics based on the influence of detection on safety issues </b> are more practical and reasonable. PKL [220] and SDE [56] partly
resolve the problem by considering the effects of detection in
downstream tasks, but additional challenges will be introduced
when modeling those effects. PKL requires a pre-trained
motion planner for evaluating the detection performance, but a
pre-trained planner also has innate errors that could make the
evaluation process inaccurate. SDE requires reconstructing
object boundaries which is generally complicated and challenging.

![Overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-4.png?raw=true)


# 2. LiDAR-based 3D Object Detection

![Chronological overview of the LiDAR-based 3D object detection methods](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-5.png?raw=true)

## 2.1. Data representations for 3D object detection

<b> Problem and Challenge. </b> In contrast to images where pixels are
regularly distributed on an image plane, point cloud is a sparse
and irregular 3D representation that requires specially designed
models for feature extraction. Range image is a dense and compact representation, but range pixels contain 3D information instead of RGB values. Hence directly applying conventional convolutional networks on range images may not be an optimal solution. On the other hand, detection in autonomous driving scenarios generally has a requirement for real-time inference. Therefore, how to develop a model that could effectively handle point
cloud or range image data while maintaining a high efficiency
remains an open challenge to the research community.

### 2.1.1. Point cloud-based 3D object detection

Point clouds are first
passed through a point-based backbone network, in which the
points are gradually sampled and features are learned by point
cloud operators. 3D bounding boxes are then predicted based
on the downsampled points and features.  

![An illustration of point-based 3D object detection methods](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-7.png?raw=true)

![A taxonomy of point-based detection methods based on point cloud sampling and feature learning](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-8.png?raw=true)

There are two basic components of a point-based 3D object detector: point cloud sampling
and feature learning. 

<b> Point Cloud Sampling: </b> Farthest Point Sampling (FPS) in PointNet++ has been broadly adopted in point-based detectors,
in which the farthest points are sequentially selected from the
original point set. PointRCNN is a pioneering work that
adopts FPS to progressively downsample input point cloud and
generate 3D proposals from the downsampled points. Similar design paradigm has also been adopted in many following works
with improvements like segmentation guided filtering, feature space sampling, random sampling, voxel-based
sampling, and coordinate refinement.

<b> Point Cloud Feature Learning: </b> A serveal of works leverage set abstraction in PointNet to learn features from
point cloud. Specifically, context points are first collected within
a pre-defined radius by ball query. Then, the context points and 
features are aggregated through multi-layer perceptrons and maxpooling to obtain the new features. There are also other works
resorting to different point cloud operators, including graph operators, attentional operators, and
Transformer. 

<b> Note: Potentials and challenges on point cloud feature
learning and sampling: </b> The representation power of point-based
detectors is mainly restricted by two factors: 

- The number of context points and the context radius adopted in feature learning. Increasing the number of context points will gain more representation power but at the cost of increasing much memory consumption. 

- Suitable context radius in ball query is also an important
factor: the context information may be insufficient if the radius
is too small and the fine-grained 3D information may lose if the
radius is too large. 

These two factors have to be determined carefully to balance the efficacy and efficiency of detection models.

Point cloud sampling is a bottleneck in inference time for
most point-based methods. Random uniform sampling can be
conducted in parallel with high efficiency. However, considering points in LiDAR sweeps are not uniformly distributed, random uniform sampling may tend to over-sample those regions
of high point cloud density while under-sample those sparse regions, which normally leads to poor performance compared to
farthest point sampling. Farthest point sampling and its variants
can attain a more uniform sampling result by sequentially selecting the farthest point from the existing point set. Nevertheless,
farthest point sampling is intrinsically a sequential algorithm and
can not become highly parallel. Thus farthest point sampling is
normally time-consuming and not ready for real-time detection.

### 2.1.2. Grid-based 3D object detection

<b> General Framework: </b> Grid-based 3D object detectors first rasterize point clouds into discrete grid representations, i.e. voxels,
pillars, and bird’s-eye view (BEV) feature maps. Then they apply conventional 2D convolutional neural networks or 3D sparse
neural networks to extract features from the grids. Finally, 3D
objects can be detected from the BEV grid cells. An illustration
of grid-based 3D object detection is shown in Figure 5 and a taxonomy of grid-based detectors is in Table 3. There are two basic
components in grid-based detectors: grid-based representations
and grid-based neural networks.

![An illustration of grid-based 3D object detection methods](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-9.png?raw=true)

![A taxonomy of grid-based detection methods based on models and data representations](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-10.png?raw=true)

<b> Grid-based representations: </b>

1. <b> Voxel-based representation </b> 
 
If we rasterize the detection space into a regular 3D
grid, voxels are the grid cells. A voxel can be non-empty if point
clouds fall into this grid cell. Since point clouds are sparsely distributed, most voxel cells in the 3D space are empty and contain
no point. In practical applications, only those non-empty voxels
are stored and utilized for feature extraction. VoxelNet is
a pioneering work that utilizes sparse voxel grids and proposes
a novel voxel feature encoding (VFE) layer to extract features
from the points inside a voxel cell. 

In addition, there are two categories of approaches trying to improve the voxel representation
for 3D object detection: 

(1) Multi-view voxels. Some methods
propose a dynamic voxelization and fusion scheme from diverse
views, e.g. from both the bird’s-eye view and the perspective
view, from the cylindrical and spherical view, from
the range view. 

(2) Multi-scale voxels. Some papers generate voxels of different scales or use reconfigurable voxels.

2. <b> Pillar-based representation </b>

Pillars can be viewed as special voxels in which the
voxel size is unlimited in the vertical direction. Pillar features
can be aggregated from points through a PointNet and then
scattered back to construct a 2D BEV image for feature extraction. PointPillars is a seminal work that introduces the pillar representation.

3. <b> Bird’s-eye view (BEV) representation </b>

Bird’s-eye view feature map is a dense 2D
representation, where each pixel corresponds to a specific region
and encodes the points information in this region. BEV feature
maps can be obtained from voxels and pillars by projecting the
3D features into the bird’s-eye view, or they can be directly obtained from raw point clouds by summarizing points statistics
within the pixel region. The commonly-used statistics include
binary occupancy and the height and density of
local point cloud.

<b> Grid-based neural networks: </b> There are 2 major types of gridbased networks: 2D convolutional neural networks for BEV feature maps and pillars, and 3D sparse neural networks for voxels.

1. <b> 2D convolutional neural networks </b>

Conventional 2D convolutional neural networks can be applied to the BEV feature map
to detect 3D objects from the bird’s-eye view. In most works, the
2D network architectures are generally adapted from those successful designs in 2D object detection, e.g. ResNet, Region Proposal Network (RPN) and Feature Pyramid Network (FPN) and spatial attention.

2. <b> 3D sparse neural networks </b>

3D sparse convolutional neural
networks are based on two specialized 3D convolutional operators: sparse convolutions and submanifold convolutions,
which can efficiently conduct 3D convolutions only on those
non-empty voxels.

SECOND is a seminal work that
implements these two sparse operators with GPU-based hash
tables and builds a sparse convolutional network to extract 3D
voxel features.

SECOND and this improve version Series (pending).

<b> Note: Pros and cons of different grid representations </b> 

In
contrast to the 2D representations like BEV feature maps and
pillars, voxels contain more structured 3D information. In addition, deep voxel features can be learned through a 3D sparse network. However, <b> a 3D neural network brings additional time and
memory costs </b> . BEV feature map is the most efficient grid representation that directly projects point cloud into a 2D pseudo image without specialized 3D operators like sparse convolutions or
pillar encoding. 2D detection techniques can also be seamlessly
applied to BEV feature maps without much modification. BEVbased detection methods generally can obtain high efficiency
and a real-time inference speed. However, <b> simply summarizing
points statistics inside pixel regions loses too much 3D information </b>, which leads to less accurate detection results compared to
voxel-based detection. Pillar-based detection approaches leverage PointNet to encode 3D points information inside a pillar cell,
and the features are then scattered back into a 2D pseudo image
for efficient detection, which balances the effectiveness and efficiency of 3D object detection

<b> Note: Challenges of the grid-based detection methods. </b> 
A
critical problem that all grid-based methods have to face is <b> choosing the proper size of grid cells </b>. Grid representations are essentially discrete formats of point clouds by converting the continuous point coordinates into discrete grid indices. The quantization process inevitably loses some 3D information and its
efficacy largely depends on the size of grid cells: smaller grid
size yields high resolution grids, and hence maintains more finegrained details that are crucial to accurate 3D object detection.
Nevertheless, reducing the size of grid cells leads to a quadratic
increase in memory consumption for the 2D grid representations
like BEV feature maps or pillars. As for the 3D grid representation like voxels, the problem can become more severe. Therefore, <b> how to balance the efficacy brought by smaller grid sizes
and the efficiency influenced by the memory increase remains an
open challenge to all grid-based 3D object detection methods </b>.


### 2.1.3. Point-voxel based 3D object detection

Point-voxel based approaches resort to a hybrid architecture that
leverages both points and voxels for 3D object detection. Those
methods can be divided into two categories: the single-stage and
two-stage detection frameworks. 

![An illustration of point-voxel based 3D object detection methods](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-11.png?raw=true)

![A taxonomy of point-voxel based detection methods](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-12.png?raw=true)

<b> Single-stage point-voxel detection frameworks </b> Single-stage
point-voxel based 3D object detectors try to bridge the features
of points and voxels with the point-to-voxel and voxel-to-point
transform in the backbone networks. Points contain fine-grained
geometric information and voxels are efficient for computation,
and combining them together in the feature extraction stage naturally benefits from both two representations. The idea that leverages point-voxel feature fusion in backbones has been explored
by many works, with the contributions like <b> point-voxel convolutions </b>, <b> auxiliary point-based networks </b>,
and <b> multi-scale feature fusion </b>.

<b> Two-stage point-voxel detection frameworks </b> Two-stage pointvoxel based 3D object detectors resort to different data representations for different detection stages. Specifically, at the first
stage, they employ a voxel-based detection framework to generate a set of 3D object proposals. In the second stage, keypoints
are first sampled from the input point cloud, and then the 3D proposals are further refined from the keypoints through novel point
operators. PV-RCNN is a seminal work that adopts
as the first-stage detector, and the RoI-grid pooling operator is
proposed for the second-stage refinement. The following works
try to improve the second-stage head with novel modules and operators, e.g. RefinerNet, VectorPool, point-wise attention, scale-aware pooling, RoI-grid attention,
channel-wise Transformer, and point density-aware refinement module.

<b> Note: Potentials and challenges of the point-voxel based
methods: </b> The point-voxel based methods can naturally benefit
from both the fine-grained 3D shape and structure information
obtained from points and the computational efficiency brought
by voxels. However, some challenges still exist in these methods.
<b> For the hybrid point-voxel backbones, the fusion of point and
voxel features generally relies on the voxel-to-point and pointto-voxel transform mechanisms, which can bring non-negligible
time costs </b>. For <b> the two-stage point-voxel detection frameworks,
a critical challenge is how to efficiently aggregate point features for 3D proposals, as the existing modules and operators are
generally time-consuming </b>. In conclusion, compared to the pure
voxel-based detection approaches, the point-voxel based detection methods can obtain a better detection accuracy while at the
cost of increasing the inference time.

### 2.1.4. Range-based 3D object detection

Range image is a dense and compact 2D representation in which
each pixel contains 3D distance information instead of RGB values. Range-based methods address the detection problem from
two aspects: designing new models and operators that are tailored for range images, and selecting suitable views for detection. An illustration of the range-based 3D object detection methods is shown in Figure 7 and a taxonomy is in Table 5.

![An illustration of range-based 3D object detection](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-13.png?raw=true)

![A taxonomy of range-based detection methods based on views, models, and operators](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-14.png?raw=true)

<b> Range-based detection models </b> Since range images are 2D representations like RGB images, range-based 3D object detectors
can naturally borrow the models in 2D object detection to handle
range images. LaserNet is a seminal work that leverages
the deep layer aggregation network (DLA-Net) to obtain
multi-scale features and detect 3D objects from range images.
Some papers also adopt other 2D object detection architectures,
e.g. U-Net, RPN and
R-CNN, FCN, and FPN.

<b> Range-based operators </b> Pixels of range images contain 3D distance information instead of color values, so the standard convolutional operator in conventional 2D network architectures is not
optimal for range-based detection, as the pixels in a sliding window may be far away from each other in the 3D space. Some
works resort to novel operators to effectively extract features
from range pixels, including range dilated convolutions,
graph operators, and meta-kernel convolutions.

<b> Views for range-based detection </b> Range images are captured
from the range view (RV), and ideally, the range view is a spherical projection of a point cloud. It has been a natural solution
for many range-based approaches to detect
3D objects directly from the range view. Nevertheless, detection
from the range view will inevitably suffer from the occlusion
and scale-variation issues brought by the spherical projection.
To circumvent these issues, many methods have been working
on leveraging other views for predicting 3D objects, e.g. the
cylindrical view (CYV), a combination of the range-view, bird’s-eye view (BEV), and/or point-view (PV).

<b> Note: potentials and challenges of the range-based methods </b> Range image is a dense and compact 2D representation, so
the conventional or specialized 2D convolutions can be seamlessly applied on range images, which makes the feature extraction process quite efficient. Nevertheless, compared to bird’s-eye
view detection, detection from the range view is vulnerable to
occlusion and scale variation. Hence, feature extraction from the
range view and object detection from the bird’s eye view becomes the most practical solution to range-based 3D object detection.

## 2.2. Learning objectives for 3D object detection

<b> Problem and Challenge: </b> Learning objectives are critical in object detection. Since 3D objects are quite small relative to the
whole detection range, special mechanisms to enhance the localization of small objects are strongly required in 3D detection.
On the other hand, considering point cloud is sparse and objects
normally have incomplete shapes, accurately estimating the centers and sizes of 3D objects is a long-standing challenge.

### 2.2.1. Anchor-based 3D object detection

Anchors are pre-defined cuboids with fixed shapes that can be
placed in the 3D space. 3D objects can be predicted based on the
positive anchors that have a high intersection over union (IoU) with ground truth. We will introduce the anchor-based 3D object detection methods from the aspect of anchor configurations
and loss functions. 

![An illustration of anchor-based 3D object detection](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-15.png?raw=true)

<b> Prerequisites: </b> The ground truth 3D objects can be represented
as $[x^g, y^g, z^g, l^g, w^g, h^g, θ^g]$ with the class $cls^g$
. The anchors $[x^a, y^a, z^a, l^a, w^a, h^a, θ^a] are used to generate predicted 3D objects $[x, y, z, l, w, h, θ]$ with a predicted class probability $p$.

<b> Anchor configurations: </b> Anchor-based 3D object detection approaches generally detect 3D objects from the bird’s-eye view,
in which 3D anchor boxes are placed at each grid cell of a BEV
feature map. 3D anchors normally have a fixed size for each category, since objects of the same category have similar sizes.

<b> Loss functions: </b> The anchor-based methods employ the classification loss $L_{cls}$ to learn the positive and negative anchors, and
the regression loss $L_{reg}$ is utilized to learn the size and location of an object based on a positive anchor. Additionally, $L_θ$ is
applied to learn the object’s heading angle. The loss function is

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-16.png?raw=true)

VoxelNet is a seminal work that leverages the anchors
that have a high IoU with the ground truth 3D objects as positive anchors, and the other anchors are treated as negatives. To
accurately classify those positive and negative anchors, for each
category, the binary cross entropy loss can be applied to each
anchor on the BEV feature map, which can be formulated as

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-17.png?raw=true)

where $p$ is the predicted probability for each anchor and the target $q$ is 1 if the anchor is positive and 0 otherwise. In addition to
the binary cross entropy loss, the focal loss has also
been employed to enhance the localization ability:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-18.png?raw=true)

where $α = 0.25$ and $γ = 2$ are adopted in most works.

The regression targets can be further applied to those positive
anchors to learn the sizes and locations of 3D objects:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-19.png?raw=true)

where $d^a =\sqrt{(l^a)^2 + (w^a)^2}$ is the diagonal length of an anchor from the bird’s-eye view. Then the SmoothL1 loss is
adopted to regress the targets, which is represented as

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-20.png?raw=true)

To learn the heading angle $θ$, the radian orientation offset can
be directly regressed with the SmoothL1 loss:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-21.png?raw=true)

However, directly regressing the radian offset is normally hard
due to the large regression range. Alternatively, the bin-based
heading estimation is a better solution to learn the heading
angle, in which the angle space is first divided into bins, and binbased classification $L_{dir}$ and residual regression are employed:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-22.png?raw=true)

where $∆θ'$
is the residual offset within a bin. The sine function
can also be utilized to encode the radian offset:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-23.png?raw=true)

In addition to the loss functions that learn the objects’ sizes,
locations, and orientations separately, the intersection over union
(IoU) loss that considers all object parameters as a whole
can also be applied in 3D object detection:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-24.png?raw=true)

where $b^g$
and $b$ are the ground truth and predicted 3D bounding boxes, and $IoU(·)$ calculates the 3D IoU in a differential
manner. Apart from the IoU loss, the corner loss is also
introduced to minimize the distances between the eight corners
of the ground truth and predicted boxes, that is

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-25.png?raw=true)

where $c^g_i$
and $c^i$ are the $i$th corner of the ground truth and predicted cuboid respectively.

<b> Note: potentials and challenges of the anchor-based approaches: </b> The anchor-based methods can benefit from the prior
knowledge that 3D objects of the same category should have
similar shapes, so they can generate accurate object predictions
with the help of 3D anchors. However, since 3D objects are relatively small with respect to the detection range, a large number
of anchors are required to ensure complete coverage of the whole
detection range, e.g. around 70k anchors are utilized in on
the KITTI dataset. Furthermore, for those extremely small
objects such as pedestrians and cyclists, applying anchor-based
assignments can be quite challenging. Considering the fact that
anchors are generally placed at the center of each grid cell, if the
grid cell is large and objects in the cell are small, the anchor of
this cell may have a low IoU with the small objects, which may
hamper the training process

### 2.2.2. Anchor-free 3D object detection

Anchor-free approaches eliminate the complicated anchor designs and can be flexibly applied to diverse views, e.g. the bird’seye view, point view, and range view.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-26.png?raw=true)

<b> Grid-based assignment: </b> In contrast to the anchor-based methods that rely on the IoUs with anchors to determine the positive
and negative samples, the anchor-free methods leverage various
grid-based assignment strategies for BEV grid cells, pillars, and
voxels. PIXOR is a pioneering work that leverages the
grid cells inside the ground truth 3D objects as positives, and
the others as negatives. CenterPoint
utilizes a Gaussian kernel at each object center to assign positive labels. The regression target is:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-27.png?raw=true)

where $dx$ and $dy$ are the offsets between positive grid cells and
object centers. The SmoothL1 loss is leveraged to regress $∆$.

<b> Point-based assignment: </b> Most point-based detection approaches
resort to the anchor-free and point-based assignment strategy, in
which the points are first segmented and those foreground points
inside or near 3D objects are selected as positive samples, and
3D bounding boxes are finally learned from those foreground
points.

<b> Range-based assignment: </b> Anchor-free assignments can also be
employed on range images. A common solution is to select the
range pixels inside 3D objects as positive samples. Different from other methods where
the regression targets are based on the global 3D coordinate system, the range-based methods resort to an object-centric coordinate system for regression.

<b> Set-to-set assignment: </b> DETR is an influential 2D detection
method that introduces a set-to-set assignment strategy to automatically assign the predictions to the respective ground truths
via the Hungarian algorithm:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-28.png?raw=true)

where $M$ is a one-to-one mapping from each positive sample to
a 3D object. The set-to-set assignments have also been explored
in 3D object detection approaches and further introduces a novel cost function for the Hungarian matching.

<b> Note: potentials and challenges of the anchor-free approaches: </b> The anchor-free detection methods abandon the complicated anchor design and exhibit stronger flexibility in terms of
the assignment strategies. With the anchor-free assignments, 3D
objects can be predicted directly on various representations, including points, range pixels, voxels, pillars, and BEV grid cells.
The learning process is also greatly simplified without introducing additional shape priors. Among those anchor-free methods,
the center-based methods have shown great potential in
detecting small objects and have outperformed the anchor-based
detection methods on the widely used benchmarks.

Despite these merits, a general challenge to the anchor-free
methods is to properly select positive samples to generate 3D
object predictions. In contrast to the anchor-based methods that
only select those high IoU samples, the anchor-free methods may
possibly select some bad positive samples that yield inaccurate
object predictions. Hence, careful design to filter out those bad
positives is important in most anchor-free methods.

### 2.2.3. 3D object detection with auxiliary tasks

Numerous approaches resort to auxiliary tasks to enhance the
spatial features and provide implicit guidance for accurate 3D
object detection. The commonly used auxiliary tasks include semantic segmentation, intersection over union prediction, object
shape completion, and object part estimation.

<b> Semantic segmentation: </b> Semantic segmentation can help 3D
object detection in 3 aspects: 

(1) Foreground segmentation could
provide implicit information on objects’ locations. Point-wise
foreground segmentation has been broadly adopted in most pointbased 3D object detectors for proposal generation. 

(2) Spatial features can be enhanced by segmentation.
In Segvoxelnet, a semantic context encoder is leveraged to enhance spatial features with semantic knowledge. 

(3) Semantic segmentation can be utilized as a pre-processing step to filter out background samples and make 3D object detection more efficient.
Some model leverage semantic segmentation to remove those
redundant points to speed up the subsequent detection model.

<b> IoU prediction: </b> Intersection over union (IoU) can serve as a
useful supervisory signal to rectify the object confidence scores.
Cia-ssd proposes an auxiliary branch to predict an IoU score SIoU
for each detected 3D object. During inference, the original confidence scores $S_{conf} = S_{cls}$ from the conventional classification
branch are further rectified by the IoU scores $S_{IoU}$:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-29.png?raw=true)

where the hyper-parameter $β$ controls the degrees of suppressing
the low-IoU predictions and enhancing the high-IoU predictions.
With the IoU rectification, the high-quality 3D objects are easier
to be selected as the final predictions. 

<b> Object shape completion: </b> Due to the nature of LiDAR sensors, faraway objects generally receive only a few points on their
surfaces, so 3D objects are generally sparse and incomplete. A
straightforward way of boosting the detection performance is
to complete object shapes from sparse point clouds. Complete
shapes could provide more useful information for accurate and
robust detection. Many shape completion techniques have been
proposed in 3D detection, including a shape decoder, shape
signatures, and a probabilistic occupancy grid.

<b> Object part estimation: </b> Identifying the part information inside
objects is helpful in 3D object detection, as it reveals more finegrained 3D structure information of an object. Object part estimation has been explored in some works.

<b> Note: future prospects of multitask learning for 3D object detection: </b> 3D object detection is innately correlated with
many other 3D perception and generation tasks. Multitask learning of 3D detection and segmentation is more beneficial compared to training 3D object detectors independently, and shape
completion can also help 3D object detection. There are also
other tasks that can help boost the performance of 3D object detectors. For instance, scene flow estimation could identify static
and moving objects, and tracking the same 3D object in a point
cloud sequence yields a more accurate estimation of this object.
Hence, it will be promising to integrate more perception tasks
into the existing 3D object detection pipeline.









