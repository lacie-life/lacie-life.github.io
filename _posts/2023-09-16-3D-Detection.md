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

<b> Note: Potentials and challenges of the range-based methods </b> Range image is a dense and compact 2D representation, so
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

<b> Note: Potentials and challenges of the anchor-based approaches: </b> The anchor-based methods can benefit from the prior
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

<b> Note: Potentials and challenges of the anchor-free approaches: </b> The anchor-free detection methods abandon the complicated anchor design and exhibit stronger flexibility in terms of
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

<b> Note: Future prospects of multitask learning for 3D object detection: </b> 3D object detection is innately correlated with
many other 3D perception and generation tasks. Multitask learning of 3D detection and segmentation is more beneficial compared to training 3D object detectors independently, and shape
completion can also help 3D object detection. There are also
other tasks that can help boost the performance of 3D object detectors. For instance, scene flow estimation could identify static
and moving objects, and tracking the same 3D object in a point
cloud sequence yields a more accurate estimation of this object.
Hence, it will be promising to integrate more perception tasks
into the existing 3D object detection pipeline.

# 3. Camera-based 3D Object Detection

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-32.png?raw=true)

## 3.1. Monocular 3D object detection

<b> Problem and Challenge: </b> Detecting objects in the 3D space from
monocular images is an ill-posed problem since a single image
cannot provide sufficient depth information. Accurately predicting the 3D locations of objects is the major challenge in monocular 3D object detection. Many endeavors have been made to
tackle the object localization problem, e.g. inferring depth from
images, leveraging geometric constraints and shape priors. Nevertheless, the problem is far from being solved. Monocular 3D
detection methods still perform much worse than the LiDARbased methods due the poor 3D localization ability, which leaves
an open challenge to the research community.

### 3.1.1. Image-only monocular 3D object detection

Inspired by the 2D detection approaches, a straightforward solution to monocular 3D object detection is to directly regress
the 3D box parameters from images via a convolutional neural network. The direct-regression methods naturally borrow designs from the 2D detection network architectures, and can be
trained in an end-to-end manner. These approaches can be divided into the single-stage/two-stage, or anchor-based/anchorfree methods

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-30.png?raw=true)

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-31.png?raw=true)


<b> Single-stage anchor-based methods </b> 

Anchor-based monocular detection approaches rely on a set of 2D-3D anchor boxes
placed at each image pixel, and use a 2D convolutional neural
network to regress object parameters from the anchors. 

Specifically, for each pixel $[u, v]$ on the image plane, a set of 3D anchors
$[w^a, h^a, l^a, θ^a]_{3D}$, 2D anchors $[w^a, h^a]_{2D}$, and depth anchors $d^a$
are pre-defined. An image is passed through a convolutional network to predict the 2D box offsets $δ_{2D} = [δ_x, δ_y, δ_w, δ_h]_{2D}$ and
the 3D box offsets $δ_{3D} = [δ_x, δ_y, δ_d, δ_w, δ_h, δ_l, δ_θ]_{3D}$ based on
each anchor. Then, the 2D bounding boxes $b_{2D} = [x, y, w, h]_{2D}$ can be decoded as

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-33.png?raw=true)

and the 3D bounding boxes $b_{3D} = [x, y, z, l, w, h, θ]_{3D}$ can be
decoded from the anchors and $δ_{3D}$:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-34.png?raw=true)

where $[u^c, v^c]$ is the projected object center on the image plane.
Finally, the projected center $[u^c, v^c]$ and its depth $d^c$
are transformed into the 3D object center $[x, y, z]_{3D}$:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-35.png?raw=true)

where $K$ and $T$ are the camera intrinsics and extrinsics

M3D-RPN is a seminal paper that proposes the anchorbased framework, and many papers have tried to improve this
framework, e.g. extending it into video-based 3D detection,
introducing differential non-maximum suppression, designing an asymmetric attention module.

<b> Single-stage anchor-free methods: </b> 

Anchor-free monocular detection approaches predict the attributes of 3D objects from images without the aid of anchors. Specifically, an image is passed
through a 2D convolutional neural network and then multiple
heads are applied to predict the object attributes separately. The
prediction heads generally include a category head to predict the
object’s category, a keypoint head to predict the coarse object
center $[u, v]$, an offset head to predict the center offset $[δ_x, δ_y]$
based on $[u, v]$, a depth head to predict the depth offset $δ_d$, a size
head to predict the object size $[w, h, l]$, and an orientation head
to predict the observation angle $α$. The 3D object center $[x, y, z]$
can be converted from the projected center $[u^c, v^c]$ and depth $d^c$:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-36.png?raw=true)

where $σ$ is the sigmoid function. The yaw angle $θ$ of an object
can be converted from the observation angle $α$ using

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-37.png?raw=true)

CenterNet first introduces the single-stage anchor-free
framework for monocular 3D object detection. Many following papers work on improving this framework, including novel
depth estimation schemes [166, 294, 369], an FCOS-like
architecture, a new IoU-based loss function, keypoints, pair-wise relationships, camera extrinsics prediction, and view transforms.

<b> Two-stage methods: </b> 

Two-stage monocular detection approaches
generally extend the conventional two-stage 2D detection architectures to 3D object detection. Specifically, they utilize a 2D
detector in the first stage to generate 2D bounding boxes from
an input image. Then in the second stage, the 2D boxes are lifted
up to the 3D space by predicting the 3D object parameters from
the 2D RoIs. ROI-10D extends the conventional Faster RCNN architecture with a novel head to predict the parameters of 3D objects in the second stage. A similar design paradigm
has been adopted in many works with improvements like disentangling the 2D and 3D detection loss, predicting heading
angles in the first stage, learning more accurate depth information.

<b> Note: Potentials and challenges of the image-only methods: </b> 

The image-only methods aim to directly regress the 3D
box parameters from images via a modified 2D object detection
framework. Since these methods take inspiration from the 2D
detection methods, they can naturally benefit from the advances
in 2D object detection and image-based network architectures.
Most methods can be trained end-to-end without pre-training or
post-processing, which is quite simple and efficient.
A critical challenge of the image-only methods is to accurately predict depth $d^c$
for each 3D object.
This observation indicates that the depth error dominates the total errors and becomes the most critical factor hampering accurate monocular detection. Nevertheless, depth estimation from
monocular images is an ill-posed problem, and the problem becomes severer with only box-level supervisory signals.

### 3.1.2. Depth-assisted monocular 3D object detection

Depth estimation is critical in monocular 3D object detection.
To achieve more accurate monocular detection results, many papers resort to pre-training an auxiliary depth estimation network.
Specifically, a monocular image is first passed through a pretrained depth estimator, e.g. MonoDepth or DORN, to
generate a depth image. Then, there are mainly two categories
of methods to deal with depth images and monocular images.
The depth-image based methods fuse images and depth maps
with a specialized neural network to generate depth-aware features that could enhance the detection performance. The pseudoLiDAR based methods convert a depth image into a pseudoLiDAR point cloud, and LiDAR-based detectors can then be applied to the point cloud to predict 3D objects. 

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-38.png?raw=true)

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-31.png?raw=true)


<b> Depth-image based methods: </b> Most depth-image based methods leverage two backbone networks for RGB and depth images
respectively. They obtain depth-aware image features by fusing
the information from the two backbones with specialized operators. More accurate 3D bounding boxes can be learned from the
depth-ware features and can be further refined with depth images. MultiFusion is a pioneering work that introduces the
depth-image based detection framework. Following papers adopt
similar design paradigms with improvements in network architectures, operators, and training strategies, e.g. a point-based attentional network, depth-guided convolutions, depth-conditioned message passing, disentangling appearance
and localization features, and a novel depth pre-training
framework.

<b> Pseudo-LiDAR based methods: </b> Pseudo-LiDAR based methods
transform a depth image into a pseudo-LiDAR point cloud, and
LiDAR-based detectors can then be employed to detect 3D objects from the point cloud. Pseudo-LiDAR point cloud is a data
representation, where they convert a
depth map $D ∈ R^{H×W}$ into a pseudo point cloud $P ∈ R^{HW×3}.
Specifically, for each pixel $[u, v]$ and its depth value d in a depth
image, the corresponding 3D point coordinate $[x, y, z]$ in the
camera coordinate system is computed as

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-39.png?raw=true)

where $[c_u, c_v]$ is the camera principal point, and $f_u$ and $f_v$ are
the focal lengths along the horizontal and vertical axis respectively. Thus $P$ can be obtained by back-projecting each pixel in
$D$ into the 3D space. $P$ is referred as the pseudo-LiDAR representation: it is essentially a 3D point cloud but is extracted
from a depth image instead of a real LiDAR sensor. Finally,
LiDAR-based 3D object detectors can be directly applied on
the pseudo-LiDAR point cloud P to predict 3D objects. Many
papers have worked on improving the pseudo-LiDAR detection
framework, including augmenting pseudo point cloud with color
information, introducing instance segmentation, designing a progressive coordinate transform scheme, improving pixel-wise depth estimation with separate foreground
and background prediction, domain adaptation from real
LiDAR point cloud, and a new physical sensor design.

PatchNet challenges the conventional idea of leveraging the pseudo-LiDAR representation $P ∈ R^{HW×3}$
for monocular 3D object detection. They conduct an in-depth investigation
and provide an insightful observation that the power of pseudoLiDAR representation comes from the coordinate transformation
instead of the point cloud representation. Hence, a coordinate
map $M ∈ R^{H×W×3}$ where each pixel encodes a 3D coordinate can attain a comparable monocular detection result with the pseudo-LiDAR point cloud representation. This observation enables us to directly apply a 2D neural network on the coordinate
map to predict 3D objects, eliminating the need of leveraging the
time-consuming LiDAR-based detectors on point clouds.

<b> Note: Potentials and challenges of the depth-assisted approaches: </b> The depth-assisted approaches pursue more accurate
depth estimation by leveraging a pre-trained depth prediction
network. Both the depth image representation and the pseudoLiDAR presentation could significantly boost the monocular detection performance. Nevertheless, compared to the image-only
methods that only require 3D box annotations, pre-training a
depth prediction network requires expensive ground truth depth
maps, and it also hampers the end-to-end training of the whole
framework. Furthermore, pre-trained depth estimation networks
suffer from poor generalization ability. Pretrained depth maps
are usually not well calibrated on the target dataset and typically the scale needs to be adapted to the target dataset. Thus
there remains a non-negligible domain gap between the source
domain leveraged for depth pre-training and the target domain
for monocular detection. Given the fact that driving scenarios are
normally diverse and complex, pre-training depth networks on a
restricted domain may not work well in real-world applications.

### 3.1.3. Prior-guided monocular 3D object detection

Numerous approaches try to tackle the ill-posed monocular 3D
object detection problem by leveraging the hidden prior knowledge of object shapes and scene geometry from images. The
prior knowledge can be learned by introducing pre-trained subnetworks or auxiliary tasks, and they can provide extra information or constraints to help accurately localize 3D objects. The
broadly adopted prior knowledge includes object shapes, geometry consistency, temporal constraints, and segmentation information.

<b> Object shapes: </b> Many methods resort to shape reconstruction
of 3D objects directly from images. The reconstructed shapes
can be further leveraged to determine the locations and poses of the 3D objects. There are 5 types of reconstructed representations: computer-aided design (CAD) models, wireframe models,
signed distance function (SDF), points, and voxels.

Some papers learn morphable wireframe models to represent 3D objects. Other works leverage DeepSDF to learn implicit signed distance functions
or low-dimensional shape parameters from CAD models, and
they further propose a render-and-compare approach to learn the
parameters of 3D objects. Some works utilize voxel
patterns to represent 3D objects. Other papers resort
to point cloud reconstruction from images and estimate the locations of 3D objects with 2D-3D correspondences.


<b> Geometric consistency: </b> Given the extrinsics matrix $T ∈ SE(3)$
that transforms a 3D coordinate in the object frame to the camera
frame, and the camera intrinsics matrix K that project the 3D
coordinate onto the image plane, the projection of a 3D point
$[x, y, z]$ in the object frame into the image pixel coordinate $[u, v]$
can be represented as

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-40.png?raw=true)

where $d$ is the depth of transformed 3D coordinate in the camera
frame. Eqn. above provides a geometric relationship between 3D
points and 2D image pixel coordinates, which can be leveraged
in various ways to encourage consistency between the predicted
3D objects and the 2D objects on images. There are mainly 5
types of geometric constraints in monocular detection: 2D-3D
boxes consistency, keypoints, object’s height-depth relationship,
inter-objects relationship, and ground plane constraints.

Some works propose to encourage the
consistency between 2D and 3D boxes by minimizing reprojection errors. These methods introduce a post-processing step to
optimize the 3D object parameters by gradually fitting the projected 3D boxes to 2D bounding boxes on images. There is also
a branch of papers that predict the object keypoints from images, and the keypoints can be leveraged to calibrate the sizes of locations of 3D objects. Object’s height-depth
relationship can also serve as a strong geometric prior. Specifically, given the physical height of an object $H$ in the 3D space,
the visual height $h$ on images, and the corresponding depth of the
object $d$, there exists a geometric constraint: $d = f · H/h$, where
$f$ is the camera focal length. This constraint can be leveraged to obtain more accurate depth estimation and has been broadly applied in a lot of works. There are also some
papers trying to model the inter-objects relationships
by exploiting new geometric relations among objects. Other papers leverage the assumption that 3D objects
are generally on the ground plane to better localize those objects.

<b> Temporal constraints </b> Temporal association of 3D objects can
be leveraged as strong prior knowledge. The temporal object
relationships have been exploited as depth-ordering and
multi-frame object fusion with a 3D Kalman filter.

<b> Segmentation: </b> Image segmentation helps monocular 3D object
detection mainly in two aspects. First, object segmentation masks
are crucial for instance shape reconstruction in some works. Second, segmentation indicates whether an image pixel is inside a 3D object from the perspective view, and this information
has been utilized to help localize 3D objects.

<b> Note: Potentials and challenges of leveraging prior knowledge in monocular 3D detection: </b> With shape reconstruction,
we could obtain more detailed object shape information from
images, which is beneficial to 3D object detection. We can also
attain more accurate detection results through the projection or
render-and-compare loss. However, there exist two challenges
for shape reconstruction applied in monocular 3D object detection. First, shape reconstruction normally requires an additional
step of pre-training a reconstruction network, which hampers
end-to-end training of the monocular detection pipeline. Second,
object shapes are generally learned from CAD models instead of
real-world instances, which imposes the challenge of generalizing the reconstructed objects to real-world scenarios.
Geometric consistencies are broadly adopted and can help
improve detection accuracy. Nevertheless, some methods formulate the geometric consistency as an optimization problem and
optimize object parameters in post-processing, which is quite
time-consuming and hampers end-to-end training.
Image segmentation is useful information in monocular 3D
detection. However, training segmentation networks requires expensive pixel annotations. Pre-training segmentation models on
external datasets will suffer from the generalization problem.

## 3.2. Stereo-based 3D object detection

<b> Problem and Challenge: </b> Stereo-based 3D object detection aims
to detect 3D objects from a pair of images. Compared to monocular images, paired stereo images provide additional geometric
constraints that can be utilized to infer more accurate depth information. Hence, the stereo-based methods generally obtain a better detection performance than the monocular-based methods. Nevertheless, stereo cameras typically require very accurate
calibration and synchronization, which are normally difficult to
achieve in real applications.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-41.png?raw=true)

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-43.png?raw=true)

<b> Stereo matching and depth estimation: </b> A stereo camera can
produce a pair of images, i.e. the left image $I_L$ and the right image $I_R$, in one shot. With the stereo matching techniques, a disparity map can be estimated from the paired stereo images leveraging multi-view geometry. Ideally, for each pixel
on the left image $I_{L}(u, v)$, there exists a pixel on the right image
$I_R(u, v + p)$ with the disparity value $p$ so that the two pixels
picture the same 3D location. Finally, the disparity map can be
transformed into a depth image with the following formula:

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-42.png?raw=true)

where $d$ is the depth value, $f$ is the focal length, and b is the
baseline length of the stereo camera. The pixel-wise disparity
constraints from stereo images enable more accurate depth estimation compared to monocular depth prediction.

<b> 2D-detection based methods: </b> Conventional 2D object detection
frameworks can be modified to resolve the stereo detection problem. Specifically, paired stereo images are passed through an
image-based detector with Siamese backbone networks to generate left and right regions of interest (RoIs) for the left and right
images respectively. Then in the second stage, the left and right
RoIs are fused to estimate the parameters of 3D objects. Stereo
R-CNN first proposes to extend 2D detection frameworks
to stereo 3D detection. This design paradigm has been adopted
in numerous papers. [Paper](https://arxiv.org/abs/1906.01193) proposes a novel stereo triangulation 
learning sub-network at the second stage;
learn instance-level disparity by object-centric stereo matching
and instance segmentation; Ida-3d proposes adaptive instance disparity estimation; introduce single-stage stereo detection frameworks; propose an energy-based framework
for stereo-based 3D object detection.

<b> Pseudo-LiDAR based methods: </b> The disparity map predicted
from stereo images can be transformed into the depth image and
then converted into the pseudo-LiDAR point cloud. Hence, similar to the monocular detection methods, the pseudo-LiDAR representation can also be employed in stereo-based 3D object detection methods. Those methods try to improve the disparity estimation in stereo matching for more accurate depth prediction. Some paper introduces a depth cost volume in stereo matching networks; Some paper proposes an end-to-end stereo matching and detection framework; [116, 128] leverage semantic segmentation and
predict disparity for foreground and background regions separately; Some paper proposes a Wasserstein loss for disparity estimation.

<b> Volume-based methods: </b> There exists a category of methods that
skip the pseudo-LiDAR representation and perform 3D object
detection directly on 3D stereo volumes. DSGN proposes
a 3D geometric volume derived from stereo matching networks
and applies a grid-based 3D detector on the volume to detect 3D
objects. Some paper improve DSGN by leveraging knowledge
distillation and 3D feature volumes respectively

<b> Note: Potentials and challenges of the stereo-based methods: </b> Compared to the monocular detection methods, the stereo-based methods can obtain more accurate depth and disparity estimation with
stereo matching techniques, which brings a stronger object localization ability and significantly boosts the 3D object detection
performance. Nevertheless, an auxiliary stereo matching network
brings additional time and memory consumption. Compared to
LiDAR-based 3D object detection, detection from stereo images
can serve as a much cheaper solution for 3D perception in autonomous driving scenarios. However, there still exists a nonnegligible performance gap between the stereo-based and the
LiDAR-based 3D object detection approaches.

## 3.3.  Multi-view 3D object detection

<b> Problem and Challenge: </b> Autonomous vehicles are generally
equipped with multiple cameras to obtain complete environmental information from multiple viewpoints. Recently, multi-view
3D object detection has evolved rapidly. Some multi-view 3D
detection approaches try to construct a unified BEV space by
projecting multi-view images into the bird’s-eye view, and then
employ a BEV-based detector on top of the unified BEV feature map to detect 3D objects. The transformation from camera views to the bird’s-eye view is ambiguous without accurate
depth information, so image pixels and their BEV locations are
not perfectly aligned. How to build reliable transformations from
camera views to the bird’s-eye view is a major challenge in these
methods. Other methods resort to 3D object queries that are generated from the bird’s-eye view and Transformers where crossview attention is applied to object queries and multi-view image
features. The major challenge is how to properly generate 3D
object queries and design more effective attention mechanisms
in Transformers.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-44.png?raw=true)

<b> BEV-based multi-view 3D object detection: </b> LSS is a pioneering work that proposes a lift-splat-shoot paradigm to solve
the problem of BEV perception from multi-view cameras. There
are three steps in LSS. 

- Lift: bin-based depth prediction is conducted on image pixels and multi-view image features are lifted to 3D frustums with depth bins. 

- Splat: 3D frustums are splatted into a unified bird’s-eye view plane and image features are
transformed into BEV features in an end-to-end manner. 

- Shoot: downstream perception tasks are performed on top of the BEV
feature map. This paradigm has been successfully adopted by
many following works. 

BEVDet improves LSS
with a four-step multi-view detection pipeline, where the image
view encoder encodes features from multi-view images, the view
transformer transforms image features from camera views to the
bird’s-eye view, the BEV encoder further encodes the BEV features, and the detection head is employed on top of the BEV
features for 3D detection. The major bottleneck is
depth prediction, as it is normally inaccurate and will result in
inaccurate feature transforms from camera views to the bird’seye view. To obtain more accurate depth information, many papers resort to mining additional information from multi-view images and past frames, e.g. leverages explicit depth supervision, introduces surround-view temporal stereo, 
uses dynamic temporal stereo, combines both short-term
and long-term temporal stereo for depth prediction. 

In addition,
there are also some papers that completely abandon
the design of depth bins and categorical depth prediction. They
simply assume that the depth distribution along the ray is uniform, so the camera-to-BEV transformation can be conducted
with higher efficiency.

<b> Query-based multi-view 3D object detection: </b> In addition to
the BEV-based approaches, there is also a category of methods
where object queries are generated from the bird’s-eye view and
interact with camera view features. Inspired by the advances in
Transformers for object detection, DETR3D introduces a sparse set of 3D object queries, and each query corresponds to a 3D reference point. The 3D reference points can
collect image features by projecting their 3D locations onto the
multi-view image planes and then object queries interact with
image features through Transformer layers. Finally, each object
query will decode a 3D bounding box. Many following papers
try to improve this design paradigm, such as introducing spatiallyaware cross-view attention and adding 3D positional embeddings on top of image features. BEVFormer introduces dense grid-based BEV queries and each query corresponds to a pillar that contains a set of 3D reference points.
Spatial cross-attention is applied to object queries and sparse
image features to obtain spatial information, and temporal selfattention is applied to object queries and past BEV queries to
fuse temporal information.

# 4. Multi-Modal 3D Object Detection

In this section, we introduce the multi-modal 3D object detection
approaches that fuse multiple sensory inputs. According to the
sensor types, the approaches can be divided into three categories: LiDAR-camera, radar, and map fusion-based methods.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-45.png?raw=true)

## 4.1.  Multi-modal detection with LiDAR-camera fusion

<b> Problem and Challenge: </b> Camera and LiDAR are two complementary sensor types for 3D object detection. Cameras provide
color information from which rich semantic features can be extracted, while LiDAR sensors specialize in 3D localization and
provide rich information about 3D structures. Many endeavors
have been made to fuse the information from cameras and LiDARs for accurate 3D object detection. Since LiDAR-based detection methods perform much better than camera-based methods, the state-of-the-art approaches are mainly based on LiDARbased 3D object detectors and try to incorporate image information into different stages of a LiDAR detection pipeline. In view
of the complexity of LiDAR-based and camera-based detection
systems, combining the two modalities together inevitably brings
additional computational overhead and inference time latency.
Therefore, how to efficiently fuse the multi-modal information
remains an open challenge. 

### 4.1.1.  Early-fusion based 3D object detection

Early-fusion based methods aim to incorporate the knowledge
from images into point cloud before they are fed into a LiDAR-based detection pipeline. Hence the early-fusion frameworks are
generally built in a sequential manner: 2D detection or segmentation networks are firstly employed to extract knowledge from
images, and then the image knowledge is passed to point cloud,
and finally the enhanced point cloud is fed to a LiDAR-based
3D object detector. Based on the fusion types, the early-fusion
methods can be divided into two categories: region-level knowledge fusion and point-level knowledge fusion.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-46.png?raw=true)


<b> Region-level knowledge fusion </b> Region-level fusion methods
aim to leverage knowledge from images to narrow down the object candidate regions in 3D point cloud. Specifically, an image is
first passed through a 2D object detector to generate 2D bounding boxes, and then the 2D boxes are extruded into 3D viewing
frustums. The 3D viewing frustums are applied on LiDAR point
cloud to reduce the searching space. Finally, only the selected
point cloud regions are fed into a LiDAR detector for 3D object detection. F-PointNet first proposes this fusion mechanism, and many endeavors have been made to improve the fusion
framework. Some paper divides a viewing frustum into grid cells and
applies a convolutional network on the grid cells for 3D detection; [RoarNet](https://arxiv.org/abs/1811.03818) proposes a novel geometric agreement search;
exploits the pillar representation; [This](https://arxiv.org/abs/1803.00387) introduces a model fitting
algorithm to find the object point cloud inside each frustum.

<b> Point-level knowledge fusion: </b> Point-level fusion methods aim
to augment input point cloud with image features. The augmented
point cloud is then fed into a LiDAR detector to attain a better detection result. PointPainting is a seminal work that
leverages image-based semantic segmentation to augment point
clouds. Specifically, an image is passed through a segmentation
network to obtain pixel-wise semantic labels, and then the semantic labels are attached to the 3D points by point-to-pixel
projection. Finally, the points with semantic labels are fed into a
LiDAR-based 3D object detector. This design paradigm has been followed by a lot of papers. Apart from semantic
segmentation, there also exist some works trying to exploit other
information from images, e.g. depth image completion.

<b> Note: potentials and challenges of the early-fusion methods: </b> The early-fusion based methods focus on augmenting point
clouds with image information before they are passed through
a LiDAR 3D object detection pipeline. Most methods are compatible with a wide range of LiDAR-based 3D object detectors
and can serve as a quite effective pre-processing step to boost
detection performance. Nevertheless, the early-fusion methods
generally perform multi-modal fusion and 3D object detection
in a sequential manner, which brings additional inference latency. Given the fact that the fusion step generally requires a
complicated 2D object detection or semantic segmentation network, the time cost brought by multi-modal fusion is normally
non-negligible. Hence, how to perform multi-modal fusion efficiently at the early stage has become a critical challenge.

### 4.1.2. Intermediate-fusion based 3D object detection

Intermediate-fusion based methods try to fuse image and LiDAR
features at the intermediate stages of a LiDAR-based 3D object
detector, e.g. in backbone networks, at the proposal generation
stage, or at the RoI refinement stage. These methods can also
be classified according to the fusion stages. 

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-47.png?raw=true)

<b> Fusion in backbone networks: </b> Many endeavors have been made
to progressively fuse image and LiDAR features in the backbone
networks. In those methods, point-to-pixel correspondences are
firstly established by LiDAR-to-camera transform, and then with
the point-to-pixel correspondences, features from a LiDAR backbone can be fused with features from an image backbone through
different fusion operators. The multi-modal fusion can be conducted in the intermediate layers of a grid-based detection backbone, with novel fusion operators such as continuous convolutions, hybrid voxel feature encoding,
and Transformer. The multi-modal fusion can also
be conducted only at the output feature maps of backbone networks, with fusion modules and operators including gated attention, unified object queries, BEV pooling, learnable alignments, point-to-ray fusion, Transformer,
and other techniques. In addition to the fusion
in grid-based backbones, there also exist some papers incorporating image information into the point-based detection backbones.

<b> Fusion in proposal generation and RoI head: </b> There exists a
category of works that conduct multi-modal feature fusion at the
proposal generation and RoI refinement stage. In those methods,
3D object proposals are first generated from a LiDAR detector,
and then the 3D proposals are projected into multiple views, i.e.
the image view and bird’s-eye view, to crop features from the
image and LiDAR backbone respectively. Finally, the cropped
image and LiDAR features are fused in an RoI head to predict
parameters for each 3D object. MV3D and AVOD are
pioneering works leveraging multi-view aggregation for multimodal detection. Other papers use the Transformer
decoder as the RoI head for multi-modal feature fusion.

<b> Note: potentials and challenges of the intermediate-fusion methods: </b> The intermediate methods encourage deeper integration of multi-modal representations and yield 3D boxes of higher
quality. Nevertheless, camera and LiDAR features are intrinsically heterogeneous and come from different viewpoints, so there
still exist some problems on the fusion mechanisms and view
alignments. Hence, how to fuse the heterogeneous data effectively and how to deal with the feature aggregation from multiple
views remain a challenge to the research community.

### 4.1.3. Late-fusion based 3D object detection

<b> Fusion at the box level: </b> Late-fusion based approaches operate
on the outputs, i.e. 3D and 2D bounding boxes, from a LiDARbased 3D object detector and an image-based 2D object detector respectively. In those methods, object detection with
camera and LiDAR sensor can be conducted in parallel, and the
output 2D and 3D boxes are fused to yield more accurate 3D
detection results. CLOCs introduces a sparse tensor that
contains paired 2D-3D boxes and learns the final object confidence scores from this sparse tensor. CLOCs improved by
introducing a light-weight 3D detector-cued image detector.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-48.png?raw=true)


<b> Note: Potentials and challenges of the late-fusion methods: </b> The late-fusion based approaches focus on the instancelevel aggregation and perform multi-modal fusion only on the outputs of different modalities, which avoids complicated interactions on the intermediate features or on the input point cloud.
Hence these methods are much more efficient compared to other
approaches. However, without resorting to deep features from
camera and LiDAR sensors, these methods fail to integrate rich
semantic information of different modalities, which limits the
potential of this category of methods.

## 4.2. Multi-modal detection with radar signals

<b> Problem and Challenge: </b> Radar is an important sensory type in
driving systems. In contrast to LiDAR sensors, radar has four irreplaceable advantages in real-world applications: Radar is much
cheaper than LiDAR sensors; Radar is less vulnerable to extreme
weather conditions; Radar has a larger detection range; Radar
provides additional velocity measurements. Nevertheless, compared to LiDAR sensors that generate dense point clouds, radar
only provides sparse and noisy measurements. Hence, how to
effectively handle the radar signals remains a critical challenge.

<b> Radar-LiDAR fusion: </b> Many papers try to fuse the two modalities by introducing new fusion mechanisms to enable message
passing between the radar and LiDAR signals, including voxelbased fusion, attention-based fusion, introducing a
range-azimuth-doppler tensor\, leveraging graph neural networks, exploiting dynamic occupancy maps, and introducing 4D radar data.

<b> Radar-camera fusion: </b> Radar-camera fusion is quite similar to
LiDAR-camera fusion, as both radar and LiDAR data are 3D
point representations. Most radar-camera approaches adapt the existing LiDAR-based detection architectures to
handle sparse radar points and adopt similar fusion strategies as
LiDAR-camera based methods.

## 4.3. Multi-modal detection with high-definition maps

<b> Problem and Challenge: </b> High-definition maps (HD maps) contain detailed road information such as road shape, road marking,
traffic signs, barriers, etc. HD maps provide rich semantic information on surrounding environments and can be leveraged as a
strong prior to assist 3D object detection. How to effectively incorporate map information into a 3D object detection framework
has become an open challenge to the research community.

<b> Multi-modal detection with map information: </b> High-definition
maps can be readily transformed into a bird’s-eye view representation and fused with rasterized BEV point clouds or feature
maps. The fusion can be conducted by simply concatenating the
channels of a rasterized point cloud and an HD map from the bird’s-eye view, feeding LiDAR point cloud and HD map
into separate backbones and fusing the output feature maps of
the two modalities, or simply filtering out those predictions
that do not fall into the relevant map regions. Other map
types have also been explored, e.g. visibility map, vectorized map.

# 5. Transformer-based 3D Object Detection

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-51.png?raw=true)

## 5.1. Transformer architectures for 3D object detection

<b> Problem and Challenge: </b> While most 3D object detectors are
based on convolutional architectures, recently Transformer-based
3D detectors have shown great potential and dominated 3D object detection leaderboards. Compared to convolutional networks,
the query-key-value design in Transformers enables more flexible interactions between different representations and the selfattention mechanism results in a larger receptive field than convolutions. However, fully-connected self-attention has quadratic
time and space complexity w.r.t. the number of inputs, training
Transformers can easily fall into sub-optimal results when the
data size is small. Hence, it’s critical to define proper query-keyvalue triplets and design specialized attention mechanisms for
Transformer-based 3D object detectors.

<b> Transformer architectures: </b> The development of Transformer
architectures in 3D object detection has experienced three stages:

(1) Inspired by vanilla Transformer, new Transformer modules with special attention mechanisms are proposed to obtain
more powerful features in 3D object detection. 

(2) Inspired by
DETR, query-based Transformer encoder-decoder designs
are introduced to 3D object detectors. 

(3) Inspired by ViT,
patch-based inputs and architectures similar to Vision Transformers are introduced in 3D object detection.

In the first stage, many papers try to introduce novel Transformer modules into conventional 3D detection pipelines. In these
papers, the choices of query, key, and value are quite flexible and
new attention mechanisms are proposed. Pointformer introduces Transformer modules to point backbones. It takes point
features and coordinates as queries and applies self-attention to
a group of point clouds. Voxel Transformer replaces convolutional voxel backbones with Transformer modules, where sparse and submanifold voxel attention are proposed and applied
to voxels. CT3D proposes a novel Transformer-based detection head, where proposal-to-point attention and channel-wise
attention are introduced.

In the second stage, many papers propose DETR-like architectures for 3D object detection. They leverage a set of object
queries and use those queries to interact with different features
to predict 3D boxes. DETR3D introduces object queries
and generates a 3D reference point for each query. They use reference points to aggregate multi-view image features as keys and
values, and apply cross-attention between object queries and image features. Finally, each query can decode a 3D bounding box
for detection. Many following works have adopted the design of
object queries and reference points. BEVFormer generates
dense queries from BEV grids. TransFusion produces object
queries from initial detections and applies cross-attention to LiDAR and image features in a Transformer decoder. UVTR
fuses object queries with image and LiDAR voxels in a Transformer decoder. FUTR3D fuses object queries with features
from different sensors in a unified way.
In the third stage, many papers try to apply the designs of
Vision Transformers to 3D object detectors. Following [63, 168],
they split inputs into patches and apply self-attention within each
patch and across different patches. SST proposes a sparse
Transformer, in which voxels in a local region are grouped into
a patch and sparse regional attention is applied to the voxels in a
patch, and then region shift is applied to change the grouping so
new patches can be generated. SWFormer improves
with multi-scale feature fusion and voxel diffusion.

## 5.2. Transformer applications in 3D object detection

Applications of Transformer-based 3D detectors. Transformer
architectures have been broadly adopted in various types of 3D
object detectors. For point-based 3D object detectors, a pointbased Transformer has been developed to replace the conventional PointNet backbone. For voxel-based 3D detectors, a
lot of papers propose novel voxel-based Transformers to replace the conventional convolutional backbone. For
point-voxel based 3D object detectors, a new Transformer-based
detection head has been proposed for better proposal refinement. For monocular 3D object detectors, Transformers can
be used to fuse image and depth features. For multi-view
3D object detectors, Transformers are utilized to fuse multi-view
image features for each query. For multi-modal 3D
object detectors, many papers leverage Transformer
architectures and special cross-attention mechanisms to fuse features of different modalities. For temporal 3D object detectors,
Temporal-Channel Transformer is proposed to model temporal relationships across LiDAR frames.

# 6. Temporal 3D Object Detection

In this section, we introduce the temporal 3D object detection
methods. Based on the data types, these methods can be divided
into three categories: detection from LiDAR sequences, detection from streaming inputs, and detection from videos. 

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-49.png?raw=true)


## 6.1. 3D object detection from LiDAR sequences

<b> Problem and Challenge: </b> While most methods focus on detection from a single-frame point cloud, there also exist many approaches leveraging multi-frame point clouds for more accurate
3D object detection. These methods are trying to tackle the temporal detection problem by fusing multi-frame features via various temporal modeling tools, and they can also obtain more complete 3D shapes by merging multi-frame object points into a single frame. Temporal 3D object detection has exhibited great success in offline 3D auto-labeling pipelines. However, in onboard applications, these methods still suffer from memory and latency issues, as processing multiple frames inevitably brings additional time and memory costs, which can become severe when
models are running on embedded devices.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-50.png?raw=true)

<b> 3D object detection from sequential sweeps: </b> Most detection
approaches using multi-frame point clouds resort to proposallevel temporal information aggregation. Namely, 3D object proposals are first generated independently from each frame of point
cloud through a shared detector, and then various temporal modules are applied on the object proposals and the respective RoI
features to aggregate the information of objects across different frames. The adopted temporal aggregation modules include
temporal attention, ConvGRU, graph network,
LSTM, and Transformer. Temporal 3D object detection is also applied in the 3D object auto-labeling pipelines. In addition to temporal detection from multi-frame point
clouds, there are also some works leveraging sequential range images for 3D object detection.

## 6.2. 3D object detection from streaming data

<b> Problem and Challenge: </b> Point clouds collected by rotating LiDARs are intrinsically a streaming data source in which LiDAR
packets are sequentially recorded in a sweep. It typically takes
50-100 ms for a rotating LiDAR sensor to generate a 360◦
complete LiDAR sweep, which means that by the time a point cloud
is produced, it no longer accurately reflects the scene at the exact time. This poses a challenge to autonomous driving applications which generally require minimal reaction times to guarantee driving safety. Many endeavors have been made to directly
detect 3D objects from the streaming data. These methods generally detect 3D objects on the active LiDAR packets immediately without waiting for the full sweep to be built. Streaming
3D object detection is a more accurate and low-latency solution
to vehicle perception compared to detection from full LiDAR
sweeps.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-52.png?raw=true)

<b> Streaming 3D object detection: </b> Similar to temporal detection
from multi-frame point clouds, streaming detection methods can treat each LiDAR packet as an independent sample to detect 3D objects and apply temporal modules on the sequential
packets to learn the inter-packets relationships. However, a LiDAR packet normally contains an incomplete point cloud and
the information from a single packet is generally not sufficient
for accurately detecting 3D objects. To this end, some papers
try to provide more context information for detection in a single packet. The proposed techniques include a spatial memory
bank and a multi-scale context padding scheme.

## 6.3. 3D object detection from videos

<b> Problem and Challenge: </b> Video is an important data type and
can be easily obtained in autonomous driving applications. Compared to single-image based 3D object detection, video-based
3D detection naturally benefits from the temporal relationships
of sequential images. While numerous works focus on singleimage based 3D object detection, only a few papers investigate
the problem of 3D object detection from videos, which leaves an
open challenge to the research community.

<b> Video-based 3D object detection: </b> Video-based detection approaches generally extend the image-based 3D object detectors
by tracking and fusing the same objects across different frames.
The proposed trackers include LSTM [100] and the 3D Kalman
filter. In addition, there are some works leveraging both videos and multi-frame point clouds for more accurate
3D object detection. Those methods propose 4D sensor-time fusion to learn features from both temporal and multi-modal data.

# 7. Label-Efficient 3D Object Detection

In this section, we introduce the methods of label-efficient 3D
object detection. In previous sections, we generally assume the
3D detectors are trained under full supervision on a specific data
domain and with a sufficient amount of annotations. However,
in real-world applications, the 3D object detection methods inevitably face the problems of poor generalizability and lacking
annotations. To address these issues, label-efficient techniques
can be employed in 3D object detection, including domain adaptation, weakly-supervised learning,
semi-supervised learning, and self-supervised learning for 3D object detection.

## 7.1. Domain adaptation for 3D object detection

<b> Problem and Challenge: </b> Domain gaps are ubiquitous in the
data collection process. Different sensor settings and placements,
different geographical locations, and different weathers will result in completely different data domains. In most conditions,
3D object detectors trained on a certain domain cannot perform
well on other domains. Many techniques have been proposed to
address the domain adaptation problem for 3D object detection,
e.g. leveraging consistency between source and target domains,
and self-training on target domains. Nevertheless, most methods
only focus on solving one specific domain transfer problem. Designing a domain adaptation approach that can be generally applied in any domain transfer tasks in 3d object detection will be a
promising research direction.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-53.png?raw=true)

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-54.png?raw=true)

<b> Cross-sensor domain adaptation: </b> Different datasets have different sensory settings, e.g. a 32-beam LiDAR sensor used in
nuScenes versus a 64-beam LiDAR sensor in KITTI,
and the data is also collected at different geographic locations,
e.g. KITTI is collected in Germany while Waymo is
collected in United States. These factors will lead to severe domain gaps between different datasets, and the detectors trained
on a dataset generally exhibit quite poor performance when they
are tested on other datasets. [This](https://arxiv.org/abs/2005.08139) is a notable work that observes
the domain gaps between datasets, and they introduce a statistic normalization approach to handle the gaps. Many following
works leverage self-training to resolve the domain adaptation
problem. In those methods, a detector pre-trained on the source
dataset will produce pseudo labels for the target dataset, and then
the detector is re-trained on the target dataset with pseudo labels. These methods make improvements mainly on obtaining
pseudo labels of higher quality, e.g. proposes a scale-anddetect strategy, introduces a memory bank, leverages
the scene flow information, and exploits playbacks to enhance the quality of pseudo labels. In addition to the self-training
approaches, there also exist some papers building alignments between source and target domains. The domain alignments can
be established through a scale-aware and range-aware alignment
strategy, multi-level consistency, and a contrastive
co-training scheme.

In addition to the domain gaps among datasets, different sensors also produce data of distinct characteristics. A 32-beam LiDAR produces much sparser point clouds compared to a 64-
beam LiDAR, and images obtained from different cameras also have diverse sizes and intrinsics. [This](https://ieeexplore.ieee.org/document/8814047) introduces a multi-task
learning scheme to tackle the domain gaps between different LiDAR sensors, and [this](https://arxiv.org/abs/2108.07142) proposes the position-invariant transform
to address the domain gaps between different cameras.

<b> Cross-weather domain adaptation: </b> Weather conditions have a
huge impact on the quality of collected data. On rainy days, raindrops will change the surface property of objects so that fewer
LiDAR beams can be reflected and detected, so point clouds collected on rainy days are much sparser than those obtained under
dry weather. Besides fewer reflections, rain also causes false positive reflections from raindrops in mid-air. [This](https://arxiv.org/abs/2108.07142) addresses the
cross-weather domain adaptation problem with a novel semantic
point generation scheme.

<b> Sim-to-real domain adaptation: </b> Simulated data has been broadly
adopted in 3D object detection, as the collected real-world data
cannot cover all driving scenarios. However, the synthetic data
has quite different characteristics from the real-world data, which
gives rise to a sim-to-real adaptation problem. Many approaches
are proposed to resolve this problem, including GAN based
training and introducing an adversarial discriminator
to distinguish real and synthetic data.

## 7.2. Weakly-supervised learning for 3D object detection

<b> Problem and Challenge: </b> Existing 3D object detection methods
highly rely on training with vast amounts of manually labeled 3D
bounding boxes, but annotating those 3D boxes is quite laborious and expensive. Weakly-supervised learning can be a promising solution to this problem, in which weak supervisory signals,
e.g. less expensive 2D annotations, are exploited to train the 3D
object detection models. Weakly-supervised 3D object detection
requires fewer human efforts for data annotation, but there still
exists a non-negligible performance gap between the weaklysupervised and the fully-supervised methods.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-55.png?raw=true)

<b> Weakly-supervised 3D object detection: </b> Weakly-supervised approaches leverage weak supervision instead of fully annotated
3D bounding boxes to train 3D object detectors. The weak supervisions include 2D image bounding boxes, a pretrained image detector, BEV object centers and vehicle instances. Those methods generally design novel learning mechanisms to skip the 3D box supervision and learn to detect 3D objects by mining useful information from weak signals.

## 7.3. Semi-supervised 3D object detection

<b> Problem and Challenge: </b> In real-world applications, data annotation requires much more human effort than data collection.
Typically a data acquisition vehicle can collect more than 100k
frames of point clouds in a day, while a skilled human annotator can only annotate 100-1k frames per day. This will inevitably lead to a rapid accumulation of a large amount of unlabeled data. Hence how to mine useful information from largescale unlabeled data has become a critical challenge to both the
research community and the industry. Semi-supervised learning, which exploits a small amount of labeled data and a huge
amount of unlabeled data to jointly train a stronger model, is a
promising direction. Combining 3D object detection with semisupervised learning can boost detection performance. 

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-56.png?raw=true)

<b> Semi-supervised 3D object detection: </b> There are mainly two
categories of approaches in semi-supervised 3D object detection:
pseudo-labeling and teacher-student learning. The pseudo labeling approaches [18, 284] first train a 3D object detector with the
labeled data, and then use the 3D detector to produce pseudo
labels for the unlabeled data. Finally, the 3D object detector is
re-trained with the pseudo labels on the unlabeled domain. The
teacher-student methods adapt the Mean Teacher
training paradigm to 3D object detection. Specifically, a teacher
detector is first trained on the labeled domain, and then the teacher
detector guides the training of a student detector on the unlabeled
domain by encouraging the output consistencies between the two
detection models.

## 7.4. Self-supervised 3D object detection

<b> Problem and Challenge: </b> Self-supervised pre-training has become a powerful tool when there exists a large amount of unlabeled data and limited labeled data. In self-supervised learning, models are first pre-trained on large-scale unlabeled data
and then fine-tuned on the labeled set to obtain a better performance. In autonomous driving scenarios, self-supervised pretraining for 3D object detection has not been widely explored.
Existing methods are trying to adapt the self-supervised methods, e.g. contrastive learning, to the 3D object detection problem, but the rich semantic information in multi-modal data has
not been well exploited. How to effectively handle the raw point
clouds and images to pre-train an effective 3D object detector
remains an open challenge.

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-57.png?raw=true)

<b> Self-supervised 3D object detection: </b> Self-supervised methods
generally apply the contrastive learning techniques to
3D object detection. Specifically, an input point cloud is first
transformed into two views with augmentations, and then contrastive learning is employed to encourage the feature consistencies of the same 3D locations across the two views. Finally,
the 3D detector pre-trained with contrastive learning is further
fine-tuned on the labeled set to attain better performance. PointContrast first introduces the contrastive learning paradigm
in 3D object detection, and the following papers improve this
paradigm by leveraging the depth information and clustering. In addition to self-supervised learning for point
cloud detectors, there are also some works trying to exploit both
point clouds and images for self-supervised 3D detection, e.g. proposes an intra-modal and inter-modal contrastive learning scheme on the multi-modal inputs

# 8. Analysis and Outlooks

## 8.1. Research trends

### 8.1.1. Trends of dataset selection

Before 2018, most methods were evaluated on the KITTI dataset,
and the evaluation metric they adopted is 2D average precision
(AP2D), where they project the 3D bounding boxes into the image plane and compare them with the ground truth 2D boxes.
From 2018 until now, more and more papers have adopted the 3D
or BEV average precision (AP3D or APBEV ), which is a more
direct metric to measure 3D detection quality. For the LiDARbased methods, the detection performances on KITTI quickly
get converged over the years, e.g. AP3D of easy cases increases
from 71.40% to 90.90%, and even AP3D of hard
cases reaches 79.14%. Therefore, since 2019, more and
more LiDAR-based approaches have turned to larger and more
diverse datasets, such as the nuScenes dataset and the Waymo
Open dataset. Large-scale datasets also provide more useful data
types, e.g. raw range images provided by Waymo facilitate the
development of range-based methods. For the camera-based detection methods, AP3D of monocular detection on KITTI increases from 1.32% to 23.22%, leaving huge room
for improvement. Until now, only a few monocular methods have
been evaluated on the Waymo dataset. For the multi-modal detection approaches, the methods before 2019 are mostly tested
on the KITTI dataset, and after that most papers resort to the
nuScenes dataset, as it provides more multi-modal data.

### 8.1.2. Trends of inference time

PointPillars has achieved remarkable inference speed with
only 16ms latency, and its architecture has been adopted by many
following works. However, even with the emergence of more powerful hardware, the inference speed didn’t exhibit a significant improvement over the years. This is mainly
because most methods focus on performance improvement and
pay less attention to efficient inference. Many papers have introduced new modules into the existing detection pipelines, which
also brings additional time costs. For the pseudo-LiDAR based
detection methods, the stereo-based methods, and most multimodal methods, the inference time is generally more than 100 ms, which cannot satisfy the real-time requirement and hampers
the deployment in real-world applications.

### 8.1.3. Trends of the LiDAR-based methods

LiDAR-based 3D object detection has witnessed great advances
in recent years. Among the LiDAR-based methods, the voxelbased and point-voxel based detection approaches attain superior performances.
The pillar-based detection methods are extremely fast. The range-based and BEV-based approaches are also quite efficient. The point-based detectors
can obtain a good performance, but their inference speeds are
greatly influenced by the choices of sampling and operators.

For point-based 3D object detectors, moderate AP has been
increasing from 53.46% to 79.57% on the KITTI
benchmark. The performance improvements are mainly owing
to two factors: more robust point cloud samplers and more powerful point cloud operators. The development of point cloud samplers starts with Farthest Point Sampling (FPS), and
many following point cloud detectors have been improving point
cloud samplers based on FPS, including fusion-based FPS ,
target-based FPS, FPS with coordinates refinement.
A good point cloud sampler could produce candidate points that
have better coverage of the whole scene, so it avoids missing
detections when the point cloud is sparse, which helps improve
the detection performance. Besides point cloud samplers, point
cloud operators have also progressed rapidly, from the standard
set abstraction to graph operators
and Transformers. Point cloud operators are crucial for
extracting powerful feature representations from point clouds.
Hence powerful point cloud operators can help detectors better
obtain semantic information about 3D objects and improve performance.

For grid-based 3D object detectors, moderate AP has been
increasing from 50.81% to 82.09% on the KITTI
benchmark. The performance improvements are mainly driven
by better backbone networks and detection heads. The development of backbone networks has experienced four stages: 

(1)
2D networks to process BEV images that are generated by point
cloud projection [10, 335], 

(2) 2D networks to process pillars
that are generated by PointNet encoding, 

(3) 3D sparse
convolutional networks to process voxelized point clouds,

(4) Transformer-based architectures [187, 70, 270]. The trend of
backbone designs is to encode more 3D information from point
clouds, which leads to more powerful BEV representations and
better detection performance, but those early designs are still
popular due to efficiency. Detection head designs have experienced the transition from anchor-based heads to centerbased heads, and the object localization ability has been
improved with the development of detection heads. Other head
designs such as IoU rectification and sequential head
can further boost performance.

For point-voxel based 3D object detectors, moderate AP has
been increasing from 75.73% to 82.08% on the KITTI
benchmark. The performance improvements come from more
power operators and modules that
can effectively fuse point and voxel features.

For range-based 3D object detectors, L1 mAP has been increasing from 52.11% to 78.4% on the Waymo Open
dataset. The performance improvements come from designs of
specialized operators that can handle range images
more effectively, as well as view transforms and multi-view aggregation.


### 8.1.4. Trends of the camera-based methods

Camera-based 3D object detection has shown rapid progress recently. Among the camera-based methods, the stereo-based detection methods generally outperform the monocular detection
approaches by a large margin. This is mainly because depth and disparity estimated from stereo images are much more accurate than
those estimated from monocular images, and accurate depth estimation is the most important factor in camera-based 3D object
detection. Multi-camera 3D object detection has been progressing fast with the emergence of BEV perception and Transformers. State-of-the-art method attains 54.0% mAP and 61.9
NDS on nuScenes, which has outperformed some prestigious
LiDAR-based 3D object detectors.

For monocular 3D object detectors, moderate AP has been
increasing from 1.51% to 16.34% on the KITTI
benchmark. The major challenge of monocular 3D object detection is how to obtain accurate 3D information from a single
2D image, as localization errors dominate detection errors. The
performance improvements are driven by more accurate depth
prediction, which can be achieved by better network architecture designs, leveraging depth images
or pseudo-LiDAR point clouds, introducing geometry constraints, and 3D object reconstruction.

For stereo-based 3D object detectors, moderate AP has been
increasing from 4.37% to 64.66% on the KITTI benchmark. The performance improvements mainly come from better
network designs and data representations. Early works rely
on stereo-based 2D detection networks to produce paired object bounding boxes and then predict object-centric stereo/depth
information with a sub-network. However, those object-centric
methods generally lack global disparity information which hampers accurate 3D detection in a scene. Later on, pseudo-LiDAR
based approaches generate disparity maps from stereo images and then transform disparity maps into 3D pseudo-LiDAR
point clouds that are finally passed to a LiDAR detector to perform 3D detection. The transformation from 2D disparity maps
to 3D point clouds is crucial and can significantly boost 3D detection performance. Many following papers are based on the
pseudo-LiDAR paradigm and improve it with stronger stereo
matching network and end-to-end training of stereo matching and LiDAR detection. Recent methods transforms disparity maps into 3D volumes and apply grid-based detectors on the volumes, which results in better performance.

For multi-view 3D object detection, mAP has been increasing from 41.2% to 54.0% on the nuScenes dataset.
For BEV-based approaches, the performance improvements are
mainly from better depth prediction. More accurate
depth information results in more accurate camera-to-BEV transformation so detection performance can be improved. For querybased methods, the performance improvements come from better designs of 3D object queries, more powerful image features, and new attention mechanisms.

### 8.1.5. Trends of the multi-modal methods

The multi-modal methods generally exhibit a performance improvement over the single-modal baselines but at the cost of
introducing additional inference time. For instance, the multimodal detector outperforms the LiDAR baseline by
8.8% mAP on nuScenes, but the inference time of also increases to 542 ms compared to the baseline 70 ms. The problem
can be more severe in the early-fusion based approaches, where
the 2D networks and the 3D detection networks are connected
in a sequential manner. Most multi-modal detection methods are
designed and tested on the KITTI dataset, in which only a frontview image and the corresponding point cloud are utilized. Recently more and more methods are proposed and evaluated on
the nuScenes dataset, in which multi-view images, point clouds,
and high-definition maps are provided.

For early-fusion based methods, moderate AP increases from
70.39% to 76.51% on the KITTI benchmark, and
mAP increases from 46.4% to 66.8% on nuScenes
dataset. There are two crucial factors that contribute to the performance increase: knowledge fusion and data augmentation.
From the results, we can observe that point-level knowledge fusion is generally more effective than region-level fusion. This is because region-level knowledge fusion
simply reduces the detection range, while point-level knowledge
fusion can provide fine-grained semantic information which is
more beneficial in 3D detection. Besides, consistent data augmentations between point clouds and images can also significantly boost detection performance.

For intermediate and late fusion based methods, moderate
AP increases from 62.35% to 80.67% on the KITTI
benchmark, and mAP increases from 52.7% to 69.2%
on the nuScenes dataset. Most methods focus on three critical
problems: where to fuse different data representations, how to
fuse these representations, and how to build reliable alignments
between points and image pixels. For the where-to-fuse problem,
different approaches try to fuse image and LiDAR features at
different places, e.g. 3D backbone networks, BEV feature maps,
RoI heads, and outputs. From the results we can observe that
fusion at any place can boost detection performance over singlemodality baselines, and fusion in the BEV space
is more popular recently for its performance and efficiency. For
the how-to-fuse problem, the development of fusion operators
has experienced simple concatenation, continuous convolutions, attention, and Transformers, and fusion with Transformers exhibit prominent performance on all benchmarks. For the point-to-pixel alignment problem, most papers reply on fixed extrinsics and intrinsics to construct point-to-pixel correspondences. However, due to occlusion and calibration errors, those correspondences can be noisy
and misalignment will harm performance. Recent works
circumvent this problem by directly fusing camera and LiDAR
BEV feature maps, which is more robust to noise.

### 8.1.6. Systematic comparisons

Considering all the input sensors and modalities, LiDAR-based
detection is the best solution to the 3D object detection problem,
in terms of both speed and accuracy. For instance, achieves 80.28% moderate AP3D and still runs at 30 FPS on KITTI.
Multi-modal detection is built upon LiDAR-based detection, and
can obtain a better detection performance compared to the LiDAR baselines, becoming state-of-the-art in terms of accuracy.
Camera-based 3D object detection is a much cheaper and quite
efficient solution in contrast to LiDAR and multi-modal detection. Nevertheless, the camera-based methods generally have a
worse detection performance due to inaccurate depth predictions
from images. The state-of-the-art monocular and stereo
detection approach only obtain 16.34% and 64.66% moderate
AP3D respectively on KITTI. Recent advances in multi-view 3D
object detection are quite promising. The state-of-the-art
achieves 54.0% mAP on nuScenes, which could perform on par
with some classic LiDAR detectors. In conclusion, LiDARbased and multi-modal detectors are the best solutions considering speed and accuracy as the dominant factors, while camerabased detectors can be the best choice considering cost as the
most important factor, and multi-view 3D detectors are becoming promising and may outperform LiDAR detectors in the future.

## 8.2. Future outlooks

### 8.2.1. Open-set 3D object detection

Nearly all existing works are proposed and evaluated on close
datasets, in which the data only covers limited driving scenarios
and the annotations only include basic classes, e.g. cars, pedestrians, cyclists. Although those datasets can be large and diverse,
they are still not sufficient for real-world applications, in which
critical scenarios like traffic accidents and rare classes like unknown obstacles are important but not covered by the existing
datasets. Therefore, existing 3D object detectors that are trained
on the close sets have a limited capacity of dealing with those
critical scenarios and cannot identify the unknown categories. To
overcome the above limitations, designing 3D object detectors
that can learn from the open world and recognize a wide range
of object categories will be a promising research direction. [This](https://arxiv.org/pdf/2112.01135.pdf)
is a good start for open-set 3D object detection and hopefully
more methods will be proposed to tackle this problem.

### 8.2.2. Detection with stronger interpretability

Deep learning based 3D object detection models generally lack
interpretability. Namely, some important questions on how the
networks can identify 3D objects in point clouds, how occlusion and noise of 3D objects can affect the model outputs, and
how much context information is needed for detecting a 3D object, have not been properly answered due to the black-box property of deep neural networks. On the other hand, understanding
the behaviors of 3D detectors and answering these questions are
quite important if we want to perform 3D object detection in a
more robust manner and avoid those unexpected cases brought
by black-box detectors. Therefore, the methods that can understand and interpret the existing 3D object detection models will
be appealing in future research.

### 8.2.3. Efficient hardware design for 3D object detection

Most existing works focus on designing algorithms to tackle the
3D object detection problem, and their models generally run on
GPUs. Nevertheless, unlike image operators that are highly optimized for GPU devices, point clouds and voxels are sparse and
irregular, and the commonly adopted 3D operators like set abstraction or 3D sparse convolutions are not well suited for GPUs.
Hence those LiDAR object detectors cannot run as efficiently as
the image detectors on the existing hardware devices. To handle
this challenge, designing novel devices where the hardware architectures are optimized for 3D operators as well as the task of
3D object detection will be an important research direction and
will be beneficial for real-world deployment. [This](https://pointacc.mit.edu/) is a pioneering hardware work to accelerate point cloud processing, and we
believe more and more papers will come in this field. In addition,
new sensors, e.g. solid-state LiDARs, LiDARs with doppler, 4D
radars, will also inspire the design of 3D object detectors.

### 8.2.4. Detection in end-to-end self-driving systems

Most existing works treat 3D object detection as an independent
task and try to maximize the detection metrics such as average
precision. Nevertheless, 3D object detection is closely correlated
with other perception tasks as well as downstream tasks such as
prediction and planning, so simply pursuing high average precision for 3D object detection may not be optimal when considering the autonomous driving system as a whole. Therefore,
conducting 3D object detection and other tasks in an end-to-end
manner, and learning 3D detectors from the feedback of planners, will be the future research trends of 3D object detection.

# Conclusion

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-58.png?raw=true)

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-59.png?raw=true)

![img](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-7-60.png?raw=true)





