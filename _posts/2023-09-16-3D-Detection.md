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








































