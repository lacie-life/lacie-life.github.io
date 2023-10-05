---
title: Paper note 7 - PointRCNN - 3D Object Proposal Generation and Detection from Point Cloud
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-29 11:11:14 +0700
categories: [Computer Vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

[PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/pdf/1812.04244.pdf)

## 1. Comparison with state-of-the-art methods

![image](/assets/img/post_assest/paper-note-8-1.png)

In autonomous driving, the most commonly used 3D
sensors are the LiDAR sensors, which generate 3D point
clouds to capture the 3D structures of the scenes. The difficulty of point cloud-based 3D object detection mainly lies
in irregularity of the point clouds. State-of-the-art 3D detection methods either leverage the mature 2D detection
frameworks by projecting the point clouds into bird’s view(see Fig. 1 (a)), to the frontal view, or
to the regular 3D voxels, which are not optimal and
suffer from information loss during the quantization.

Instead of transforming point cloud to voxels or other
regular data structures for feature learning, Qi et al.
proposed PointNet for learning 3D representations directly
from point cloud data for point cloud classification and segmentation. As shown in Fig. 1 (b), their follow-up work
applied PointNet in 3D object detection to estimate the 3D
bounding boxes based on the cropped frustum point cloud
from the 2D RGB detection results. However, the performance of the method heavily relies on the 2D detection performance and cannot take the advantages of 3D information
for generating robust bounding box proposals.

Unlike object detection from 2D images, 3D objects in
autonomous driving scenes are naturally and well separated by annotated 3D bounding boxes. In other words, the training data for 3D object detection directly provides the semantic masks for 3D object segmentation. This is a key
difference between 3D detection and 2D detection training
data. In 2D object detection, the bounding boxes could only
provide weak supervisions for semantic segmentation 

Based on this observation, we present a novel two-stage
3D object detection framework, named PointRCNN, which
directly operates on 3D point clouds and achieves robust
and accurate 3D detection performance (see Fig. 1 (c)). 

 The
proposed framework consists of two stages, the first stage
aims at generating 3D bounding box proposal in a bottomup scheme. By utilizing 3D bounding boxes to generate
ground-truth segmentation mask, the first stage segments
foreground points and generates a small number of bounding box proposals from the segmented points simultaneously. Such a strategy avoids using the large number of 3D
anchor boxes in the whole 3D space as previous methods
do and saves much computation.

The second stage of PointRCNN conducts canonical 3D
box refinement. After the 3D proposals are generated, a
point cloud region pooling operation is adopted to pool
learned point representations from stage-1. Unlike existing
3D methods that directly estimate the global box coordinates, the pooled 3D points are transformed to the canonical coordinates and combined with the pooled point features
as well as the segmentation mask from stage-1 for learning
relative coordinate refinement. This strategy fully utilizes
all information provided by our robust stage-1 segmentation
and proposal sub-network. To learn more effective coordinate refinements, we also propose the full bin-based 3D box
regression loss for proposal generation and refinement, and
the ablation experiments show that it converges faster and
achieves higher recall than other 3D box regression loss.

Our contributions could be summarized into three-fold.

(1) We propose a novel bottom-up point cloud-based 3D
bounding box proposal generation algorithm, which generates a small number of high-quality 3D proposals via segmenting the point cloud into foreground objects and background. The learned point representation from segmentation is not only good at proposal generation but is also helpful for the later box refinement. 

(2) The proposed canonical
3D bounding box refinement takes advantages of our highrecall box proposals generated from stage-1 and learns to
predict box coordinates refinements in the canonical coordinates with robust bin-based losses. 

(3) Our proposed 3D
detection framework PointRCNN outperforms state-of-theart methods with remarkable margins and ranks first among
all published works as of Nov. 16 2018 on the 3D detection
test board of KITTI by using only point clouds as input.

## 2. PointRCNN for Point Cloud 3D Detection

### 2.1. Bottom-up 3D proposal generation via point cloud segmentation

![image](/assets/img/post_assest/paper-note-8-2.png)

Existing 2D object detection methods could be classified into one-stage and two-stage methods, where one-stage methods are generally faster but directly
estimate object bounding boxes without refinement, while
two-stage methods generate proposals firstly
and further refine the proposals and confidences in a second
stage. However, direct extension of the two-stage methods
from 2D to 3D is non-trivial due to the huge 3D search space
and the irregular format of point clouds. AVOD places
80-100k anchor boxes in the 3D space and pool features for
each anchor in multiple views for generating proposals. FPointNet generates 2D proposals from 2D images, and
estimate 3D boxes based on the 3D points cropped from the
2D regions, which might miss difficult objects that could
only be clearly observed from 3D space.

We propose an accurate and robust 3D proposal generation algorithm as our stage-1 sub-network based on wholescene point cloud segmentation. We observe that objects in
3D scenes are naturally separated without overlapping each
other. All 3D objects’ segmentation masks could be directly
obtained by their 3D bounding box annotations, i.e., 3D
points inside 3D boxes are considered as foreground points.

We therefore propose to generate 3D proposals in a
bottom-up manner. Specifically, we learn point-wise features to segment the raw point cloud and to generate 3D
proposals from the segmented foreground points simultaneously. Based on this bottom-up strategy, our method avoids
using a large set of predefined 3D boxes in the 3D space
and significantly constrains the search space for 3D proposal generation. The experiments show that our proposed
3D box proposal method achieves significantly higher recall
than 3D anchor-based proposal generation methods

<b> Learning point cloud representations: </b> To learn discriminative point-wise features for describing the raw point
clouds, we utilize the PointNet++ with multi-scale
grouping as our backbone network. There are several other
alternative point-cloud network structures, such as VoxelNet with sparse convolutions, which could
also be adopted as our backbone network.

<b> Foreground point segmentation: </b> The foreground points
provide rich information on predicting their associated objects’ locations and orientations. By learning to segment the
foreground points, the point-cloud network is forced to capture contextual information for making accurate point-wise
prediction, which is also beneficial for 3D box generation.
We design the bottom-up 3D proposal generation method
to generate 3D box proposals directly from the foreground
points, i.e., the foreground segmentation and 3D box proposal generation are performed simultaneously.

Given the point-wise features encoded by the backbone
point cloud network, we append one segmentation head
for estimating the foreground mask and one box regression
head for generating 3D proposals. For point segmentation,
the ground-truth segmentation mask is naturally provided
by the 3D ground-truth boxes. The number of foreground
points is generally much smaller than that of the background
points for a large-scale outdoor scene. Thus we use the focal loss to handle the class imbalance problem as:

![image](/assets/img/post_assest/paper-note-8-3.png)

During training point cloud segmentation, we keep the default settings $αt = 0.25$ and $γ = 2$ as the original paper.

<b> Bin-based 3D bounding box generation: </b> As we mentioned above, a box regression head is also appended for simultaneously generating bottom-up 3D proposals with the
foreground point segmentation. During training, we only
require the box regression head to regress 3D bounding box
locations from foreground points. Note that although boxes
are not regressed from the background points, those points
also provide supporting information for generating boxes
because of the receptive field of the point-cloud network.

A 3D bounding box is represented as $(x, y, z, h, w, l, θ)$
in the LiDAR coordinate system, where $(x, y, z)$ is the object center location, $(h, w, l)$ is the object size, and θ is the
object orientation from the bird’s view. To constrain the
generated 3D box proposals, we propose bin-based regression losses for estimating 3D bounding boxes of objects.

![image](/assets/img/post_assest/paper-note-8-4.png)

For estimating center location of an object, as shown in
Fig. 3, we split the surrounding area of each foreground
point into a series of discrete bins along the $X$ and $Z$ axes.
Specifically, we set a search range S for each $X$ and $Z$ axis
of the current foreground point, and each 1D search range is
divided into bins of uniform length δ to represent different
object centers $(x, z)$ on the $X-Z$ plane. We observe that using bin-based classification with cross-entropy loss for the
$X$ and $Z$ axes instead of direct regression with smooth $L1$
loss results in more accurate and robust center localization.

The localization loss for the $X$ or $Z$ axis consists of two
terms, one term for bin classification along each $X$ and $Z$
axis, and the other term for residual regression within the
classified bin. For the center location $y$ along the vertical $Y$
axis, we directly utilize smooth $L1$ loss for the regression
since most objects’ $y$ values are within a very small range.
Using the $L1$ loss is enough for obtaining accurate $y$ values.

To remove the redundant proposals, we conduct nonmaximum suppression (NMS) based on the oriented IoU
from bird’s view to generate a small number of high-quality
proposals. For training, we use 0.85 as the bird’s view IoU
threshold and after NMS we keep top 300 proposals for
training the stage-2 sub-network. For inference, we use oriented NMS with IoU threshold 0.8, and only top 100 proposals are kept for the refinement of stage-2 sub-network.

### 2.2. Point cloud region pooling

After obtaining 3D bounding box proposals, we aim at
refining the box locations and orientations based on the previously generated box proposals. To learn more specific local features of each proposal, we propose to pool 3D points
and their corresponding point features from stage-1 according to the location of each 3D proposal.

### 2.3. Canonical 3D bounding box refinement

![image](/assets/img/post_assest/paper-note-8-6.png)

<b> Canonical transformation: </b> To take advantages of our
high-recall box proposals from stage-1 and to estimate only
the residuals of the box parameters of proposals, we transform the pooled points belonging to each proposal to the
canonical coordinate system of the corresponding 3D proposal. As shown in Fig. 4, the canonical coordinate system for one 3D proposal denotes that (1) the origin is located at the center of the box proposal; (2) the local $X'$ and $Z'$
axes are approximately parallel to the ground plane
with $X'$ pointing towards the head direction of proposal and
the other $Z'$
axis perpendicular to $X'$; (3) the $Y'$
axis remains the same as that of the LiDAR coordinate system.
All pooled points’ coordinates p of the box proposal should
be transformed to the canonical coordinate system as $p˜$ by
proper rotation and translation. Using the proposed canonical coordinate system enables the box refinement stage to
learn better local spatial features for each proposal.

<b> Feature learning for box proposal refinement: </b> The refinement sub-network combines both the transformed local spatial points (features) $p˜$
as well as their global semantic features $f^{(p)}$
from stage-1
for further box and confidence refinement.
Although the canonical transformation enables robust local spatial features learning, it inevitably loses depth information of each object. For instance, the far-away objects
generally have much fewer points than nearby objects because of the fixed angular scanning resolution of the LiDAR sensors. To compensate for the lost depth information, we include the distance to the sensor.

<b> Losses for box proposal refinement: </b> We adopt the similar bin-based regression losses for proposal refinement.
A ground-truth box is assigned to a 3D box proposal for
learning box refinement if their 3D IoU is greater than
0.55.

## 3. Experiments


















