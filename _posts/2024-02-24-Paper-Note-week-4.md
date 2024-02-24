---
title: Paper note - [Week 4]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2024-02-24 11:11:14 +0700
categories: [Computer vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Paper note - [Week 4]

## [Transformation-Equivariant 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/2211.11962.pdf)

(AAAI 2021)

### Motivation

- Most of these methods construct detection frameworks based on ordinary voxels or pointbased operations.

- It is desired that the predictions from a 3D detector are
equivariant to transformations such as rotations and reflections. In other words, when an object changes its orientation
in the input points, the detected bounding box of this object
should have a same shape but change its orientation accordingly. However, most voxel- and point-based approaches do
not explicitly model such a transformation equivariance, and could produce unreliable detection results when processing
transformed point clouds.

-  Some detectors achieve approximate transformation equivariance through data augmentations. Their performance, however, heavily relies on generating comprehensive training samples and adopting more complex networks with larger capacity.

- Some other approaches employ a Test Time Augmentation
(TTA) scheme or transformation equivariance => time problem.

### Contribution

- Present TED, a TransformationEquivariant 3D Detector to tackle this efficiency issue.
TED first applies a sparse convolution backbone to extract multi-channel transformation-equivariant voxel features. Then TED aligns and aggregates the equivariant
features into a lightweight and compact representation
for high-performance 3D object detection.

- (1) Introduce novel TeBEV pooling and TiVoxel
pooling modules that efficiently learn equivariant features
from point clouds, leading to better 3D object detection.

- (2) Propose a new DA-Aug mechanism to create more
sparse training samples from nearby dense objects. DA-Aug
be used as a general off-the-shelf augmentation method to
enhance distant object detection. 

### Method

TED has three key parts: (1) the Transformationequivariant Sparse Convolution (TeSpConv) backbone; (2)
Transformation-equivariant Bird Eye View (TeBEV) pooling; and (3) Transformation-invariant Voxel (TiVoxel) pooling. TeSpConv applies shared weights on multiple transformed point clouds to record the transformation-equivariant
voxel features. TeBEV pooling aligns and aggregates the
scene-level equivariant features into lightweight representations for proposal generation. TiVoxel pooling aligns and
aggregates the instance-level invariant features into compact
representations for proposal refinement. In addition, we designed a Distance-Aware data Augmentation (DA-Aug) to
enhance geometric knowledge of sparse objects. DA-Aug creates sparse training samples from nearby dense objects.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-4-1.png?raw=true)

They present TED,
which is both transformation-equivariant and efficient. This
is done by a simple yet effective design: let TeSpConv
stack multi-channel transformation-equivariant voxel features, while TeBEV pooling and TiVoxel pooling align and
aggregate the equivariant features into lightweight scenelevel and instance-level representations for efficient and effective 3D object detection.

#### Transformation-Equivariant Voxel Backbone

To efficiently encode raw points into transformationequivariant features, we first design a Transformationequivariant Sparse Convolution (TeSpConv) backbone. The
TeSpConv is constructed from the widely used sparse convolution (SpConv). Similar to CNNs, SpConv is
translation-equivariant. However, the SpConv is not equivariant to rotation and reflection. To address this, we extend SpConv to rotation and reflection equivariant by
adding transformation channels. Similar to a 2D counterpart, the equivariance is achieved by two
aspects: (1) weights highly shared between transformation
channels; (2) transformation of input points with different
rotation angles and reflection.

<b> Note </b> for the multi-modal setting, they encode the features of pseudo-points by the same network architecture.
Compared with the voxel features encoded by regular sparse
convolution, features will contain diverse features under different rotation and reflection transformations.

#### Transformation-Equivariant BEV Pooling

The voxel features contain a large number of
transformation channels; thus, directly feeding them into
RPN introduces huge additional computation and requires
larger GPU memory. To address this, we propose the TeBEV
pooling, which aligns and aggregates the scene-level voxel
features into a compact BEV map by bilinear interpolation
and max-pooling.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-4-2.png?raw=true)

#### Transformation-Invariant Voxel Pooling

Lots of recent detectors apply Region of Interest (RoI) pooling operation that extracts instance-level transformationinvariant features from scene-level transformationequivariant backbone features for proposal refinement. Nevertheless, directly
applying such a pooling operation to extract features from
our backbone is infeasible. The reasons are: 

(1) the proposal $B∗$
in the coordinate system $T^1$
is unaligned with the
voxel features transformed by different $T^i$. 

(2) The voxel features from our TeSpConv contains multiple
transformation channels, and directly feeding the extracted
features to the detection head needs huge additional computation and GPU memory. To address these issues, we
propose TiVoxel pooling, which aligns and aggregates the
instance-level voxel features into a compact feature vector
by multi-grid pooling and cross-grid attention.

#### Distance-Aware Data Augmentation

The geometry incompleteness of distant objects commonly
results in a huge detection performance drop. To address
this, we increase the geometric knowledge of distant sparse
objects by creating sparse training samples from nearby
dense objects. A simple method is to apply random sampling or farthest point sampling (FPS). However, it destroys
the distribution pattern of the point clouds scanned by LiDAR.
 
To address this, we propose a distance-aware sampling
strategy, which considers the scanning mechanics of LiDAR
and scene occlusion.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-4-2.png?raw=true)


### Experiments

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-4-3.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-4-4.png?raw=true)

### Conclusion

- Still use voxel

## [3D-CVF: Generating joint camera and LiDAR features using cross-view spatial feature fusion for 3D object detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720715.pdf)

(ECCV 2022)


















