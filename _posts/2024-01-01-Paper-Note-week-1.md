---
title: Paper note - [Week 1]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-12-29 11:11:14 +0700
categories: [Computer vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Paper note - [Week 1]

## BEV-MODNet: Monocular Camera based Bird’s Eye View Moving Object Detection for Autonomous Driving
[2021 IEEE Intelligent Transportation Systems Conference (ITSC)]

### Motivation 

BEV maps provides a better representation than image
view as they minimize the occlusions between objects that
lie on the same line of sight with the sensor. But a projection is usually error prone due to the absence
of depth information 

=> Apply Deep learning for improve this inaccuracy by learning the objects representation directly in BEV representation.

=> Make end-to-end learning of BEV motrion segmentation.

### Contribution

- Create a dataset comprising of 12.9k images containing BEV pixel-wise annotation for moving and static
vehicles for 5 classes.
- Design and implement a simple end-to-end baseline
architecture demonstrating reasonable performance.
- Results against conventional Inverse
Perspective Mapping (IPM)  approach and show a
significant improvement of over 13%.

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1.png?raw=true)

- 2 encoder network: RGB and Optical flow 

- Final output is a binary mask which predicts a class for each pixel among the two classes (Moving and Static)

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-1.png?raw=true)

### Conclusion

- Using 2 network

## Bi-Directional Bird’s-Eye View Features Fusion for 3D Multimodal Object Detection and Tracking
[2023 International Automatic Control Conference (CACS)]

### Motivation

- Multi-view fusion, where both RGB and Lidar data are spatially transformed into the same representation space and then feature maps are fused. 

- BEVFusion successfully addresses this issue by implementing the Lift, Splat, Shoot (LSS) method for RGB perspective
transformation. They also use Interval Reduction to speed up
the time-consuming BEV pooling process, effectively
enhancing the transformation into BEV RGB features.

- The initial step is to convert RGB, and Lidar
features into a unified BEV representation. However, the
original approach only used an essential backbone for 3D
feature extraction and did not incorporate RGB features in this
process. In addition, the converted RGB BEV features were
not effectively utilized. 

=> enhance the BEVFusion method by introducing a new BEV-based feature fusion approach.

### Contribution

- Propose the BEV feature fusion method in this study. This
involves transforming the BEV RGB features in different
scales and then fusing them with the 3D space.

=> This allows for the inclusion of RGB semantic information during feature
extraction, resulting in more effective Lidar features.

- Utilizes the Focal Conv module to improve the extraction of foreground point cloud features.

=> This module utilizes manifold convolution to distinguish
between foreground and background, and its non-fixed shape
convolution kernels enhance feature extraction. This addresses
the issue of feature expansion commonly encountered with
regular sparse convolutions. 

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-3.png?raw=true)

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-2.png?raw=true)


- RGB Encoder using SwinTransformer and FPN

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-4.png?raw=true)

- Lidar encoder branch

For the backbone, the model architecture introduced by
[focal sparse convolution paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.pdf) is adopted.


- BEV feature fusion module

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-5.png?raw=true)

When attempting to fuse the RGB BEV feature with
Lidar features, discrepancies in feature channel numbers and
sizes present a challenge. These differences hinder direct
spatial alignment using calibration parameters, necessitating
the use of a decoder to match the dimensions of the RGB BEV
feature map to those of the point cloud feature map, thereby
enabling fusion.

- BEV encoder and dense head

After obtaining the BEV-view RGB and Lidar features
through the encoder, selecting the appropriate fusion method
becomes vital. Since the channel numbers and sizes of the two
feature maps are aligned, a straightforward approach such as
addition or concatenation could be employed for fusion.
However, in the context of BEVFusion, the authors
emphasize that spatial misalignment issues might arise due to
the inherent errors in the RGB encoder’s perspective
transformation. To mitigate the impact of spatial
misalignment, they choose to perform fusion through
convolutions. For the final dense head, we adopt the model
architecture proposed in TransFusion. The loss function
follows the settings in TransFusion and is defined as Equation
(2):

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-6.png?raw=true)

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-7.png?raw=true)

- Model A serves as the baseline,

- Model B incorporates the BEV Feature Fusion Module into the baseline.

- Model C replaces the baseline’s backbone with focal sparse convolution. 

- Model D adds the BEV Feature Fuse Module on the modified backbone in Model C.

### Conclusion

- Multiview and no code public
- Still complex and slow

## RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation
[ArXvi 2021]

### Motivation

Although we mostly regard the point cloud as the raw data
format, the range image is the native representation of the
rotating LIDAR sensor. It retains
all original information without any loss. Beyond this, the
dense and compact properties make it efficient to process.

=> Extract features from the range image.

Problem with range image:

- The large scale variation makes it difficult to decide the anchor size in the range view.
- The occlusion causes the bounding boxes to easily overlap with each other.

### Contribution

- Propose the RangeRCNN framework which takes
the range image as the initial input to extract dense
and lossless features for fast and accurate 3D object
detection.

- Design a 2D CNN utilizing dilated convolution to
better adapt the flexible receptive field of the range
image.

- Propose the RV-PV-BEV module for transferring the
feature from the range view to the bird’s eye view for
easier anchor generation.

- Propose an end-to-end two-stage pipeline that utilizes a region convolutional neural network (RCNN) for
better height estimation. The whole network does not
use 3D convolution or point-based convolution which
makes it simple and efficient.

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-8.png?raw=true)

- Range Image Backbone

range, coordinate, and intensity as the input channel (5 x h x w)

KITTI: 5 x 48 x 512 (only label in front view)
Waymo: 5 x 64 x 2650

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-9.png?raw=true)


![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-10.png?raw=true)

- RV-PV-BEV Module

Range image it is
difficult to assign anchors in the range image plane due to
the large scale variation. The severe occlusion also makes
it difficult to remove redundant bounding boxes in the NonMaximum Suppression (NMS) module

=> Transfer the feature extracted from the range
image to the bird’s eye image.

For each point, we record its corresponding pixel coordinates in the range image plane, so we can obtain the
pointwise feature by indexing the output feature of the range
image backbone. Then, we project the pointwise feature to
the BEV plane. For points corresponding with the same pixel
in the BEV image, we use the average pooling operation
to generate the representative feature for the pixel. Here the
point view only serves as the bridge to transfer features from
the range image to the BEV image. We do not use the pointbased convolution to extract features from points.

- 3D RoI Pooling

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-11.png?raw=true)

Based on the bird’s eye image, they generate 3D proposals
using the region proposal network (RPN). However, neither
the range image nor the bird’s eye image explicitly learns
features along the height direction of the 3D bounding box,
which causes their predictions to be relatively accurate in the
BEV plane, but not in the 3D space. As a result, we want
to explicitly utilize the 3D space information. They conduct
a 3D RoI pooling based on the 3D proposals generated
by RPN. The proposal is divided into a fixed number of
grids. Different grids contain different parts of the object.
As these grids have a clear spatial relationship, the height
information is encoded among their relative positions. They
directly vectorize these grids from three dimensions to one
dimension sorted by their 3D positions.
They apply several fully connected layers to the vectorized
features and predict the refined bounding boxes and the
corresponding confidences.

- Loss Function

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-12.png?raw=true)

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-13.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-14.png?raw=true)


### Conclusion

- New way extract feature from range image and use in BEV
- Using 2D Backbone but still low because use encoder + decoder
- Quite fast but still slow with PointPillar

## RangeIoUDet: Range Image based Real-Time 3D Object Detector Optimized by Intersection over Union

[CVPR 2021]

### Motivation

-  simple framework for easy deployment, fast inference time, and 2D convolution based
model without extra customized operations.

### Contribution

- Propose a single-stage 3D detection model
RangeIoUDet based on the range image, which is simple, effective, fast, and only uses 2D convolution.

- Enhance pointwise features by supervising the
point-based IoU, which makes the network better learn
the implicit 3D information from the range image.

- Propose the 3D Hybrid GIoU (HyGIoU) loss for
supervising the 3D bounding box with higher location
accuracy and better quality evaluation.

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-15.png?raw=true)

- Baseline Model of RangeIoUDet

Similar with RangeRCNN

- Pointwise Feature Optimized by LovaszSoftmax Loss

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-16.png?raw=true)


The 2D FCN outputs the pixel-wise feature of the range
image, which is further transferred to the point cloud to obtain the pointwise feature. Due to the 2D receptive field in
the range image plane, points far away in the 3D space may
obtain similar features if they are adjacent in the range image. The pointwise feature is directly passed to the following module without any extra supervision. We argue that
the implicit 3D position information encoded in the range
image is not fully exploited. We propose to supervise the
pointwise feature to make the 2D FCN learn better.

One simple idea is to directly apply a segmentation loss
function to the pointwise feature, shown in Fig. 3(a). Directly supervising the pointwise feature of the point cloud
is equivalent to supervise the pixel-wise feature of the range
image, which does not further utilize the 3D position information of point clouds. In fact, this simple idea can not
improve the detection accuracy. We analyze that the main
problem is the lack of the 3D receptive field. However, utilizing the 3D receptive field needs to introduce the point
based or voxel based convolution, which may slow down
the inference speed and increase the difficulty of the deployment. Considering the above factors, we design the pointbased IoU module shown at the bottom right of Fig. 2.

To make use of the 3D receptive field of point clouds, we
search the local neighbors of each point using ball query and
apply PointNet to extract local features (shown in Fig. 3(b)).
We choose different radii for achieving multi-scale features.
The multi-scale features are extracted in parallel and concatenated pointwisely. Finally, the features extracted in the 3D space is supervised by Lovasz-Softmax loss [1] to directly optimize the point-based IoU for better distinguishing
the foreground and background. The local PointNet refines
the pointwise feature which makes the final segmentation
result better. Meanwhile, the better supervision promotes
the 2D FCN to learn better through back-propagation. As a
result, the pointwise feature passed to BEV becomes better
even though the local PointNet is not directly applied to it.
When designing this module, we adopt the parallel structure (Fig. 3(d)) instead of the cascade strcuture (Fig. 3(c)) to
extract the multi-scale feature. The parallel structure allows
the gradient to be faster and more easily backpropagated to
the 2D FCN, which leads to the point-based IoU supervision
to have a more direct impact on the pointwise feature. The
deeper structure may improve the quality of the pointwise
segmentation but degrade the detection performance.

-  3D Bounding Box Optimized by 3D Hybrid GIoU Loss

The performance of 3D object detection is affected
by two factors: (1) the positioning accuracy of the 3D
bounding box, which is determined by seven parameters
(x, y, z, L, W, H, θ); (2) the confidence score of the box,
which has a great influence on the quality evaluation. The
positioning accuracy and quality of the predicted box are
both highly related to its IoU with the corresponding ground
truth, which motivates us to explore the value of the boxbased IoU in 3D detection. This section proposes a 3D Hybrid GIoU loss, including hybrid GIoU regression loss and
3D IoU quality loss to improve the above two aspects.


![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-17.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-18.png?raw=true)

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-19.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-20.png?raw=true)

### Conclusion

- New Loss
- Fast
- No code public

## RangeDet: In Defense of Range View for LiDAR-based 3D Object Detection

[ICCV 2021]

### Motivation

#### Advantages of range view
-  Organizing the point cloud in range view misses no information

-The compactness also enables fast neighborhood queries based on range image coordinates, while point
view methods usually need a time-consuming ball query algorithm to get the neighbors. 

- Moreover, the valid detection range of range-view-based detectors can be as far as
the sensor’s availability, while we have to set a threshold for
the detection range in BEV-based 3D detectors.

####  The key to high-performance range-view-based detection

- The challenge of detecting objects with sparse
points in BEV is converted to the challenge of scale variation in the range image, which is never seriously considered
in the range-view-based 3D detector.

- The 2D range view is naturally compact, which
makes it possible to adopt high resolution output without
huge computational burden. However, how to utilize such
characteristics to improve the performance of detectors is
ignored by current range-image-based designs.

-  Unlike in 2D image, though
the convolution on range image is conducted on 2D pixel
coordinates, while the output is in the 3D space. This point
suggests an inferior design in the current range-view-based
detectors: both the kernel weight and aggregation strategy
of standard convolution ignore this inconsistency, which
leads to severe geometric information loss even from the
very beginning of the network.

=> RangeDet

### Contribution

- Propose a pure range-view-based framework – RangeDet, which is a single-stage anchorfree detector designated to address the aforementioned challenges.

- For the first challenge, they propose a simple yet effective <b> Range Conditioned Pyramid </b> to mitigate it. For the second challenge,
they use <b> weighted Non-Maximum Suppression </b> to remedy the
issue. For the third one, they propose <b> Meta-Kernel </b> to capture
3D geometric information from 2D range view representation. 

- Explore how
to transfer common data augmentation techniques from 3D space to the range view.

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-222.png?raw=true)

#### Review of Range View Representation

- For a LiDAR with m beams and n times measurement
in one scan cycle, the returned values from one scan form a
m × n matrix, called <b> range image </b>

-  Each column
of the range image shares an azimuth, and each row of the
range image shares an inclination. They indicate the relative
vertical and horizontal angle of a returned point w.r.t the
LiDAR original point. The pixel value in the range image
contains the range (depth) of the corresponding point, the
magnitude of the returned laser pulse called intensity and
other auxiliary information. 

-  One pixel in the range image
contains at least three geometric values: range $r$, azimuth $θ$,
and inclination $φ$. These three values then define a spherical
coordinate system.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-21.png?raw=true)

$$ x  = r cos(φ) cos(θ) $$
$$ y  = r sin(φ) cos(θ) $$
$$ z  = r sin(φ) $$

where x, y, z denote the Cartesian coordinates of points.
Note that range view is only valid for the scan from one
viewpoint. It is not available for general point cloud since
they may overlap for one pixel in the range image.

#### Range Conditioned Pyramid

- Similar with FPN

-  The difference lies in how to assign each object to a different layer for training.

- In the original FPN, the ground-truth bounding box
is assigned based on its area in the 2D image. Nevertheless, simply adopting this assignment method ignores the
difference between the 2D range image and 3D Cartesian
space. A nearby passenger car may have similar area with
a far away truck but their scan patterns are largely different.
Therefore, we designate the objects with a similar range to
be processed by the same layer instead of purely using the
area in FPN. Thus we name our structure as Range Conditioned Pyramid (RCP).

#### Meta-Kernel Convolution

Compared with the RGB image, the depth information
endows range images with a Cartesian coordinate system,
however standard convolution is designed for 2D images on
regular pixel coordinates. For each pixel within the convolution kernel, the weights only depend on the relative pixel
coordinates, which can not fully exploit the geometric information from the Cartesian coordinates. In this paper, we
design a new operator which learns dynamic weights from
relative Cartesian coordinates or more meta-data, making
the convolution more suitable to the range image.

For better understanding, we first disassemble standard
convolution into four components: sampling, weight acquisition, multiplication and aggregation.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-23.png?raw=true)

- Quite similar with pointnet

#### Weighted Non-Maximum Suppression

- From [this paper](https://arxiv.org/pdf/1505.01749.pdf)

#### Data Augmentation in Range View

Random global rotation, Random global flip and CopyPaste are three typical kinds of data augmentation for
LiDAR-based 3D object detectors. Although they are
straightforward in 3D space, it’s non-trivial to transfer them
to RV while preserving the structure of RV.

- Some changing for using with RV

#### [IoU Prediction head](https://arxiv.org/pdf/2008.13367.pdf)
#### Regression head

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-24.png?raw=true)

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-25.png?raw=true)

- 12fps in 2080Ti

### Conclusion

- New way for extract feature in range view with geometric information
- Loss function
- Pyramid Network
- Slow
- [Code](https://github.com/tusen-ai/RangeDet/tree/main)

## RangePerception: Taming LiDAR Range View for Efficient and Accurate 3D Object Detection

[NeurIPS 2023]

### Motivation

- Two critical unslolved challengens in existing RV-Based detection methods: 

- <b> Spatial Misalignment: </b> Existing RV-based detectors treat range images the same way as RGB images,
by directly feeding them into 2D convolution backbones. This workflow neglects the nature that range
images contain rich depth information, and even two range pixels are adjacent in range coordinate,
their actual distance in 3D space could be more than 30 meters.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-26.png?raw=true)

As visualized in figure above, foreground
pixels on the margins of vehicles and pedestrians are often far from their neighboring background
pixels in 3D space. Directly processing such 3D-space-uncorrelated pixels with 2D convolution
kernels can only produce noisy features, hindering geometric information extraction from the margins
of foreground objects.

- <b> Vision Corruption: </b> 

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-27.png?raw=true)

 When objects of interest are located on the margins of range images, as shown
in Fig. 1(c,f), their corresponding foreground range pixels are separately distributed around the left
and right borders of the range image. Since CNNs have limited receptive fields, features around
the left border cannot be shared with features around the right border and vice versa, when 2D
convolution backbones are used as feature extractors. This phenomenon, called Vision Corruption,
can significantly impact the detection accuracy of objects on the margins of range images. Previous
RV-based detection methods have overlooked this issue and directly processed range images with 2D
convolution backbones without compensating for the corrupted areas.

### Contribution

- <b> RangePerception Framework: </b> A novel high-performing 3D detection framework, named RangePerception, is introduced in this paper. 

- <b> Range Aware Kernel: </b> As part of RangePerception’s feature extractor, Range Aware Kernel (RAK)
is a trailblazing algorithm tailored to RV-based networks. RAK disentangles the range image space
into multiple subspaces, and overcomes the Spatial Misalignment issue by enabling independent
feature extraction from each subspace.

- <b> Vision Restoration Module: </b> To resolve the Vision Corruption issue, Vision Restoration Module
(VRM) is brought to light in this study. VRM extends the receptive field of the backbone network by
restoring previously corrupted areas. VRM is particularly helpful to the detection of vehicles, as will
be illustrated in the experiment section.

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-28.png?raw=true)

####  Range Aware Kernel

As a key component of RangePerception’s feature extractor, Range Aware Kernel is an innovative
algorithm specifically designed for RV-based networks. <b> RAK disentangles the range image space
into multiple subspaces </b>, and overcomes the Spatial Misalignment issue by <b> enabling independent
feature extraction from each subspace </b>.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-29.png?raw=true)

####  Vision Restoration Module

As described in Sec. 2, each column in range image I corresponds to a shared azimuth $θ ∈ [0, 2π]$,
indicating the spinning angle of LiDAR. Specifically, $θ = 0$ at left margin of range image and $θ = 2π$
at right margin of range image. Due to the periodicity of LiDAR’s scanning cycle, azimuth values
0 and 2π correspond to beginning and end of each scanning cycle, both pointing in the opposite direction of the ego vehicle. As illustrated in Fig. 4, objects located behind ego vehicle are often
separated by ray with $θ = 0$, resulting in Vision Corruption phenomena elaborated in figure below.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-30.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-31.png?raw=true)

=> By predefining a restoration angle $δ$, VRM builds an extended spherical space with azimuth $θ ∈
[−δ, 2π + δ]$. In this way, visual features originally corrupted by LiDAR’s sampling process are
restored on both sides of range image I, significantly easing the feature extraction from the margins of
I.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-32.png?raw=true)

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-33.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-34.png?raw=true)

### Conclusion

- No code
- Nice idea

## ACDet: Attentive Cross-view Fusion for LiDAR-based 3D Object Detection
[2022 International Conference on 3D Vision (3DV)]

### Motivation

- Most range-view-based methods ignore the depth discontinuity between pixels, and directly apply standard 2D
convolutions on the range image. The adjacent pixels on
range image may be far away in 3D scene, especially the
boundary area of objects. Directly applying standard 2D
convolutions on the boundary pixels will produce similar
features, but in fact they should be distinctly different. 

- Objects on range view are easily overlapped with each
other, which makes it difficult to directly generate proposals
from range view. 

-  Anchor-based head is still the mainstream of 3D object detection, due to its high accuracy. However, its performance highly depends on the manually designed parameters, which limits its generalization. Meanwhile, anchorfree head is becoming more and more popular due to its simplicity and comparable performance.

### Contribution

- An attentive cross-view fusion module based on transformer and supervised foreground mask, which attentively fuses the complementary information from
range view and BEV.

- A geometric-attention kernel, which strengthens the
feature learning in range view by aggregating neighboring features with spatially attentive weights.

- An anchor-free detection head, which integrates advanced label assignment strategy and IoU-aware classification loss to predict 3D bounding boxes with better
accuracy and higher quality.

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-35.png?raw=true)

- Cylindrical-view Projection

Range view image generation

- Geometric-attention Kernel

Propose an attentive geometric-aware kernel, named <b> geometric-attention kernel </b>, which aggregates neighboring features by considering the spatial attention weights among neighbors, and further strengthens the output features by concatenating geometric features. Furthermore, we incorporate a binary mask to skip the computation of empty pixels on range image,
which significantly speeds up the kernel operation.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-36.png?raw=true)

- Attentive Cross-view Fusion

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-37.png?raw=true)

- Foreground Mask Supervision

3D object detection from point cloud is known to be
sensitive to noises and sparsity, especially for the far away
and small objects which contain very few points. [Recent work](https://dl.acm.org/doi/abs/10.1145/3474085.3475208) directly introduces a supervised mask-guided attention mechanism to highlight the object pixels from complex background. The mask is generated by projecting the
bounding boxes of objects into the BEV feature map and
assigning pixels inside projected box as 1, otherwise 0. We
argue that the attention mask should focus on enhancing
non-empty pixels to reduce noisy clues, rather than inferring the underlying structure of objects from empty pixels.
Therefore, we only consider the non-empty pixels inside the
projected boxes as foreground and skip empty ones. We
adopt a soft assignment instead of 0-1 hard assignment in
[this](https://dl.acm.org/doi/abs/10.1145/3474085.3475208).

- Anchor-free Detection Head

...

- Loss Functions

...

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-38.png?raw=true)

### Conclusion

- [Public code](https://github.com/Jiaolong/acdet)

- New head 

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-39.png?raw=true)

- Attention for geometric information extraction

- 20fps in 3090

## Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds

[CVPR 2022]

### Motivation

- Not all points are equally important to the task of object detection.

- Motivated by this, we aim to propose a task-oriented,
instance-aware downsampling framework, to explicitly
preserve foreground points while reducing the memory/computational cost. Specifically, two variants, namely
class-aware and centroid-aware sampling strategies are
proposed. In addition, we also present a contextual instance
centroid perception, to fully exploit the meaningful context information around bounding boxes for instance center regression. Finally, we build our IA-SSD based on the
bottom-up single-stage framework.

### Contribution

- Identify the sampling issue in existing point-based
detectors, and proposed an efficient point-based 3D
detector by introducing two learning-based instanceaware downsampling strategies.

- e proposed IA-SSD is highly efficient and capable
of detecting multi-class objects on LiDAR point clouds
in a single pass. They also provided a detailed memory
footprint vs. inference-speed analysis to further validate the superiority of the proposed method.

- Extensive experiments on several large-scale datasets
demonstrate the superior efficiency and accurate detection performance of the proposed method.

### Method

Different from dense prediction tasks such as 3D semantic segmentation, where point-wise prediction is required,
<b> 3D object detection naturally focus on the small yet important foreground objects </b> (i.e., instances of interest including car, pedestrian, etc.). However, existing pointbased detectors usually adopt task-agnostic downsampling
approaches such as random sampling or farthest point
sampling in their framework. Albeit effective for
memory/computational cost reduction, the most important
foreground points are also diminished in progressive downsampling. Additionally, due to the large difference in size
and geometrical shape of different objects, existing detectors usually train separate models with various carefully
tuned hyperparameters for different types of objects. However, this inevitably affects the deployment of these models
in practice.

=> Can we train a single point-based model, which is efficient and
capable of detecting multi-class objects in a single pass?

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-40.png?raw=true)

#### Instance-aware Downsampling Strategy

To preserve foreground points as much as
possible, we turn to leverage the latent semantics of each
point, since the learned point features may incorporate
richer semantic information as the hierarchical aggregation
operates in each layer. Following this idea, we propose the
following two task-oriented sampling approaches by incorporating the foreground semantic priors into the network
training pipelines.

- <b> Class-aware Sampling: </b> This sampling strategy aims
to learn semantics for each point, so as to achieve selective downsampling. To achieve this, we introduce extra
branches to exploit the rich semantics in latent features.

- <b> Centroid-aware Sampling: </b> Considering instance center
estimation is the key for final object detection, we further
propose a centroid-aware downsampling strategy to give
higher weight to points closer to instance centroid. 

#### Contextual Instance Centroid Perception

- <b> Contextual Centroid Prediction: </b> Inspired by the success of context prediction in 2D images, we attempt
to leverage the contextual cues around the bounding box for
instance centroid prediction. 

- <b> Centroid-based Instance Aggregation: </b> For shifted representative (centroid) points, we further utilize a PointNet++ module to learn a latent representation for each instance. Specifically, we transform the neighboring points
to a local canonical coordinate system, then aggregate the
point feature through shared MLPs and symmetric functions.

- <b> Proposal Generation Head: </b> The aggregated centroid
point features are then fed into proposal generation head to predict bounding boxes with classes. We encode the proposal as a multidimensional representation with location,
scale, and orientation. Finally, all proposals are filtered by
3D-NMS post-processing with a specific IoU threshold.

####  End-to-End Learning

...

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-41.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-42.png?raw=true)

### Conclusion

- [Public code](https://github.com/yifanzhang713/IA-SSD)
- Fast
- The instance-aware sampling relies on the semantic prediction
of each point, which is susceptible to class imbalances distribution.
- 0.0443s/frame with 1 sample

## CVFNet: Real-time 3D Object Detection by Learning Cross View Features

[IROS 2022]

### Motivation

- Effciently detecting
objects in 3D LiDAR point clouds still remains challenging
due to the irregular format and the sparsity of the point cloud.

- The ball query operations for aggregating local features from neighboring points involved in
PointNet++ are time-consuming.

- view-based approaches
usually suffer from the information loss during the projection
from 3D points to 2D views, which is hard to be compensated
during 3D detection. When multiple views are employed,
directly concatenating features in proposal level as in MV3D can weaken the consistency among features. These
shortcomings lead to inferior performance of the view-based
methods than the voxel or point based counterparts.

### Contribution

- A novel real-time view-based one stage 3D object detector is proposed. It comprises only 2D convolutions and
is able to learn consistent cross-view features through a
progressive fusion manner.

- A Point-Range fusion module PRNet is designed to
effectively extract and fuse the 3D point and 2D range
view features in a bilateral aggregation way.

- A Slice Pillar transformation module is presented to
transform 3D point features into 2D BEV features,
which features less 3D geometric information loss during the dimension collapse.


### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-43.png?raw=true)

-  Point-Range Fusion Module

Range view is the native form of rotating LiDAR, featuring dense range images distributed in spherical coordinates where 2D convolutions can be applied effectively.
Point view is of 3D coordinates that multi-layer perceptron
(MLPs) can be used to extract features for each point. To
extract and aggregate the features of these two views, we
propose a Point-Range fusion module within which multistage bilateral fusion is carried out.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-44.png?raw=true)


- Slice Pillar Transformation

Bird’s eye view(BEV) is a popular representation in 3D
object detection as it features no occlusion or scale variation.
There are two common ways to project the point cloud to
BEV: voxelization and pillarization.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-45.png?raw=true)

- Sparse Pillar Head

After the slice pillar transformation module and a series
of convolution layers on BEV, we have the BEV feature
map at hand. To estimate the proposals in each BEV grid,
most existing methods use a dense prediction head. However,
since most grids are empty, we introduce a sparse pillar
head to suppress the useless proposals. The sparse pillar
head consists of classifcation and regression head. Given
a sparse BEV map, we frst defne nonempty grids as valid
pillars and use 2D convolutions to extract features, which are
then indexed to the pillar-wise features. Finally, classifcation
head is used to predict the probability of a pillar’s class,
and regression head is responsible for regressing 3D box
parameters. The advantages of our sparse pillar head are
two folds. Firstly, it balances the distribution of positive
and negative samples by fltering out the empty grids which
are otherwise counted as negative samples. Secondly, the
computation cost is further reduced.

- Loss Function

...

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-46.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-47.png?raw=true)

### Conclusion

- No code
- Fast

## RI-Fusion: 3D Object Detection Using Enhanced Point Features With Range-Image Fusion for Autonomous Driving

[IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT 2022]

### Motivation

- Range image and RGB fusion 

### Contribution

- Propose a novel LiDAR and camera image fusion
module termed RI-Fusion, in which the point cloud is
converted to a range view to reduce the semantic gap
between two modalities. This module is plug-and-play
and can be easily accessed by existing LiDAR-based
methods to enhance the performance.

- Propose a range-image attention (RI-Attention) module, based on a self-attention mechanism, to achieve
an effective fusion of range and RGB images. This
makes full use of global features for the interaction
between two heterogeneous modes and leads to remarkable improvements, especially for small objects such as
pedestrians and cyclists.


### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-48.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-49.png?raw=true)

### Results

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-50.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-51.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-52.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-53.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-1-54.png?raw=true)

### Conclusion

- Attention
- Module













