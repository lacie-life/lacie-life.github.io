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







