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














