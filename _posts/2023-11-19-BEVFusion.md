---
title: Paper note 9 - BEVFusion - Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-11-19 11:11:14 +0700
categories: [Machine Learning]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

[BEVFusion - Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542)

## 1.Problem

Data from different sensors are expressed in fundamentally different modalities: e.g., cameras capture
data in perspective view and LiDAR in 3D view. To resolve this view discrepancy, we have to find a
unified representation that is suitable for multi-task multi-modal feature fusion. Due to the tremendous
success in 2D perception, the natural idea is to project the LiDAR point cloud onto the camera and
process the RGB-D data with 2D CNNs. However, this LiDAR-to-camera projection introduces
severe geometric distortion (see Figure 1a), which makes it less effective for geometric-oriented tasks,
such as 3D object recognition.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-1.png?raw=true)

Recent sensor fusion methods follow the other direction. They augment the LiDAR point cloud with
semantic labels, CNN features or virtual points from 2D images, and then apply an existing LiDAR-based detector to predict 3D bounding boxes. Although they have demonstrated
remarkable performance on large-scale detection benchmarks, these point-level fusion methods barely
work on semantic-oriented tasks, such as BEV map segmentation. This is because
the camera-to-LiDAR projection is semantically lossy (see Figure 1b): for a typical 32-beam LiDAR
scanner, only 5% camera features will be matched to a LiDAR point while all others will be dropped.
Such density differences will become even more drastic for sparser LiDARs (or imaging radars).

=> Solution: BEVFusion to unify multi-modal features in a shared bird’s-eye view (BEV)
representation space for task-agnostic learning. We maintain both geometric structure and semantic
density (see Figure 1c) and naturally support most 3D perception tasks (since their output space can
be naturally captured in BEV). While converting all features to BEV, we identify the major prohibitive
efficiency bottleneck in the view transformation: i.e., the BEV pooling operation alone takes more
than 80% of the model’s runtime.

## 2. Contribution

- Propose BEVFusion to unify multi-modal features in a shared bird’s-eye view (BEV)
representation space for task-agnostic learning

-  Propose a specialized kernel with precomputation and
interval reduction to eliminate this bottleneck, achieving more than 40× speedup. 

- Apply the fully-convolutional BEV encoder to fuse the unified BEV features and append a few task-specific
heads to support different target tasks.

## 3. Related work

### 3.1. LiDAR-Based 3D Perception
- Single-stage 3D object detection
- Extract flattened point cloud features from LiDAR point cloud using PointNet, SaprseConvNet, etc. and perform detection in BEV space
- Anchor-free 3D object detection
- U-Net like 3D semantic segmentiation model
- Two-stage object detection

### 3.2. Camera-Based 3D Perception
- Studying camera-only 3D perception due to the high cost of LiDAR sensors

- FOCOS3D
Perform depth perception by adding 3D regression braches to the image detector

- DETR3D, PETR, Graph-DETR3D
Design an object detection head based on DETR and use an object query that can be learned in 3D space
    - DETR: Method of performing 2D object detection while each object query interacts with 2D image features

- BEVDet, M2BEV: Applying LSS and OFT to 3D object detection
Project camera image features onto LiDAR point cloud and convert them into BEV representation to perform 3D object detection
BEVDet estimates the depth information of image features from LiDAR points, and M2BEV assumes uniform depth distribution in the 
direction of the camera ray

- CaDNN: Use the view transformer to project the camera image feature onto the LiDAR point cloud and convert it into a BEV representation for 3D object detectiondmf tngod
Improve accuracy by adding depth estimation supervision to the output of the view transformer
    - View transformer
A module that receives image features and 3D coordinates as input and rearranges the image features according to 3D coordinates

- BEVFormer, CVT, Ego3RT
Perform view transformer using multi-head attention
BEVDet4D, BEVFormer, PETRv2
Perform 3D object detection using multi-camera temporal cue

<i> Temporal cue
It refers to temporal information, and in object detection, the movement or speed of an object is mainly estimated by comparing images or features of the previous frame and the current frame.
Temporal cues can provide more information than using single frame data alone. </i>

### 3.3. Multi-Sensor Fusion

<b> LiDAR-to-Camera Projection </b>

- Because 2D perception has been covered in many existing studies, we initially focused on processing the RGB-D projected LiDAR point cloud onto the camera image using 2D CNN.
- However, LiDAR-to-Camera projection, which projects the LiDAR point cloud onto the camera plane and converts it to 2.5D sparse depth, causes geometric distortion.
Since two adjacent pixels in a depth map may be far apart in 3D space, the camera view is unsuitable for tasks where the geometric characteristics of an object or scene are important, such as 3D object detection.

<b> Camera-to-LiDAR Projection (Point-Level Fusion) </b>

- Recently, in the field of sensor fusion, LiDAR point clouds are augmented with semantic labels, CNN features, or virtual points extracted from 2D images, and 3D bounding boxes are detected using existing LiDAR-based detectors.

- Perform LiDAR-based object detection by painting the semantic features of the camera image onto the foreground of the LiDAR point cloud.

- Camera-to-LiDAR projection suffers loss of semantic features due to differences in semantic density of each data format, and is therefore unsuitable for semantic-oriented tasks such as BEV map segmentation.
When using a 32-channel LiDAR scanner, only 5% of image features match LiDAR points and the rest are lost.

- Applicable to object-centric and geometric-centric tasks

- LiDAR input-level decoration
AutoAlign, FocalSparseCNN, PointPainting, PointAugmenting, MVP, - FusionPainting
LiDAR feature-level decoration
DeepContinuousFusion, DeepFusion

<b> Proposal-Level Fusion </b>

- Proposal-level fusion is an object-centric method and is difficult to apply to other tasks such as BEV map segmentation.

- MV3D
After creating an object proposal in 3D space, project it onto the image to extract RoI features

- F-PointNet, F-ConvNet, CenterFusion
Obtain 2D proposal from RGB image and convert to 3D frustum
Extract features by grouping local points containing the object from the points inside the frustum and then predict the bounding box with the object's direction in 3D space

- FUTR3D, TransFusion
Define object query in 3D space and fuse with image features of 2D proposal

### 4. Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-2.png?raw=true)

1. Feature extraction using modality-specific encoder for given multi-modal inputs
2. Convert multi-modal features to unified BEV representation
(The proposed BEV representation can maintain geometric and semantic information, so it is applicable to most 3D perception tasks)
3. We alleviate local misalignment of unified BEV features of both data formats using a convolution-based BEV encoder.
4. Perform multiple 3D tasks by adding task-specific heads

#### 4.1. Unified Representation

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-3.png?raw=true)

<b> View Discrepancy </b>

- Different types of data features exist in different views.
    + Camera feature : perspective view
    + LiDAR/radar feature : 3D/BEV view

- Additionally, each camera feature can have a different viewing angle (front, back, left, right).

- Because the same element may correspond to a completely different spatial location in different views, simply fusing a specific tensor by element does not work. None

<i> It is important to find a shared representation that allows all sensor features to be easily converted without loss of information and applicable to various types of tasks </i>

<b> To Bird's-Eye View </b>

- The method adopted in this paper uses the BEV view as a unified representation for sensor fusion.

- Since the output is also in the BEV region, it can be applied to most perception tasks.

- Transforming to BEV domain can maintain geometric structure (from LiDAR features) and semantic density (from camera features).

    + LiDAR-to-BEV projection: Geometric distortion does not occur because the sparse LiDAR feature is flattened along the z-dimension
    + Camera-to-BEV projection: Convert camera feature pixels into rays in 3D space and create a dense BEV feature map that fully preserves the camera's semantic information

#### 4.2. Efficient Camera-to-BEV Transformation

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-4.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-5.png?raw=true)

- Camera-to-BEV conversion is very important because the depth corresponding to each pixel of the camera feature is inherently ambiguous.

1. Predicting the discrete depth distribution of each pixel
2. Each pixel in the feature map is aligned with the camera ray.DDistributed into discrete points

<i> Number of channels in feature map $C$ The size of the camera feature point cloud generated from the feature map of one camera is $(H,W,D,C)$ expected </i>

3. 3D feature point cloud $x$,and step size along axis $r$ Quantized with
4. All features $r×r$ collected into the BEV gridWith uses to flatten along the axis <b> BEV pooling </b>

<i> Example: N = 6, (H, W) = (32, 88), and D = (60-1)/0.5=118

Six multi-view cameras are used, and each 256 Depth is discretized to 1 ~ 60m, step size 0.5m </i>

- Camera feature point cloud is very large, so computation speed is slow.

- To solve this problem, this paper proposes an optimized BEV pooling method using precomputation and interval reduction.

<b> BEV-pooling </b>

- Precomputation
    + The first step of BEV pooling: Grid association
    The process of grouping all points in the camera feature point cloud into one group belonging to the BEV grid

    + Unlike LiDAR point cloud, the coordinates of each point in the camera feature point cloud do not change if the internal and external parameters of the camera do not change.

    + Based on this, 3D coordinates and BEV grid indices of all points in the camera feature point cloud are calculated in advance.

    + Sort all points according to grid indices and record and store the rank of each point.

    + During inference, all feature points are rearranged and used according to pre-calculated rankings.

- Interval Reduction
    + The next step of BEV pooling: Feature aggregation
    Aggregate the features in each BEV grid with symmetric functions (e.g., mean, max, and sum)
    
    + As shown in Figure 3.b, existing pooling calculates the prefix sum for all points and then subtracts the result of the previous boundary from the boundary where the index changes.
    
    Prefix sum operation requires tree reduction on GPU.

    Compared to requiring only values ​​at the boundary, it is inefficient by generating many unused partial sums.
    
    + In this paper, to accelerate feature aggregation, we propose a special GPU kernel that directly parallelizes the BEV grid.
    
    Calculate the sum of sections within the grid by assigning a GPU thread to each grid.

    + The proposed kernel eliminates dependencies between grid outputs and therefore does not require multi-level tree reduction.
    
    Feature aggregation delay time can be reduced from 500ms to 2ms (Figure 3.c)


#### 4.3. Fully-Convolutional Fusion

- Features of two sensors transformed into a shared BEV representation can be easily fused with elementwise operators such as concatenation.

- However, although LiDAR BEV features and camera BEV features are in the same BEV space, there may be some spatial discrepancy due to inaccurate calculations in the view transformer process that calculates depth from the camera feature map.

- To solve this problem, this paper uses a convolution-based BEV encoder including residual blocks to compensate for local discrepancies.

- As a result, the proposed method has strengths in more accurate depth estimation.

#### 4.4. Multi-Task Heads

In this paper, a head specialized for various tasks was applied to the fused BEV feature map, and can be applied to most 3D perception tasks.

### 5. Experiments

In this paper, we evaluate the BEVFusion model with 3D object detection and BEV map segmentation based on Camera-LiDAR sensor fusion, which are geometric- and semantic-oriented tasks.

#### 5.1. Setup

#### Model

<b> Backbone </b>

- Image : Swin-T
- LiDAR : VoxelNet

<b> Camera feature extractor : FPN </b>

- Create a feature map 1/8 the size of the input
- In the previous methods, the 256 box

We apply FPN to fuse multi-scale camera features to produce a feature map of 1/8 input size. We downsample camera images to 256×704 and voxelize the LiDAR point cloud with 0.075m (for detection) and 0.1m (for segmentation)

- LiDAR point cloud performs voxelization at 0.075m in detection work and 0.1m in segmentation work.

- Since detection and segmentation require BEV feature maps of different sizes and ranges, grid sampling and bilinear interpolation are applied to transform the BEV feature maps before being input to each task-specific head.

#### Training

- Train the entire model in an end-to-end method
- Prevent overfitting by applying data augmentation to images and LiDAR
- Optimization used AdamW and weight decay $10^2$ use

#### Dataset

- nuScenes
6 monocular camera images and 360°, 40,157 annotated samples including FoV and 32-beams LiDAR scan data.

#### 5.2. 3D object Detection

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-6.png?raw=true)

<b> Results </b>

- BEVFusion shows high efficiency because it uses the BEV area, which can utilize all of the camera features, rather than just 5%, as a fusion space.
- Additionally, by using an efficient BEV pooling operator, we showed much fewer MACs for the same performance.

<i> <b> MAC (Multiply-ACcumulates) </b>

- Indicator that measures model complexity and computation amount in computer vision

- Indicates the total number of multiplications and additions the model performs, typically 1 MAC = 2 FLOPs (Floating Point Operations)

- MACs depend on the size of the model and the size of the input image. The smaller the MACs, the more efficient and faster the model. </i>

#### 5.3. BEV Map Segmentation

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-7.png?raw=true)

<b> Results </b>

- In Table 1 (object detection), the performance of the Camera-only SOTA model is worse than the LiDAR-only model, while in Table 2 (BEV map segmentation), the Camera-only BEVFusion outperforms the LiDAR-only baseline.

- In multi-modality, BEVFusion shows higher performance than the other two methods.

- The other two models used the Camera-to-LiDAR projection method under the assumption that the LiDAR format is more efficient in sensor fusion, but the experimental results of this paper argue that this is an incorrect assumption.

### 6. Analysis

<b> Weather and Lighting </b>

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-8.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-9.png?raw=true)

- Table 3 analyzes the performance of BEVFusion under various weather conditions.
- LiDAR-only models have difficulty detecting objects because heavy noise occurs in LiDAR sensors during rainy weather.
However, because cameras are robust to environmental changes, BEVFusion improves performance in rainy weather and reduces the performance gap compared to sunny days.
- At night, BEVFusion increases mIoU by about 12.8% compared to camera-only model BEVDet/LSS.
This shows how important geometric information extracted from LiDAR is when the camera sensor becomes useless.

<b> Sizes and Distances </b>

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-10.png?raw=true)


<b> Sparser LiDARs </b>

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-11.png?raw=true)

<b> Ablation Studies </b>

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-12.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-9-13.png?raw=true)

### 7. Conclusion

<b> Conclusion </b>

- BEVFusion is an efficient and general multi-task multi-sensor 3D perception framework.
- BEVFusion preserves both geometric and semantic information by fusing camera and LiDAR features in a shared BEV space.
- Accelerate the conversion process from camera feature map to BEV by more than 40x using the proposed BEV pooling.
- Breaking the long-standing practice that existing point-level sensor fusion is the optimal choice for multi-sensor 3D perception

<b> Limitations </b>

- Performance degradation occurs when training multiple tasks together
- Larger speedups were not achieved in multitasking situations.
- In this paper, accurate depth estimation was not explored in depth, and there is potential to improve the performance of BEVFusion.






