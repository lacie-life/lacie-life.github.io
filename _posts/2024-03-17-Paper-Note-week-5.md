---
title: Paper note - [Week 5]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2024-03-17 11:11:14 +0700
categories: [Computer vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Paper note - [Week 4]

## [A COMPREHENSIVE REVIEW OF YOLO ARCHITECTURES IN COMPUTER VISION: FROM YOLOV1 TO YOLOV8 AND YOLO-NAS](https://arxiv.org/pdf/2304.00501.pdf)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-1.png?raw=true)


### 1. Object Detection Metrics and Non-Maximum Suppression (NMS)

The Average Precision (AP), traditionally called Mean Average Precision (mAP), is the commonly used metric for
evaluating the performance of object detection models. It measures the average precision across all categories, providing
a single value to compare different models. The COCO dataset makes no distinction between AP and mAP. In the rest
of this paper, we will refer to this metric as AP.

#### 1.1. How AP works?

The AP metric is based on precision-recall metrics, handling multiple object categories, and defining a positive
prediction using Intersection over Union (IoU).

- <b> Precision and Recall: </b> Precision measures the accuracy of the model’s positive predictions, while recall measures the
proportion of actual positive cases that the model correctly identifies. There is often a trade-off between precision and
recall; for example, increasing the number of detected objects (higher recall) can result in more false positives (lower
precision). To account for this trade-off, the AP metric incorporates the precision-recall curve that plots precision against recall for different confidence thresholds. This metric provides a balanced assessment of precision and recall by
considering the area under the precision-recall curve.

- <b> Handling multiple object categories: </b> Object detection models must identify and localize multiple object categories
in an image. The AP metric addresses this by calculating each category’s average precision (AP) separately and then
taking the mean of these APs across all categories (that is why it is also called mean average precision). This approach
ensures that the model’s performance is evaluated for each category individually, providing a more comprehensive
assessment of the model’s overall performance.

- <b> Intersection over Union: </b> Object detection aims to accurately localize objects in images by predicting bounding
boxes. The AP metric incorporates the Intersection over Union (IoU) measure to assess the quality of the predicted
bounding boxes. IoU is the ratio of the intersection area to the union area of the predicted bounding box and the ground
truth bounding box. It measures the overlap between the ground truth and predicted bounding boxes.
The COCO benchmark considers multiple IoU thresholds to evaluate the model’s performance at different levels of
localization accuracy.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-2.png?raw=true)

#### 1.2. Computing AP

#### a. VOC Dataset
This dataset includes 20 object categories. To compute the AP in VOC, we follow the next steps:
- For each category, calculate the precision-recall curve by varying the confidence threshold of the model’s
predictions.
- Calculate each category’s average precision (AP) using an interpolated 11-point sampling of the precision-recall
curve.
- Compute the final average precision (AP) by taking the mean of the APs across all 20 categories.

#### b. Microsoft COCO Dataset

This dataset includes 80 object categories and uses a more complex method for calculating AP. Instead of using an
11-point interpolation, it uses a 101-point interpolation, i.e., it computes the precision for 101 recall thresholds from 0
to 1 in increments of 0.01. Also, the AP is obtained by averaging over multiple IoU values instead of just one, except
for a common AP metric called $AP_{50}$, which is the AP for a single IoU threshold of 0.5. The steps for computing AP in
COCO are the following:

- For each category, calculate the precision-recall curve by varying the confidence threshold of the model’s
predictions.
- Compute each category’s average precision (AP) using 101-recall thresholds.
- Calculate AP at different Intersection over Union (IoU) thresholds, typically from 0.5 to 0.95 with a step size
of 0.05. A higher IoU threshold requires a more accurate prediction to be considered a true positive.
- For each IoU threshold, take the mean of the APs across all 80 categories.
- Finally, compute the overall AP by averaging the AP values calculated at each IoU threshold.

The differences in AP calculation make it hard to directly compare the performance of object detection models across
the two datasets. The current standard uses the COCO AP due to its more fine-grained evaluation of how well a model
performs at different IoU thresholds.

#### 1.3. Non-Maximum Suppression (NMS)

Non-Maximum Suppression (NMS) is a post-processing technique used in object detection algorithms to reduce the
number of overlapping bounding boxes and improve the overall detection quality. Object detection algorithms typically
generate multiple bounding boxes around the same object with different confidence scores. NMS filters out redundant
and irrelevant bounding boxes, keeping only the most accurate ones. Algorithm 1 describes the procedure. Figure 4
shows the typical output of an object detection model containing multiple overlapping bounding boxes and the output
after NMS.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-3.png?raw=true)

### 2. YOLO: You Only Look Once (CVPR 2016)

YOLO by Joseph Redmon et al. was published in CVPR 2016. It presented for the first time a real-time end-to-end
approach for object detection. The name YOLO stands for "You Only Look Once," referring to the fact that it was
able to accomplish the detection task with a single pass of the network, as opposed to previous approaches that either
used sliding windows followed by a classifier that needed to run hundreds or thousands of times per image or the more
advanced methods that divided the task into two-steps, where the first step detects possible regions with objects or
regions proposals and the second step run a classifier on the proposals. Also, YOLO used a more straightforward output
based only on regression to predict the detection outputs as opposed to Fast R-CNN that used two separate outputs,
a classification for the probabilities and a regression for the boxes coordinates.

YOLOv1 unified the object detection steps by detecting all the bounding boxes simultaneously. 

To accomplish
this, YOLO divides the input image into a $S × S$ grid and predicts $B$ bounding boxes of the same class, along
with its confidence for $C$ different classes per grid element. Each bounding box prediction consists of five values:
$P_c, b_x, b_y, b_h, b_w$ where $P_c$ is the confidence score for the box that reflects how confident the model is that the box
contains an object and how accurate the box is. The $b_x$ and $b_y$ coordinates are the centers of the box relative to the grid
cell, and $b_h$ and $b_w$ are the height and width of the box relative to the full image. The output of YOLO is a tensor of
$S × S × (B × 5 + C)$ optionally followed by non-maximum suppression (NMS) to remove duplicate detections.

In the original YOLO paper, the authors used the PASCAL VOC dataset that contains 20 classes (C = 20); a grid
of 7 × 7 (S = 7) and at most 2 classes per grid element (B = 2), giving a 7 × 7 × 30 output prediction.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-4.png?raw=true)

#### YOLOv1 Architecture

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-5.png?raw=true)

#### Loss function

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-6.png?raw=true)

#### YOLOv1 Strengths and Limitations

<b> Strengths: </b> 

The simple architecture of YOLO, along with its novel full-image one-shot regression, made it much faster than the
existing object detectors allowing real-time performance.

<b> Limitation: </b>

- It could only detect at most two objects of the same class in the grid cell, limiting its ability to predict nearby
objects.

- It struggled to predict objects with aspect ratios not seen in the training data.

- It learned from coarse object features due to the down-sampling layers.

### 3. YOLOv2: Better, Faster, and Stronger (CVPR 2017)

#### Improvements:

- <b> Batch normalization </b> on all convolutional layers improved convergence and acts as a regularizer to reduce
overfitting.

- <b> High-resolution classifier: </b> Like YOLOv1, they pre-trained the model with ImageNet at 224 × 224. However,
this time, they finetuned the model for ten epochs on ImageNet with a resolution of 448 × 448, improving the
network performance on higher resolution input.

- <b> Fully convolutional: </b> They removed the dense layers and used a fully convolutional architecture.

- <b> Use anchor boxes to predict bounding boxes: </b> They use a set of prior boxes or anchor boxes, which are
boxes with predefined shapes used to match prototypical shapes of objects as shown in Figure below. Multiple
anchor boxes are defined for each grid cell, and the system predicts the coordinates and the class for every
anchor box. The size of the network output is proportional to the number of anchor boxes per grid cell.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-7.png?raw=true)

- <b> Dimension Clusters: </b> Picking good prior boxes helps the network learn to predict more accurate bounding
boxes. The authors ran k-means clustering on the training bounding boxes to find good priors. They selected
five prior boxes providing a good tradeoff between recall and model complexity.

- <b> Direct location prediction: </b> Unlike other methods that predicted offsets, YOLOv2 followed the same
philosophy and predicted location coordinates relative to the grid cell. The network predicts five bounding
boxes for each cell, each with five values $t_x, t_y, t_w, t_h$, and $t_o$, where to is equivalent to $P_c$ from YOLOv1
and the final bounding box coordinates are obtained.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-8.png?raw=true)

- <b> Finner-grained features: </b> YOLOv2, compared with YOLOv1, removed one pooling layer to obtain an output
feature map or grid of 13 × 13 for input images of 416 × 416. YOLOv2 also uses a passthrough layer that
takes the 26 × 26 × 512 feature map and reorganizes it by stacking adjacent features into different channels
instead of losing them via a spatial subsampling. This generates 13 × 13 × 2048 feature maps concatenated in
the channel dimension with the lower resolution 13 × 13 × 1024 maps to obtain 13 × 13 × 3072 feature maps.

- <b> Multi-scale training: </b> Since YOLOv2 does not use fully connected layers, the inputs can be different sizes. To
make YOLOv2 robust to different input sizes, the authors trained the model randomly, changing the input size
—from 320 × 320 up to 608 × 608— every ten batches.

#### YOLOv2 Architecture (Darknet-19)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-9.png?raw=true)

### 4. YOLOv3 (ArXiv 2018)

#### Improvements:

- <b> Bounding box prediction: </b> Like YOLOv2, the network predicts four coordinates for each bounding box $t_x, t_y, t_w$, and $t_h$; however, this time, YOLOv3 predicts an objectness score for each bounding box using logistic
regression. This score is 1 for the anchor box with the highest overlap with the ground truth and 0 for the rest
anchor boxes. YOLOv3, as opposed to Faster R-CNN, assigns only one anchor box to each ground truth
object. Also, if no anchor box is assigned to an object, it only incurs in classification loss but not localization
loss or confidence loss.
 
- <b> Class Prediction: </b> Instead of using a softmax for the classification, they used binary cross-entropy to train
independent logistic classifiers and pose the problem as a multilabel classification. This change allows
assigning multiple labels to the same box, which may occur on some complex datasets with overlapping
labels. For example, the same object can be a Person and a Man.

- <b> New backbone: </b> YOLOv3 features a larger feature extractor composed of 53 convolutional layers with residual
connections. 

- <b> Spatial pyramid pooling (SPP): </b> Although not mentioned in the paper, the authors also added to the backbone
a modified SPP block that concatenates multiple max pooling outputs without subsampling (stride = 1),
each with different kernel sizes k × k where k = 1, 5, 9, 13 allowing a larger receptive field. This version is
called YOLOv3-spp and was the best-performed version improving the $AP_{50}$ by 2.7%.

- <b> Multi-scale Predictions; </b> Similar to Feature Pyramid Networks,  YOLOv3 predicts three boxes at three
different scales. 

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-11.png?raw=true)

- <b> Bounding box priors: </b> Like YOLOv2, the authors also use k-means to determine the bounding box priors of
anchor boxes. The difference is that in YOLOv2, they used a total of five prior boxes per cell, and in YOLOv3,
they used three prior boxes for three different scales.

#### YOLOv3 Architecture

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-10.png?raw=true)


### Note:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-12.png?raw=true)

- The backbone is responsible for extracting useful features from the input image. It is typically a convolutional neural
network (CNN) trained on a large-scale image classification task, such as ImageNet. The backbone captures hierarchical
features at different scales, with lower-level features (e.g., edges and textures) extracted in the earlier layers and
higher-level features (e.g., object parts and semantic information) extracted in the deeper layers.

- The neck is an intermediate component that connects the backbone to the head. It aggregates and refines the features
extracted by the backbone, often focusing on enhancing the spatial and semantic information across different scales.
The neck may include additional convolutional layers, feature pyramid networks (FPN), or other mechanisms to
improve the representation of the features.

- The head is the final component of an object detector; it is responsible for making predictions based on the features
provided by the backbone and neck. It typically consists of one or more task-specific subnetworks that perform
classification, localization, and, more recently, instance segmentation and pose estimation. The head processes the
features the neck provides, generating predictions for each object candidate. In the end, a post-processing step, such as
non-maximum suppression (NMS), filters out overlapping predictions and retains only the most confident detections.

### 5. YOLOv4 (ArViv 2020)

YOLOv4 tried to find the optimal balance by experimenting with many changes categorized as bag-of-freebies and
bag-of-specials. Bag-of-freebies are methods that only change the training strategy and increase training cost but do not
increase the inference time, the most common being data augmentation. On the other hand, bag-of-specials are methods
that slightly increase the inference cost but significantly improve accuracy. Examples of these methods are those for
enlarging the receptive field, combining features, and post-processing 
among others.

#### Improvements:

- <b> An Enhanced Architecture with Bag-of-Specials (BoS) Integration </b>

The authors tried multiple architectures
for the backbone, such as ResNeXt50, EfficientNet-B3, and Darknet-53. The best-performing
architecture was a modification of Darknet-53 with cross-stage partial connections (CSPNet), and Mish
activation function as the backbone. For the neck, they used the modified version of
spatial pyramid pooling (SPP) from YOLOv3-spp and multi-scale predictions as in YOLOv3, but with
a modified version of path aggregation network (PANet) instead of FPN as well as a modified spatial
attention module (SAM). Finally, for the detection head, they use anchors as in YOLOv3. Therefore,
the model was called CSPDarknet53-PANet-SPP. The cross-stage partial connections (CSP) added to the
Darknet-53 help reduce the computation of the model while keeping the same accuracy. The SPP block, as
in YOLOv3-spp increases the receptive field without affecting the inference speed. The modified version of
PANet concatenates the features instead of adding them as in the original PANet paper.

- <b> Integrating bag-of-freebies (BoF) for an Advanced Training Approach </b> 

Apart from the regular augmentations such as random brightness, contrast, scaling, cropping, flipping, and rotation, the authors implemented
mosaic augmentation that combines four images into a single one allowing the detection of objects outside their
usual context and also reducing the need for a large mini-batch size for batch normalization. For regularization,
they used DropBlock that works as a replacement of Dropout but for convolutional neural networks
as well as class label smoothing. For the detector, they added CIoU loss and Cross mini-bath
normalization (CmBN) for collecting statistics from the entire batch instead of from single mini-batches as in
regular batch normalization.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-14.png?raw=true)

- <b> Self-adversarial Training (SAT) </b>

To make the model more robust to perturbations, an adversarial attack is
performed on the input image to create a deception that the ground truth object is not in the image but keeps
the original label to detect the correct object.

- <b> Hyperparameter Optimization with Genetic Algorithms </b> 

To find the optimal hyperparameters used for
training, they use genetic algorithms on the first 10% of periods, and a cosine annealing scheduler to alter
the learning rate during training. It starts reducing the learning rate slowly, followed by a quick reduction
halfway through the training process ending with a slight reduction.

#### YOLOv4 Architecture

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-5-13.png?raw=true)



