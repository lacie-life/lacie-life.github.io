---
title: Paper note 6 - PointPillars - Fast Encoders for Object Detection from Point Clouds
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-13 11:11:14 +0700
categories: [Computer Vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

[PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

# 1. Problem

- VoxelNet and PointNet is to slow.
- SECOND was improved but still slow.

=> <b> PointPillars: </b> a method for object detection in 3D that enables end-to-end learning with
only 2D convolutional layers. PointPillars uses <b> a novel encoder </b> that learn features on pillars (vertical columns) of the
point cloud to predict 3D oriented boxes for objects. There
are several advantages of this approach. 

- First, by learning
features instead of relying on fixed encoders, PointPillars
can leverage the full information represented by the point
cloud. 

- Further, by operating on pillars instead of voxels
there is no need to tune the binning of the vertical direction by hand. 

- Finally, pillars are highly efficient because all
key operations can be formulated as 2D convolutions which
are extremely efficient to compute on a GPU. 

An additional benefit of learning features is that PointPillars requires no
hand-tuning to use different point cloud configurations. For
example, it can easily incorporate multiple lidar scans, or
even radar point clouds.

# 2. PointPillars Network

PointPillars accepts point clouds as input and estimates
oriented 3D boxes for cars, pedestrians and cyclists. It consists of three main stages (Figure 2): 

(1) A feature encoder
network that converts a point cloud to a sparse pseudoimage; 

(2) A 2D convolutional backbone to process the
pseudo-image into high-level representation;

(3) A detection head that detects and regresses 3D boxes.

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-1.png?raw=true)

## 2.1. Pointcloud to Pseudo-Image

To apply a 2D convolutional architecture, we first convert the point cloud to a pseudo-image.
We denote by $l$ a point in a point cloud with coordinates
$x$, $y$, $z$ and reflectance $r$. As a first step the point cloud
is discretized into an evenly spaced grid in the x-y plane,
creating a set of pillars $P$ with $|P| = B$. Note that there is
no need for a hyper parameter to control the binning in the
z dimension. The points in each pillar are then augmented
with $x_c$, $y_c$, $z_c$, $x_p$ and $y_p$ where the $c$ subscript denotes
distance to the arithmetic mean of all points in the pillar and
the $p$ subscript denotes the offset from the pillar $x$, $y$ center.
The augmented lidar point $l$ is now $D = 9$ dimensional.
The set of pillars will be mostly empty due to sparsity
of the point cloud, and the non-empty pillars will in general
have few points in them. For example, at $0.162 m^2$ bins
the point cloud from an HDL-64E Velodyne lidar has 6k-9k
non-empty pillars in the range typically used in KITTI for
∼ 97% sparsity. This sparsity is exploited by imposing a
limit both on the number of non-empty pillars per sample
$(P)$ and on the number of points per pillar (N) to create a
dense tensor of size $(D, P, N)$. If a sample or pillar holds
too much data to fit in this tensor the data is randomly sampled. Conversely, if a sample or pillar has too little data to
populate the tensor, zero padding is applied.

Next, we use a simplified version of PointNet where,
for each point, a linear layer is applied followed by BatchNorm and ReLU to generate a $(C, P, N)$ sized
tensor. This is followed by a max operation over the channels to create an output tensor of size $(C, P)$. Note that the
linear layer can be formulated as a 1x1 convolution across
the tensor resulting in very efficient computation.
Once encoded, the features are scattered back to the
original pillar locations to create a pseudo-image of size
$(C, H, W)$ where $H$ and $W$ indicate the height and width
of the canvas.

## 2.2. Backbone

We use a similar backbone as VoxelNet and the structure is
shown in Figure 2. The backbone has two sub-networks:
one top-down network that produces features at increasingly small spatial resolution and a second network that performs upsampling and concatenation of the top-down features. The top-down backbone can be characterized by a series of blocks $Block(S, L, F)$. Each block operates at stride
$S$ (measured relative to the original input pseudo-image).
A block has $L$ 3x3 2D conv-layers with $F$ output channels,
each followed by BatchNorm and a ReLU. The first convolution inside the layer has stride $\frac{S}{S_{in}}$
to ensure the block
operates on stride $S$ after receiving an input blob of stride
$S_{in}$. All subsequent convolutions in a block have stride 1.
The final features from each top-down block are combined through upsampling and concatenation as follows.
First, the features are upsampled, $Up(S_{in}, S_{out}, F)$ from an
initial stride $S_{in}$ to a final stride Sout (both again measured
wrt. original pseudo-image) using a transposed 2D convolution with $F$ final features. Next, BatchNorm and ReLU
is applied to the upsampled features. The final output features are a concatenation of all features that originated from
different strides.

## 2.3. Detection Head

In this paper, we use the Single Shot Detector (SSD)
setup to perform 3D object detection. Similar to SSD, we
match the priorboxes to the ground truth using 2D Intersection over Union (IoU). Bounding box height and
elevation were not used for matching; instead given a 2D
match, the height and elevation become additional regression targets.

# 3. Implementation Details

## 3.1. Network

Instead of pre-training our networks, all weights were
initialized randomly using a uniform distribution.

The encoder network has C = 64 output features. The
car and pedestrian/cyclist backbones are the same except
for the stride of the first block (S = 2 for car, S = 1 for
pedestrian/cyclist). Both network consists of three blocks,
Block1(S, 4, C), Block2(2S, 6, 2C), and Block3(4S, 6,
4C). Each block is upsampled by the following upsampling
steps: Up1(S, S, 2C), Up2(2S, S, 2C) and Up3(4S, S, 2C).
Then the features of Up1, Up2 and Up3 are concatenated
together to create 6C features for the detection head.

## 3.2. Loss

We use the same loss function as SECOND.  Ground truth boxes and anchors are defined by
$(x, y, z, w, l, h, θ)$. The localization regression residuals between ground truth and anchors are defined by:

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-2.png?raw=true)

where $x^{gt}$ and $x^a$ are respectively the ground truth and anchor boxes and $d^a = \sqrt{ (w^a)^2 + (l^a)^2}$. The total localization loss is

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-3.png?raw=true)


Since the angle localization loss cannot distinguish
flipped boxes, we use a softmax classification loss on the
discretized directions , $L_{dir}$, which enables the network
to learn the heading.

For the object classification loss, we use thye focal loss:

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-5.png?raw=true)


where $p^a$
is the class probability of an anchor. We use the
original paper settings of $α = 0.25$ and $γ = 2$. The total
loss is therefore:

![Network overview](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note-6-6.png?raw=true)


where Npos is the number of positive anchors and $β_{loc} = 2$, $β_{cls} = 1$, and $β_{dir} = 0.2$.
To optimize the loss function we use the Adam optimizer
with an initial learning rate of 2 ∗ 10−4
and decay the learning rate by a factor of 0.8 every 15 epochs and train for 160
epochs. We use a batch size of 2 for validation set and 4 for
our test submission.

# 4. Experiments

- [Code](https://github.com/nutonomy/second.pytorch)



