---
title: Paper note - [Week 2]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-12-29 11:11:14 +0700
categories: [Computer vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

## [Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?](https://arxiv.org/pdf/2206.07959.pdf)

[ICRA 2023]

### Motivation

- Focus on innovating new techniques for “lifting” features from the 2D image plane(s) onto the BEV plane
- What about RGB and Radar?

### Contribution

-  Elucidate high-impact factors in the design and training protocol of BEV perception models.

    ```
    Propose a model where the “lifting” step is parameter-free and does not rely on depth estimation: we simply define a 3D volume of coordinates over the BEV plane, project these coordinates into all images, and average the features sampled from the projected locations. When this simple model is tuned well, it exceeds the performance of state-of-the-art models while also being faster and more parameter-efficient.
    ```

- Demonstrate that radar data can provide a large boost to performance with a simple fusion method, and invite the community to re-consider this commonly-neglected part of the sensor platform.

### Method

Lifting strategy: Our model is “simpler” than related
work, particularly in the 2D-to-3D lifting step, which is
handled by (parameter-free) bilinear sampling. This replaces,
for example, depth estimation followed by splatting,
MLPs, or attention mechanisms. Our
strategy can be understood as “Lift-Splat without depth
estimation”, but as as illustrated in Figure 1, our implementation is different in a key detail: our method relies on sampling
instead of splatting. Our method begins with 3D coordinates
of the voxels, and takes a bilinear sample for each one. As
a result of the camera projection, close-up rows of voxels
sample very sparsely from the image (i.e., more spread out),
and far-away rows of voxels sample very densely (i.e., more
packed together), but each voxel receives a feature. Splattingbased methods begin with a 2D grid of coordinates,
and “shoot” each pixel along its ray, filling voxels intersected
by that ray, at fixed depth intervals. As a result, splatting
methods yield multiple samples for up-close voxels, and very
few samples (sometimes zero) for far-away voxels. As we
will show in experiments, this implementation detail has
an impact on performance, such that splatting is slightly
superior at short distances, and sampling is slightly superior
at long distances. In the experiments, we also evaluate a
recently-proposed deformable attention strategy, which is
similar to bilinear sampling but with learned sampling kernel
for each voxel (i.e., learned weights and learned offsets).

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-2-1.png?raw=true)

- Add more information

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-2-2.png?raw=true)

### Results

- Lifting strategy

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-2-3.png?raw=true)

- Input resolution


![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-2-4.png?raw=true)

- Fusion


![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-2-5.png?raw=true)

### Conclusion

- Slow
- Public code


