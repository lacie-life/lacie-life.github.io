---
title: Homography Estimation
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-12-15 11:11:14 +0700
categories: [Theory]
tags: [Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

## Homography Estimation

### 1. Homography from camera parameters

#### a. Basic setup

Let's consider a point in the 3D world-view space as a 3-tuple


![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-1.png?raw=true)

The 3D world-view is captured onto a 2D image through a camera, placed in the world-view coordinates.

We can then map this 3D point to a point in an <i> arbitrary space </i>, as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-2.png?raw=true)

where $C_{int}$ is the intrinsic and $C_{ext}$ is the extrinsic camera matrix respectively. The point $(x_a, y_a, z_a)$ in the arbitrary space, can be mapped to the 2D image space by involving a scale factor as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-3.png?raw=true)

Thus, once we have a point in the arbitrary space, we can simply scale its coordinates to get the 2D coordinates in the (captured) image space.

#### b. The intrinsic matrix

Let us now see the form of $C_{int}$. Consider a camera with a focal length $f$ (in mm), with actual sensor size $(x_S, y_S)$ (in mm), and the width and the height of the captured image (effective sensor size) as $(w,h)$ (in pixels).
The optical centre $(o_x, o_y)$ of the camera is then $(w/2, h/2)$. We are now in a state to specify $C_{int}$ as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-4.png?raw=true)

One may thus observe that all entries in $C_{int}$ are in the units of pixels. The following may be seen as the effective focal lengths in pixels in $x$ and $y$ directions respectively

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-5.png?raw=true)

In practice, they are taken the same, since when it is calculated from the camera specifications, they have a very slight difference (≈ 0.2 to 0.5 %)

#### c. The Extrinsic Matrix

$C_{ext}$ consists of a rotation matrix $R$ and a translation matrix $T$ as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-6.png?raw=true)

The tuple $(T_x, T_y, T_z)$ indicates the translation of the camera in the world-space coordinates. Typically, we can consider that the camera has no $x$ and $y$ translation $(T_x = T_y = 0)$ and the height of the camera position from the ground (in mm) equates to $T_z$.

If $θ$, $φ$, $ψ$ be the orientation of the camera with respect to $x$, $y$ and $z$ axes respectively (as angles in radians), we may get $r_{ij} ; $i$, $j ∈ {1,2,3}$ as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-7.png?raw=true)

#### d. The Homography

The homography matrix H to be estimated is a 3 × 3 matrix, and encompasses the parts of both <i> intrinsic </i> and <i> extrinsic </i> camera matrices as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-8.png?raw=true)

The above can be directly established from the fact, that when we are looking for a planar surface in the world-view to compute the homography, $Z_w = 0$, and thus,

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-9.png?raw=true)

Hence, the homography $H$ will map a world-view point in the arbitrary space. This space suffices well, in case we just need to compute the distances between any two given points. However, in reality, the coordinates in the pixel space will be calculated by considering the scale factor as specified in Eq. (2).

### 2. Homography from co-planar points

#### a. Basic setup

Homography lets us relate two cameras viewing the same planar surface; Both the cameras and the surface that they view (generate images of) are located in the world-view coordinates. In other words, two 2D images are related by a <i> homography $H$ </i>, if both view the same plane from a <i> different angle </i>. The homography relationship does not depend on the scene being viewed.

Consider two such images viewing the same plane in the world-view.
Let $(x_1, y_1)$ be a point in the first image, and $(\hat x_1,\hat y_1)$ be the corresponding point in the second image. Then, these points are related by the <i> estimated </i> homography H, as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-10.png?raw=true)

Thus, any point in the first image may be mapped to a corresponding point in the second image through homography, and the operation may be seen as that of an image warp.

#### b. The homography

Let us parametrize the 3 × 3 homography matrix $H$ as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-11.png?raw=true)

Thus, the estimation of $H$ requires the estimation of 9 parameters. In other words, H has 9 degrees of freedom. If we choose two tuples of corresponding points, <b> [co-planar] </b> in their respective planes, as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-12.png?raw=true)

```
[co-planar] The homography relation is provable only under the co-planarity of the points, since everywhere, we are assuming that the z-coordinate of any point in any image is 1. In practice, for instance, one may thus choose four points on a floor, or a road, which indicate a nearly planar surface in the scene.
```

Then, from Eq. (8, 9, 10), we may solve the following to estimate $H$:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-13.png?raw=true)

Where $(\hat x_i, \hat y_i ) ∈ \hat T_1$ and $(x_i, y_i) ∈ T_1 for i, j ∈ {1,2,3,4}$. This will then translate to the following system of equations to be solved:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/homo-14.png?raw=true)

We now have 8 equations, which can be used to estimate the 8 degrees of freedom of $H$ (except $h_{33}$). For this, we would require that the 8 × 8 matrix above has a full rank (no redundant information), in the sense that none of the rows are linearly dependent. This implies that no three points in either of $T_1$ or $\hat T_1$ should be collinear.

We then need to tackle $h_{33}$. Note that in Eq. (13), if $h_{33}$ is pre-equated to 1, we would simply shift the entire set of $h_{ij}$ hyperplanes to another reference frame, but their directions would not change. Practically, we would thus, simply see a different $z_a$ value, while mapping a 2D image coordinate according to Eq. (8), which would subsequently get divided out in Eq. (9). Hence, we keep $h_{33} = 1$ in H, and Eq. (13) may then be solved using least-squares estimation.

In OpenCV, one can use the function <i> findHomography </i>, which does precisely the same, as delineated above. It accepts two tuples of four corresponding points, and computes the homography $H$ with $h_{33}$ always and strictly 1. Any 2D image point will then be mapped to a $z_a$ amplified version of the corresponding point in the other plane.

#### c. Homography with a Hypothetical Camera

In various applications, such as virtual advertising, absolute distance measurements for smart city planning, one needs to assume the presence of a hypothetical camera $C$, and compute a homography matrix, which can project any point in the observed scene to the plane of the image captured by $C$.

Imagining $C$ in the bird’s eye (top view) is a <b> [popular choice] </b>. In such a case, one may choose $T_1$ with four co-planar points in the observed scene, while the corresponding tuple $ \hat T_1$ can simply have four points as corners of a hypothetical rectangle, with a Euclidean coordinate system, centered around (0, 0). Any point in the scene can then be mapped to its bird’s eye view, i.e. how may it look from the top.

```
[popular choice] There has been a recent surge of research papers, which exploit the bird's eye view (BEV) for behavioural prediction and planning in autonmous driving.
```

Note here that homography-based mapping is only a warped version of the observed image, and that no new information in the scene is being synthesized. For instance, if we have only observed the frontal view of a person in the scene, its birds-eye view, won’t really start telling as to how the person’s hair is from the top; but it will only warp the frontal-view visible portion of his head in a way that it would roughly look like a top-view.

#### d. Negative values in projections with Homography

Note that while solving for $H$, there is no constraint that the projection points in the arbitrary space should be positive, i.e. $x_a$, $y_a$ and $z_a$ may be negative. Once scaled out by $z_a$, this would mean that a mapped point $(\hat x_i, \hat y_i )$ may be negative. 

This may look intuitively undesirable since the image coordinates are generally taken to be positive. However, this may be seen as only a reference axis shift, and after mapping the entire image, the amount of shift may be appropriately decided.













