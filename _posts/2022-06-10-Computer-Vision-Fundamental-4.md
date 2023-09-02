---
title: Computer Vision Fundamental - [Part 4]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-06-13 11:11:11 +0700
categories: [Skill, Computer Vision]
tags: [Theory]
# img_path: /assets/img/post_assest/
render_with_liquid: false
---

# Chapter 4 - Estimating Point Correspondence
Goal: find point descriptors/characteristic image features to be able to identify keypoints accross several images from different views.

Things to consider:
- small vs. wide baseline case (small vs. large displacements)
	- higher fps makes things easier!
	- small baseline: plausible corresponding points are likely to be close (only search there)
- textured reconstructions can look misleadingly good (egg with face projected looks a lot like a human head)
- non-Lambertian materials (shiny: view point dependence of reflection)
- non-rigid transformations (person bending their head)
- partial occlusions (sunglasses) - a point may not have a correspondence



##### Small Deformation vs. Wide Baseline
- *small deformation case*: classical *optical flow estimation*.
	- Lucas/Kanade method (1981) (find correspondences sparsely)
	- Horn/Schunk method (1981) (find correspondences densely)

- *wide baseline stereo*: Select feature points and find an appropriate pairing of points

Comment: with improving methods, increasingly large deformations can be handled by optical flow estimation (e.g. coarse-to-fine approaches)


### Small Deformations: Optical Flow
The rigid transformation of a point $x_1$ in one image to $x_2$ in the other image is given by

$$x_2 = h(x_1) = \frac{1}{\lambda_2(X)}(R \lambda_1(X) x_1 + T)$$

##### Local approximation models for motion
This can be approximated locally e.g. by a *translational model*

$$h(x) = x + b$$

or an affine model

$$h(x) = Ax + b$$

The 2D affine model can also be written as

$$h(x) = x + u(x) = x + S(x) p = \begin{pmatrix} x  & y & 1 & & & \\ & & & x & y & 1\end{pmatrix}(p_1, p_2, p_3, p_4, p_5, p_6)^\top$$

for some parameters $p_i$ depending on the rotation/translation.

Affine models include much more types of motion (divergent motions, rotations etc.)

##### Optical Flow Estimation
The *optical flow* refers to the part of the motion that can be seen in the image plane (i.e. the projection of the real motion onto the image plane).

- **Lucas-Kanade**: sparse method (estimate motion field at certain points, under the assumption that the motion in a small neighborhood is *constant*)
- **Horn-Schunck**: dense method (estimate motion field at every pixel, under the assumption that the motion in a small neighborhood is *smooth*)

Lucas-Kanade was prefered at the time the methods were published because it is simpler and already was realtime-capable in the 80's. In more recent years, Horn-Schunck is becoming more popular ("now we have GPUs").

### The Lucas-Kanade Method

#### Some Assumptions we make
* *Brightness Constancy Assumption* (also optical flow constraint): every moving point has constant brightness. Formally, $I(x(t), t) = \text{const.} ~\forall t$.
	* This is *almost never* fulfilled. But often approximately
	* the equivalent formulation $$\frac{d}{dt}I(x(t), t) = \nabla I^\top \frac{dx}{dt} + \frac{\partial I}{\partial t} = 0$$is also called the (differential) optical flow constraint
* *Constant Motion in a Neighborhood Assumption*: the velocity of movement is constant in a neighborhood $W(x)$ of a point $x$: $$\nabla I(x', t)^\top v + \frac{\partial I}{\partial t}(x', t) = 0 \quad \forall x' \in W(x)$$

#### Lukas-Kanade (1981) formulation
Since the two assumptions are not exactly fulfilled usually, the method minimizes the least-squares error instead:
$$E(v) = \int_{W(x)} \vert \nabla I(x', t)^\top v + I_t(x', t)\vert^2 dx'$$
Comment: this would be done differently today (not quadratic e.g.). $E$ (cost function) is also called *energy*.

##### Solution
We get $\frac{dE}{dv} = 2Mv + 2q = 0$, where $M = \int_{W(x)} \nabla I \nabla I^\top dx'$ and $q = \int_{W(x)} I_t \nabla I dx'$.
If $M$ is invertible, the solution is $v = - M^{-1} q$.

##### Alternatives
Affine motion: basically, same technique. The cost function becomes

$$E(v) = \int_{W(x)} \vert \nabla I(x', t)^\top S(x') p + I_t(x', t)\vert^2 dx'$$

and is minimized with the same technique as above.

##### Limitations (translational version)
*Aperture problem*: e.g. for constant intensity regions (where $\nabla I(x) = 0, I_t(x) = 0$ for all points). To get a unique solution $b$, the *structure tensor* $M(x)$ must be invertible:

$$M(x) = \int_{W(x)} \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{pmatrix} \,dx'$$

If $M(x)$ is not invertible, but at least non-zero, at least the *normal motion* (motion in direction of the gradient) can be estimated.


#### Simple Feature Tracking with Lucas-Kanade
Assume: given $t$.

- for each $x \in \Omega$ compute the structure tensor $M(x)$
- for the points $x$ where $\det M(x) \geq 0$ (above treshold), compute the local velocity as $$b(x, t) = -M(x)^{-1} \begin{pmatrix} \int I_x I_t dx' \\ \int I_y I_t d x' \end{pmatrix}$$
- update points from $x$ to $x + b(x, t)$ and repeat for time $t + 1$.

Important point: a translation-only model works in small window, but on a larger window, or with a longer movement, we need a better model (e.g. affine).

### Robust Feature Point Extraction
Problem: unreliable to invert $M$ if it has a small determinant. Alternative: Förstner 1984, Harris & Stephens 1988 - *Harris corner detector*.
- use alternative structure tensor to detect good points
	- weight neighborhood by a Gaussian: $$M(x) = G_\sigma \nabla I \nabla I^\top = \int G_\sigma(x - x') \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{pmatrix}(x') \,dx'$$
	- select points for which $\det(M) - \kappa \,\text{trace}(M)^2 > \theta$

![Fig.3](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/harris-foerstner-detector.png?raw=true)

### Wide Baseline Matching
Problem: many points will have no correspondence in the second image. Wide baseline might be needed to counter *drift*, i.e. the accumulation of small errors (compute corredpondences again with larger distance in time).

- One needs to consider an affine model (translational is not good enough for wide baseline).
- To be more robust to illumination changes (typically greater in wide baseline): replace L2 error function by *normalized cross correlation*

##### Normalized Cross Correlation
The NCC for a given candidate transformation $h$ is
$$NCC(h) = \frac{
 \int_{W(x)} (I_1(x') - \bar{I}_1) (I_2(h(x')) - \bar{I}_2) \, dx'}
{
 \sqrt{\int_{W(x)} (I_1(x') - \bar{I}_1)^2 \, dx'  \int_{W(x)} (I_2(x') - \bar{I}_2)^2 \, dx'}}$$

where $\bar{I}_1, \bar{I}_2$ are average intensities of $W(x)$ ($\bar{I}_2$ depends on $h$). Subtracting averages leads to invariance wrt. additive intensity changes. Dividing by the intensity variances of the window leads to invariance to multiplicative changes.

Different interpretation: If we stack the normalized intensity values of a window in one vector, $v_i = \text{vec}(I_i - \bar{I}_i), i =1,2$, then $NCC(h) = \cos \angle (v_1, v_2)$.

##### Normalized Cross Correlation for affine transformation
Affine transformation $h(x) = Ax + d$: Find optimum $\arg\max_{A,d} NCC(A, d)$. Just insert the $h$ in the above formula to get the $NCC$. Efficiently finding optima is a challenge

##### Optical Flow Estimation with Deep Neural Networks
Deep NNs can also be used for correspondence estimation and have become more popular in recent years.
