---
title: Computer Vision Fundamental - [Part 2]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-06-11 11:11:11 +0700
categories: [Skill, Computer Vision]
tags: [Theory]
# img_path: /assets/img/post_assest/
render_with_liquid: false
---

# Chapter 2 - Representing a Moving Scene

### Origins of 3D Reconstruction
3D reconstruction is a classical ill-posed problem, as its solutions are not unique (most extreme example: imagine a photograph was pinned in front of the camera).

Two type of transformations are needed:
* *Perspective projection*: account for image formation process (i.e. specifics of camera)
* *Rigid-body motion*: represent movement of camera between frames

Perspective projection was studied by acient Greeks (Euclids ~400 BC) and in the Renaissance (Brunelleschi & Alberti, 1435). This lead to *projective geometry* (Desargues 1648, Monge in the 18th century).

Multi-view geometry was first treated by Erwin Kruppa in 1913: *Two views of five points are sufficient to determine a) the motion between the views and b) the 3D structure up to finitely many solutions.* The eight-point algorithm was proposed bei Longuet-Higgins in 1981, further work along these lines followed (three views 1987-1995, factorization techniques 1992).

The joint estimation of a) camera motion and b) 3D structure is called *structure and motion* or *visual SLAM*.

### 3D space and rigid body motion
##### Euclidean space terminology
Definitions:
* $\mathbb{E}^3$ is identified with $\mathbb{R}^3$ and consists of points $\mathbf{X} = (X_1, X_2, X_3)^\top$
* *bound vector*: $v = \mathbf{X} - \mathbf{Y} \in \mathbb{R}^3$, taking its endpoints into account
* *free vector*: if we don't take the endpoints into account
* *curve length*: $l(\gamma) = \int_0^1 |\gamma'(s)|\,ds$ for $\gamma: [0, 1] \to \mathbb{R}^3$

##### Skew-symmetric matrices
* $so(3)$ denote the space of *skew-symmetric matrices*
* *cross product*: $u \times v = (u_2 v_3 - u_3 v_2, \,u_3v_1 - u_1v_3, \,u_1v_2 - u_2v_1)^\top$. This vector is orthogonal to both $u$ and $v$.
* The hat operator $\widehat{~}: \mathbb{R}^3 \to so(3)$ models the cross product and defines an isomorphism between 3-dim. vectors and skew-symmetric matrices. Its inverse is denoted $\vee: so(3) \to \mathbb{R}^3$.

##### Rigid-body motions
A *rigid-body motion* is a family $g_t: \mathbb{R}^3 \to \mathbb{R}^3$, $t \in [0, T]$ which preserves norms and cross products:
* $||g_t(v)|| = ||v||, ~\forall v$  (distance between moving points stays the same)
* $g_t(u) \times g_t(v) = g_t(u \times v), ~ \forall u, v$ (orientation stays the same)
In other words: each $g_t$'s rotation part is in $SO(3)$ ($O(3)$ because of the norm preservation, and then $SO(3)$ because the cross product preservation implies a determinant of $1$). So $g_t(x) = Rx + T$ for some $R \in SO(3)$, $T \in \mathbb{R}^3$.

They are also *inner product preserving*, as $\langle u, v \rangle = \tfrac{1}{4}(||u+v||^2 - ||u-v||^2)$ (the *polarization identity*). As a consequence, they are *volume-preserving* because the preserve the *triple product*: $\langle g_t(u), g_t(v) \times g_t(w) \rangle = \langle u, v \times w \rangle$.


##### Exponential coordinate reparametrization of rotations
Motivation: $SO(3)$ is actually only 3-dimensional, not 9-dimensional.

Assume a trajectory $R(t): \mathbb{R} \to SO(3)$. Its derivative $R'(t)$ is connected to $R(t)$ by a linear differential equation, which leads a matrix exponential solution for $R(t)$:

$$R(t) R^\top(t) = I 
 \Rightarrow R'(t)R^\top(T) + R(t)R'^\top(t) = 0
 \Rightarrow R'(t) R^\top(t) = -(R'(t)R^\top(t))^\top
$$
So $R'(t) R^\top(t)$, i.e. of the form $R'(t) R^\top(t) = \widehat{w}(t) \in so(3)$, and therefore $R'(t) = \widehat{w}(t) R(t)$.

Now assume an infinitesimal rotation: If $R(0) = I$ then $R(dt) \approx I + \widehat{w}(0) dt$. This means that an infinitesimal rotation can be approximated by an element of $so(3)$.

### Lie Group SO(3) of Rotations
**Definitions**
* A *Lie group* is a smooth manifold that is also a group, s.t. the group operations $+$ and ${}^{-1}$ are smooth maps.
* An *algebra over a field $K$* is a $K$-vector space that additionally has a multiplication on $V$ (non-commutative in general)
* *Lie bracket*: $[w, v] = wv - vw$

Formulated the above with this terminology:
* $SO(3)$ is a Lie group
* $so(3)$ is a Lie algebra, and tangent space at the identity of the Lie group $SO(3)$

The mapping from a Lie algebra to its associated Lie group is called *exponential map*. Its inverse is called *logarithm*.

##### Exponential map $\exp: so(3) \to SO(3)$
Assume a rotation with **constant $\widehat{w}$**.  The transformation $R(t)$ is then described by the linear differential system of equations
$$
 \begin{cases}
R'(t) = \widehat{w} R(t) \\
R(0) = I
 \end{cases}
$$
This has the solution
$$R(t) = \exp(\hat{w} t) = \sum_{n = 0}^\infty \frac{(\hat{w}t)^n}{n!}$$
This describes a *rotation around the axis $w \in \mathbb{R}^3$, $||w|| = 1$, by an angle $t$*.

So the matrix exponential defines a map $\exp: so(3) \to SO(3)$.

##### Logarithm of SO(3)
The inverse map (actually an inverse map, the inverse is non-unique) is denoted by $\log: SO(3) \to so(3)$.

The vector $w$ s.t. $\hat{w} = \log(R)$ is given by
$$
||w|| = \cos^{-1} \bigg(\tfrac{\text{trace}(R) - 1}{2}\bigg),\quad
 \frac{w}{||w||} = \frac{1}{2 \sin(||w||)} (r_{32} - r_{23}, ~r_{13} - r_{31}, ~r_{21} - r_{12})^\top
$$

The length of $||w||$ corresponds to the rotation angle, and the normalized $\frac{w}{||w||}$ to the rotation axis. The non-uniqueness can be seen from the fact that increasing the angle by multiples of $2\pi$ gives the same $R$.

##### Rodrigues' Formula
For skew-symmetric matrices $\hat{w} \in so(3)$, the matrix exponential can be computed by *Rodrigues formula*:

$$\exp(\hat{w}) = I + \frac{\hat{w}}{||w||} \sin(||w||)
+ \frac{\hat{w}^2}{||w||^2} \big(1 - \cos(||w||)\big)$$

**Proof:** Denote $t = ||w||$ and $v = w/t$. First, prove that $\hat{v}^2 = v v^\top - I$ and $\hat{v}^3 = -\hat{v}$. This allows to write $\hat{v}^n$ in closed form as
$$
 \hat{v}^n = \begin{cases}
(-1)^{k-1} \cdot \hat{v}^2, &n=2k \\
(-1)^{k} \cdot \hat{v}, &n=2k+1
 \end{cases}
$$
Plugging this into the exponential series yields:
$$
 \sum_{n = 0}^\infty \frac{(\hat{v}t)^n}{n!}
= I 
+ \sum_{n \geq 1, n=2k} \frac{(-1)^{k-1} t^n \hat{v}^2}{n!}
+ \sum_{n \geq 1, n=2k+1} \frac{(-1)^{k} t^n \hat{v}}{n!}
= I + \sin(t) \hat{v} + (1 - \cos(t)) \hat{v}^2
$$

### Lie Group SE(3) of Rigid-body Motions
Recall: SE(3) are transformations $x \mapsto Rx + T$, expressed in homog. coordinates as
$$SE(3) = \bigg\{\begin{pmatrix}R & T \\ 0 & 1\end{pmatrix} \;\bigg\lvert\; R \in SO(3)\bigg\}.$$

Consider a continuous family of rigid-body motions
$$g: \mathbb{R} \to SE(3), ~g(t) = \begin{pmatrix}R(t) & T(t) \\ 0 & 1\end{pmatrix}$$

Then the inverse of $g(t)$ is 

$$g^{-1}(t) = \begin{pmatrix}
	R^\top(t) & - R^\top(t) T \\
	0 & 1
 \end{pmatrix}$$

##### Linear differential equation for twists
We want to get a similar representation as differential equation as for $SO(3)$, i.e. write $g'(t) = \widehat{\xi}(t) g(t)$. To achieve this, consider $g'(t) g^{-1}(t)$ and later mulitply with $g(t)$:

$$g'(t) g^{-1}(t) = \begin{pmatrix}
	R'(t) R(t) & T'(t) - R'(t) R^\top T \\
	0 & 0
 \end{pmatrix}$$
As for $SO(3)$, the matrix $R'(t)R^\top$ is skew-symmetric. We call it $\widehat{w}(t)$, further define $v(t) = T'(t) - \widehat{w}(t) T(t)$, and write
$$g'(t) g^{-1}(t)
=: \begin{pmatrix}
	\widehat{w}(t) & T'(t) - \widehat{w}(t) T \\
	0 & 0
 \end{pmatrix} 
=: \begin{pmatrix}
	\widehat{w}(t) & v(t) \\
	0 & 0
 \end{pmatrix}
=: \widehat{\xi}(t) \in \mathbb{R}^{4\times 4}$$

Multiplying with $g(t)$ yields:

$$g'(t) = \widehat{x}(t) g(t)$$

The 4x4 matrix $\widehat{x}(t)$ is called a *twist*. We can view it as tangent along $g(t)$.

##### Lie algebra se(3)
The set of all twists form the tangent space at the identity of $SE(3)$:

$$se(3) = \bigg\{ \widehat{\xi} = 
 \begin{pmatrix} \widehat{w} & \widehat{v} \\ 0 & 0 \end{pmatrix}
 \;\big\lvert\;
 \widehat{w} \in so(3), v \in \mathbb{R}^3
 \bigg\}$$

We define a *hat operator* $\wedge$ and a *vee operator* $\vee$ as before:
$$\begin{pmatrix} v \\ w \end{pmatrix}^{\wedge} = \begin{pmatrix} \widehat{w} & \widehat{v} \\ 0 & 0 \end{pmatrix}  \in \mathbb{R}^{4\times 4} \qquad
 \begin{pmatrix} \widehat{w} & \widehat{v} \\ 0 & 0 \end{pmatrix}^{\vee} = \begin{pmatrix} v \\ w \end{pmatrix} \in \mathbb{R}^6$$

$\widehat{\xi} \in se(3)$ is called a *twist*, $\xi \mathbb{R}^6$ its *twist coordinates*. The vector $v$ represents the *linear velocity*, the vector $w$ the *angular velocity*.

#####  Exponential map $\exp: se(3) \to SE(3)$
Assume the motion has constant twist $\widehat{\xi}$. We get the linear differential equation system
$$
 \begin{cases}
g'(t) = \widehat{\xi} g(t) \\
g(0) = I
 \end{cases}
$$
which has the solution $g(t) = \exp(\hat{\xi}t)$.

So the exponential for $SE(3)$ is defined as

$$\exp: se(3) \to SE(3), \widehat{\xi} \to \exp(\widehat{\xi}),$$
where the $\widehat{\xi} \in se(3)$ are called *exponential coordinates* for $SE(3)$.

One can show that the exponential has the closed-form expression
$$\exp(\widehat{\xi}) = \begin{pmatrix}
	\exp(\widehat{w}) & \tfrac{1}{||w||^2} \big( I - \exp(\widehat{w}) \widehat{w} v + w w^\top v  \big) \\
	0 & 1
 \end{pmatrix}$$
(in case $w \neq 0$; otherwise $\exp(\widehat{\xi}) = [I ~ v; 0 ~ 1]$). Note: this is the analogon to the Rodrigues formula; in turn, this formula requires the Rodrigues formula for computing $\exp(\widehat{w})$.


##### Logarithm of SE(3)
To show that for any element of $SE(3)$ with rotation/translation $R, T$,  we can find a corresponding twist in $se(3)$, we use the above closed-form expression for $\exp(\widehat{\xi})$: Clearly, we can find $w$ as $\log(R)$. Then we need to solve $\tfrac{1}{||w||^2} \big( I - \exp(\widehat{w}) \widehat{w} v + w w^\top v  \big) = T$ for $v$ (not detailed in the lecture), which yields that the desired vector $v$ exists.


### Representing camera motion
Assume a point $\mathbf{X}_0$ in *world-coordinates*: this is mapped by the transformation $g(t)$ to a point $\mathbf{X}(t)$. Note: we follow the convention that points are moved by the transformation rather than the camera itself. In 3d coordinates, we define
$$\mathbf{X}(t) = R(t) \mathbf{X}_0 + T(t)$$
and in homogeneous coordinates:
$$\mathbf{X}(t) = g(t) \mathbf{X}_0.$$
We use the same notation for homogeneous and 3d representations (usually if a $g$ comes into play, we mean homogeneous coordinates).

##### Notation for concatenation
If we have one motion from $t_1$ to $t_2$ and another from $t_2$ to $t_3$, we can write $\mathbf{X}(t_3) = g(t_3, t_1) \mathbf{X}_0 = g(t_3, t_2) g(t_2, t_1) \mathbf{X}_0$.

Also, it holds that $g^{-1}(t_2, t_1) = g(t_1, t_2)$.

##### Rules of velocity transformation
We want to find the velocity of a point:
$$\mathbf{X}'(t) = g'(t) \mathbf{X}_0 = g'(t) g^{-1}(t) \mathbf{X}(t)$$
But $g'(t) g^{-1}(t)$ is simply a twist:
$$\widehat{V}(t) = g'(t) g^{-1}(t) = \begin{pmatrix}
	\widehat{w}(t) & v(t) \\
	0 & 0
 \end{pmatrix} \in se(3)$$

So $\mathbf{X}'(t) = \widehat{V}(t) \mathbf{X}(t)$ in homog. coordinates, or $\mathbf{X}'(t) = \widehat{w}(t) \mathbf{X}(t) + v(t)$. This clarifies why $w$ represents angular velocity and $v$ represents linear velocity.

##### Adjoint map: transfer between frames
Suppose in another frame, the view is displaced relative to our frame by $g_{xy}$, i.e. $\mathbf{Y}(t) = g_{xy} \mathbf{X}(t)$. The velocity in this frame is

$$mathbf{Y}'(t) = g_{xy} mathbf{X}'(t) = g_{xy} \widehat{V}(t) \mathbf{X}(t) = g_{xy} \widehat{V} g^{-1}_{xy} \mathbf{Y}(t)$$

In other words, the relative velocity of points observed from the other frame is represented by the twist $\widehat{V}_y = g_{xy} \widehat{V} g_{xy}^{-1} =: \text{ad}_{g_{xy}}(\widehat{V})$. Here we introduced the *adjoint map*:

$$\text{ad}_g: se(3) \to se(3), \widehat{\xi} \mapsto g \widehat{\xi} g^{-1}.$$


### Euler angles
*Euler angles* are a way to parametrize rotations, and an alternative to exponential parametrization. They are related, however, as we will see.

How to parametrize the space of rotations? We can choose a basis $(\widehat{w}_1, \widehat{w}_2, \widehat{w}_3)$ of $so(3)$ (skew-symm. matrices). Then we can parametrize any rotation in *Lie-Cartan coordinates of the first kind* $\alpha$ wrt. this basis as follows:
$$\alpha: (\alpha_1, \alpha_2, \alpha_3) \mapsto \exp(\alpha_1 \widehat{w}_1 + \alpha_2 \widehat{w}_2  + \alpha_3 \widehat{w}_3)$$

Alternatively, we can paremetrize it in *Lie-Cartan coordinates of the second kind* $\beta$ wrt. the basis:
$$\beta: (\beta_1, \beta_2, \beta_3) \mapsto \exp(\beta_1 \widehat{w}_1) \exp(\beta_2 \widehat{w}_2)\exp(\beta_3 \widehat{w}_3)$$

If we choose $w_1 = (0, 0, 1)^\top, w_2 = (0, 1, 0)^\top, w_3 = (1, 0, 0)^\top$, i.e. rotations around the z/y/x axes, the coordinates $\beta_i$ are called *Euler angles*.

This shows that the Euler angles are just a fairly random way, among infinitely many ways, to parametrize rotations. Advantage of the first-kind Lie-Cartan coordinates: allows to stay in the Lie algebra as long as possible, where the group operation (matrix addition instead of multiplication) is less expensive.

### Summary
![Fig.1](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/chapter2-summary.png?raw=true)
