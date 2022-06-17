---
title: Computer Vision Fundamental - [Part 6]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-06-15 11:11:11 +0700
categories: [Sill, Computer Vision]
tags: [Theory]
img_path: /assets/img/post_assest/
render_with_liquid: false
---

# Chapter 6 - Reconstruction from Multiple Views
Not just two, but multiple views. Each new view gives 6 new parameters, but many more point measurements -> ratio params / measurements improves.

Different approaches:
- trifocal tensors ("trilinear relations" between three images, generalize Fundamental Matrix)
	- Textbooks: Faugeras and Luong 2001; Hartley and Zisserman 2003
- matrices instead of tensors
	- Textbook: Invitation to 3D vision


### Preimages
- Preimage of a point/line on the image plane: points that get projected to that point/line 
- Preimage of points/lines from multiple views: Intersection of preimages
$$\text{preimage}(x_1,\dots,x_m) = \bigcap_i \text{preimage}(x_i)$$

The preimage of multiple lines should be a line for the reconstruction to be consistent.

![Fig.1](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/multiView.png?raw=true)

Next denote time-dependent image coordinates by $x(t)$. Parametrize 3D lines in homog. coord. as $L = {X_0 + \mu V}$. $L$'s preimage is a plane $P$ with normal $\ell(t)$, $P = \text{span}(\hat{\ell})$.

The $\ell$ is orthogonal to points $x$ on $L$: $\ell(t) x(t) = \ell(t) K(t) \Pi_0 g(t) X = 0$ (why?)

Then $\lambda_i x_i = \Pi_i X$ (relation i-th image of point p <-> world coordinates $X$) and $\ell_i^\top \Pi_i X_0 = \ell_i^\top \Pi_i V = 0$ (relation i-th coimage of $L$ <-> world coordinates $X_0, V$)

### Modeling Multi-View with Point Features

Assume we have the world point $X$ represented by points $x_1, \dots, x_m$ in the $m$ images, with depths $\lambda_1, \dots, \lambda_m$. This is modeled by

$$\mathcal{I} \vec{\lambda} = \Pi X$$

or in more detail,

$$\mathcal{I} \vec{\lambda} = \begin{pmatrix} x_1 & & & \\ & x_2 & & \\ & & \ddots & \\ & & & x_m \end{pmatrix} \begin{pmatrix}\lambda_1 \\ \lambda_2 \\ \vdots \\ \lambda_m \end{pmatrix} = \begin{pmatrix} \Pi_1 \\ \Pi_2 \\ \vdots \\ \Pi_m \end{pmatrix} X = \Pi X$$

* $\vec{\lambda}$ is the *depth scale vector*
* $\Pi$ (3m x 4 matrix) is the *multiple-view projection matrix*, $\Pi_i = \Pi g_i$, and contains the i-th camera rotation as well as the projection
* $\mathcal{I}$ (3m x m matrix) is the *image matrix* and contains the "2d" (homogenous) coordinates of projections $x_i$

##### The Rank Constraint

Rewrite to obtain:

$$N_p u = 0, \qquad \text{ where } N_p := [\Pi, \mathcal{I}],  u :=[X; -\vec{\lambda}]$$

$N_p$ is a $3m \times (m+4)$ matrix. Since $u$ is in the null space of $N_p$, we get the **rank constraint**:

$$\text{rank}(N_p) \leq m+3$$

There exists a reconstruction iff this rank constraint holds. Compare with epipolar constraint: contraint is on matrix that only includes camera params and 2d coords, but **no 3d coords**.

##### Writing Rank Constraint more compactly
Define $\mathcal{I}^\bot$, where in $\mathcal{I}$, each $x_i$ is substituted by $\widehat{x_i}$ (a 3m x 3m matrix). It annihilates $\mathcal{I}$: $\mathcal{I}^\bot \mathcal{I} = 0$. By multiplying the above $\mathcal{I} \vec{\lambda} = \Pi X$ with it:

$$\mathcal{I}^\bot \Pi X = 0$$

There exists a reconstruction iff $W_p = \mathcal{I}^\bot \Pi$ does not have full rank:

$$\text{rank}(W_p) \leq 3.$$

Note: $W_p$ has the form

$$W_p = [\widehat{x_1} \Pi_1; \dots; \widehat{x_m} \Pi_m]$$

### Modeling Multi-View with Line Features
Intuition: from a line in two views only, we can't say anything about the camera motion, because any two line preimages intersect. We can get results from more views.

Saw already:  $\ell_i^\top \Pi_i X_0 = \ell_i^\top \Pi_i V = 0$ for the coimages $\ell_i$ of a line $L$ with base $X_0$, direction $V$.

Define matrix $W_l = [\ell_1^\top \Pi_1 ; \dots; \ell_m^\top \Pi_m]$ (m x 4 matrix). It maps both $X_0$ and $V$ to 0; these two vectors are linearly independent ($X_0[4] = 1, V[4] = 0$). We get a new rank constraint:

$$\text{rank}(W_l) \leq 2$$


### Rank Constraints: Geometric Interpretation

##### Points case
For points, we had $W_p X = 0$, $W_p = [\widehat{x_1} \Pi_1; \dots; \widehat{x_m} \Pi_m]$ (3m x 4 matrix). There are 2m lin. indep. rows, which can be interpreted as the normals of 2m planes, and $W_p X = 0$ expresses that $X$ is in the intersection of these planes. The $2m$ planes have a unique intersection iff $\text{rank}(W_p) = 3$.

![Fig.2](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/fourPlanes-rank-constraint.png?raw=true)


##### Lines case
The constraint is only meaningful (i.e. actually a constraint) if $m > 2$. 


### Multiple-View Matrix of a Point
Goal: further compatify constraints.
We are in coordinate frame of first camera; i.e. $\Pi_1 = [I, 0], \Pi_2 = [R_2, T_2], \dots, \Pi_m = [R_m, T_m]$.

Define $D_p = \begin{pmatrix} \widehat{x_1} & x_1 & 0 \\ 0 & 0 & 1\end{pmatrix}$ (4 x 5 matrix, full rank). Multiply with $W_p$ (3m x 4 matrix) to get a 3m x 5 matrix, drop the first three rows and columns and call the submatrix $M_p$:

$$M_p = \begin{pmatrix}
	\widehat{x_2} R_2 x_1 & \widehat{x_2} T_2 \\
	\widehat{x_3} R_3 x_1 & \widehat{x_3} T_3 \\
	\vdots & \vdots \\
	\widehat{x_m} R_m x_1 & \widehat{x_m} T_m \\
 \end{pmatrix}$$
  
  $M_p$ is a 3(m-1) x 2 matrix. Now: $\text{rank}(W_p) \leq 3 \Leftrightarrow \text{rank}(M_p) \leq 1$, i.e. the two columns are linearly dependent (easy to check and work with!). Proof: 
  
  ![Fig.3](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/matrix_WpDp.png?raw=true)
  
  $M_p$ is the **multiple-view matrix**. Summary: there exists a reconstruction iff the matrices $N_p, W_p, M_p$ satisfy:
  
  $$\text{rank}(M_p) = \text{rank}(W_p) - 2 = \text{rank}(N_p) - (m+2) \leq 1$$
  
  ##### Geometric interpretation of Multiple-View Matrix
  
  The rank constraint implies that the two columns of $M_p$ are linearly dependent. In fact, even $$\lambda_1 \widehat{x_i} R_i x_1 + \widehat{x_i} T_i = 0, i = 2, \dots, m:$$
  So the scaling factor is equal to the depth value $\lambda_1$.
  
  (Proof: from the projection equation we know $\lambda_i x_i = \lambda_1 R_i x_1 + T_i$, hence $\lambda_1 \widehat{x_i} R_i x_1 + \widehat{x_i} T_i = 0$.)
  
  ##### $M_p$ => Epipolar (bilinear) constraints
  Goal: if we consider only a pair of images, the epipolar constraint should emerge from $M_p$.
  
  Proof - *linear dependence of $\widehat{x_i} R_i x_1$ and $\widehat{x_i} T_i$ implies epipolar constraint $x_i^\top \hat{T}_i R_i x_1 = 0$*:
  
$\widehat{x_i} T_i$ and $\widehat{x_i} R_i x_1$ are each normals to planes spanned by $x_i, T_i$ and $x_i, R_ix_1$, respectively. Linear dependence of these normals implies: => $x_i, T_i, R_i x_1$ live in the same plane (*coplanar*). Therefore $x_i^\top \hat{T}_i R_i x_1 = x_i^\top \hat{T}_i R_i x_1 = 0$.

#####  $M_p$ <=> Trilinear constraints
**Theorem**: *a matrix $M = [a_1 b_1; \dots; a_n b_n]$ with $a_i, b_i \in \mathbb{R}^3$ is rank-deficient <=> $a_i b_j^\top - b_i a_j^\top = 0$ for all $i, j$.*

Applied to $M_p$, this yields the *trilinear constraints*:

$$\widehat{x_i} (T_i x_1^\top R_j^\top - R_i x_1 T_j^\top) \widehat{x_j} = 0, ~\forall i, j \in [n] \qquad \text{(trilinear constraints)}$$

Different than the epipolar constraints, the trilinear constraints actually characterize the rank constraint on $M_p$. Each constraint couples *three* images: one can show that constraints on pairs of images cannot capture all the information from $m$ images, but these trilinear constraints can.

Note: we can also obtain the epipolar constraints directly from the trilinear constraints in non-degenerate cases.

Question: what does the "3 x 3 = 9 scalar trilinear equations" part mean?

##### Uniqueness of the Preimage
This slide was skipped ("a little bit to technical")

##### Degenerate cases
If $\widehat{x_j} T_j = \widehat{x_j} R_j x_1 = 0$ for some view $j$, then the epipolar constraints cannot be obtained from the trilinear constraints; also the equivalence "trilinear constraints <=> rank constraint" does not hold in degenerate cases.
1. If between three images, each pair of epipolar constraints is fulfilled, they determine a unique preimage $p$ - except if all three lines $o_i x_i$ between optical center and image point lie in the same plane.

 ![Fig.4](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/degeneracies-epipolar.png?raw=true)

2. If between three images, all three trilinear constraints hold (3 out of 9 are different considering symmetry), the determine a unique preimage $p$ - except if the three lines $o_i x_i$ are collinear.


In the example where all optical centers lie on a line, going from bilinear to trilinear constraints solves the problem.

##### Summary: Rank of $M_p$
* $M_p$ has rank 2 => no point correspondence; empty preimage.
* $M_p$ has rank 1 => point correspondence + *unique* preimage
* $M_p$ has rank 0 => point correspondence, but non-unique preimage

## Multi-View Reconstruction

Two approaches: 
1. cost-function based: maximize some objective function subject to the rank condition => non-lin. opt. problem: analogous to bundle adjustment
2. decouple structure and motion, like in the 8-point algorithm. Warning: not necessarily practical, since not necessarily optimal in the presence of noise + uncertainty (like the 8-point algorithm)

Approach 2 is called *factorization approach* (because it factors - i.e. decouples - the problem)

### Factorization Approach for Point Features
Assume: $m$ images $x_1^j, \dots, x_m^j$ each of points $p^j$, $j \in [n]$.
 
 Rank constraint => columns of $M_{p^j}$ dependent => (first column) + $\alpha^j$ (second column) = 0. As seen above, $\alpha^j = 1/\lambda_1^j$.
 
 ![Fig.5](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/Pasted-image-20210608164310.png?raw=true)
 
 This equation is linear in the camera motion parameters $R_i, T_i$, and can be written as:
 
 $$P_i \begin{pmatrix} R_i^s \\ T_i \end{pmatrix}
 = \begin{pmatrix} 
x_1^1{}^\top \otimes \widehat{x_i^1} & \alpha^1 \widehat{x_i^1} \\
x_1^2{}^\top \otimes \widehat{x_i^2} & \alpha^2 \widehat{x_i^2} \\
 \vdots & \vdots \\
x_1^n{}^\top \otimes \widehat{x_i^n} & \alpha^n \widehat{x_i^n}  \end{pmatrix}
 \begin{pmatrix} R_i^s \\ T_i \end{pmatrix}
= 0 \in \mathbb{R}^{3n}
$$

Here simply things were re-arranged, the $R$ and $T$ matrices stacked in one long vector.
One can show: $P_i \in \mathbb{R}^{3n \times 12}$ has rank 11, if more than 6 points (in general position) are given! (Intuition behind 6: 3n rows for n images, but only 2 out of three are lin. indep.)

=> one-dim. null space => projection matrix $\Pi_i = (R_i, T_i)$ given up to scalar factor!

In practice: use > 6 points, compute solution via SVD.

Like 8-point algorithm: not optimal in the presence of noise and uncertainty.

##### Decoupling compared to 8-point algorithm
Difference from 8-point algorithm: structure and motion not fully decoupled, since the 1/depth parameters $\alpha$ are needed to construct $P_i$. However, structure and motion can be iteratively estimated by estimating motion from a structure estimate, and vice versa, until convergence. Advantage: each step has a closed-form solution. This could be initialized by an 8-point algorithm reconstruction and further improve on it using the multi-view information.

Least-squares solution to find $\alpha_j$ from $R_i, T_i$: 

$$\alpha^j = - \frac{\sum_{i=2}^m (\widehat{x_i^j} T_i)^\top \widehat{x_i^j} R_i x_1^j}{\sum_{i=2}^m || \widehat{x_i^j} T_i || ^2}$$

Another interesting point: Estimating $Pi_i = (R_i, T_i)$ only requires two frames 1 and $j$, but estimating $\alpha$ requires all frames.

QUESTION: Don't we get $\alpha_j$ from $M_{p_j}$? (No... we need $R, T$ to get $M_{p_j}$).


### The Multi-View Matrix for Lines
Recall:  $\ell_i^\top \Pi_i X_0 = \ell_i^\top \Pi_i V = 0$ for the coimages $\ell_i$ of a line $L$ with base $X_0$, direction $V$; we constructed the multi-view matrix for lines:
$$W_l = [\ell_1^\top \Pi_1 ; \dots; \ell_m^\top \Pi_m] \in \mathbb{R}^{m \times 4}$$

The rank constraint is that $W_l$ should have rank at most 2, since $W_l X_0 = W_l V = 0$.  Goal: find more compact representaion; assume $\Pi_1 = (I, 0)$, i.e. first camera is in world coordinates.

Trick: multiply $W_l$ by 4x5 matrix $D_l$ s.t. last four columns of first row become zero, but keep rank the same.

![Fig.6](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/matrices-WlDl-lines.png?raw=true)

Now since the first column must be lin. indep. because of the zeros in the first row, and the matrix has rank at most 1, the submatrix starting $(W_l D_l)[2:, 2:]$ must have rank 1. This submatrix is called the *multi-view matrix for lines*.

$$M_l = \begin{pmatrix}
 \ell_2^\top R_2 \widehat{\ell_1} & \ell_2^\top T_2 \\
 \vdots & \vdots \\
 \ell_m^\top R_m \widehat{\ell_1} & \ell_m^\top T_m
 \end{pmatrix} \in \mathbb{R}^{(m-1) \times 4}$$

The previous rank-2-constraint can be characterized by a rank-1-constraint on $M_l$: A meaningful preimage of $m$ observed lines can only exist if

$$\text{rank}(M_l) \leq 1.$$

In other words: all rows and all columns must be linearly dependent.


##### Trilinear Constraints for a Line (from the Rows)
Since rows of $M_l$ are lin. dep., we have for all $i, j$: $\ell_i^\top R_i \widehat{\ell_1} \sim \ell_j^\top R_j \widehat{\ell_1}$. This states that the three vectors $R_i^\top \ell_j$, $R_j^\top \ell_j$, $\ell_1$ are coplanar. So $R_i^\top \ell_i$ is orthogonal to the cross product of $R_j^\top \ell_j$ and $\ell_1$, which leads to:

$$\ell_i^\top R_i \widehat{\ell_1} R_j^\top \ell_j = 0$$

Note: this constraint only contains the rotations, not the translations! (Observing lines allows us to directly put constraints on the rotation alone.)

By the same rank-deficiency lemma from before, we get that the linear dependency of the i-th and j-th row is equivalent to

$$\ell_j^\top T_j \ell_i^\top R_i \widehat{\ell_1} - \ell_i^\top T_i \ell_j^\top R_j \widehat{\ell_1} = 0$$

This relates the first, i-th and j-th images.

Both trilinear constraints are equivalent to the rank constraint if $\ell_i^\top T_i \neq 0$.

##### Generality of three-line constraints
Any multiview constraint on lines can be reduced to constraints which involve only three lines at a time. (Argument via 2x2 minors of matrix: see slides)


### Characterization of Unique Preimages for Lines
**Lemma:** *Given three camera frames with distinct optical centers and $\ell_1, \ell_2, \ell_3 \in \mathbb{R}^3$ represent three images lines, then their preimage $L$ is uniquely determined if*
$$
 \ell_i^\top T_{ji} \ell_k^\top R_{ki} \widehat{\ell_i} - \ell_k^\top T_{ki} \ell_j^\top R_{ji} \widehat{\ell_i} = 0
 \quad \forall i, j, k = 1, 2, 3,
$$
*except for one degenerate case: The only degenerate case is that in which the preimages of all $\ell_i$ are the same plane.*

Note: this constraint combines the two previous trilinear constraints.

Equivalent formulation using the rank constraint:

**Theorem:** *Given $m$ vectors $\ell_i$ representing images of lines w.r.t. $m$ camera frames, they correspond to the same line in space if the rank of $M_l$ relative to any of the camera frames is 1. If its rank is 0 (i.e. $M_l=0$, the line is determined up to a plane on which then all the camera centers must lie.*


## Summary of Multi-View Chapter
|       | (Pre)image                   | coimage                   | Jointly                   |
|-------|------------------------------|---------------------------|---------------------------|
| Point | $\text{rank}(N_p) \leq m+3$  | $\text{rank}(W_p) \leq 3$ | $\text{rank}(M_p) \leq 1$ |
| Line  | $\text{rank}(N_l) \leq 2m+2$ | $\text{rank}(W_l) \leq 2$ | $\text{rank}(M_l) \leq 1$ |

The rank constraints guarantee the existence of unique preimages in non-degenerate cases. 
