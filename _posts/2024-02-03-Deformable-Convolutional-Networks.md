---
title: Deformable Convolutional Networks
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2024-02-03 11:11:14 +0700
categories: [Theory]
tags: [Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

## Deformable Convolutional Networks

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-1.png?raw=true)

### 1. Introduction

A key issue with traditional convolutional neural networks (CNNs) is their inability to <b> adapt </b> to geometric variations and transformations. Using <b> fixed </b> geometric structures such as: fixed filters, fixed pooling and fixed RoI spatial bins, limits any flexibility within the network.

A way that CNNs attempt to solve this problem is by augmenting existing data samples through various transformations. This, however, requires additional computing power and more complex model parameters. Equally, this type of network will quickly crumble when facing <b> unknown </b> transformations due to a lack of generalisation in the model.

As a result, a more feasible way of tackling this problem is through the introduction of deformable convolutional networks, which attempt and succeed in adapting to unprecedented transformations. This article aims to explore two such modules: <b> deformable convolutions </b> and <b> deformable RoI pooling </b>.

- <b> Deformable Convolutions </b>: Allows the filter to dynamically adjust its sampling locations with learnable offsets so that a better spatial relationship of the input data may be modelled.

- <b> Deformable RoI Pooling </b>: The RoI is divided into a fixed number of bins with a learnable offset allowing for a more dynamic pooling operation. This lets the model focus on more relevant spatial locations within the RoI.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-2.png?raw=true)

The ability of free form deformation as illustrated in the image above clearly demonstrates the power deformable convolutional networks can have in generalising different transformations.

Now, before we dive into the article, I highly advise that you are first familiar with traditional convolutions and RoI pooling, if not already.

### 2. Deformable Convolutions

Before we immediately dive into deformable convolutions, let’s quickly revise regular convolutions and build up from there.

A convolution is an operation used to extract and capture features of an image such as edges, shapes and also abstract features undetectable to the human eye.

The 2D convolution is an operation that uses a regular grid <b> $R$ </b> that has weights <b> $w$ </b> and is sampled over an input feature map. The summation of all the sampled values equates to the convolution’s output result.

In other words, a 2D convolution is the dot product between the filter and corresponding input feature map values.

The grid <b> $R$ </b> and its corresponding weights <b> $w$ </b> are shown below.

Remember that in image processing, the coordinate axis is often defined from the top left-hand corner of the image. Equally, please note that in the grid, position $(i, j)$ corresponds to weight $w_{ji}$ (matrix notation), not $w_{ij}$.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-3.png?raw=true)

Then, for each location $p_0$ on the output feature map $y$, its value can be defined mathematically like below. In this equation, $p_n$ enumerates all locations in grid <b> $R$ </b>.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-4.png?raw=true)

Now, that’s it for regular convolutions. Let’s now extend this model.

In <b> deformable convolutions </b>, we introduce a grid <b> $R$ </b>  that is augmented by offsets $ Δ p_n $. This essentially means that the grid <b> $R$ </b> becomes irregular and grid boxes shift ever so slightly.

This can be easy to visualise with the below diagram, along with the corresponding equation for deformable convolutions.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-5.png?raw=true)

As you can see in the diagram, a regular convolution will sample locations from the receptive field shown in blue. However, <b> in deformable convolutions </b>, sampling occurs on the irregular offset locations $ p_n + Δ p_n $ shown in green. In practice, the offset $ Δp_n $ is typically fractional.

If $ Δp_n $ is fractional, then how can we know the value of $ x(p_0 + p_n + Δp_n) $?

Well, we will never know with absolute certainty, but, we can make very accurate estimates by using <b> bilinear interpolation </b>.

<i> <b> Bilinear interpolation </b> is a method used to interpolate functions of two variables. In other words, given a rectangle with only the corner values known, bilinear interpolation allows us to estimate the value of the function at any point within the rectangle. </i>

In the case of deformable convolutions, this is represented by a bilinear interpolation kernel $G(…)$ where $p$ denotes an arbitrary location ($p = p_0 + p_n + Δp_n$) and q enumerates all original spatial locations in the feature map $x$.

The implementation of this can be seen in the below equations. Accompanying this, is a visual example where a pixel has been offset to the location in green. I will provide some further explanation next.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-6.png?raw=true)

Brief Explanation about the Example:

- In the image above, there are four pixels with their values shown.
- In green, an offset pixel <b> $p$ </b> with a fractional location is shown. Note that <b> $p$ </b>’s original pixel location is <b> not important </b> as bilinear interpolation depends only on surrounding pixels.
- We must use the <b> same reference point </b> for each pixel when measuring the <b> offset distance </b> between pixels. In this example, I chose the top left-hand corner of each pixel (other points such as top right, bottom left or centre, could have also been chosen).
- <b> $ G(q, p) = 0 $ </b> for all pixels not in the offset pixel’s immediate surrounding. This is because <b> $ g(a, b) = max(0, 1 - |a - b|) = 0 $ </b> as <b> $ 1 - |a - b| < 0 $ </b> for all pixels $q$ further that one pixel length from the offset pixel $p$. This is why we <b> only use four pixels </b> for bilinear interpolation.
- Substituting all the values gives us an approximate pixel value of 5, which makes sense if we look at it visually.

I hope you have understood everything up until this point. Now, I will briefly discuss its  <b> implementation </b> within the convolutional layer with reference to the below image.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-7.png?raw=true)

In deformable convolutions, a regular filter is applied over the input feature map, producing a standard output feature map. Simultaneously, an <b> offset field </b> is generated, representing <b> 2D offsets for N filters </b> (N channels for both the x and y direction). These offsets serve as predictions for the adjustments needed in the next forward pass during training and <b> do not represent the current filter offsets </b>. During training, the convolutional filter and offsets are learned simultaneously, with backpropagation applied on the bilinear operations for offset learning. The gradient for these bilinear operations can be seen below.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-8.png?raw=true)

Great! Now you have learnt deformable convolutions. Let’s now move onto the second deformable module, deformable RoI pooling.

The authors empirically show that deformable convolution is able to “expand” the receptive field for bigger object. They measure “effective dilation” which is the mean distances between each offsets (i.e. the blue squares in the Fig. 2). They found that deformable filters that are centered on larger objects has larger “receptive field”. 

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-14.png?raw=true)

### 3. Deformable RoI Pooling

Region of Interest (RoI) pooling is an important concept that is used in all region proposal based  <b> object detection </b> methods. It’s purpose is to <b> convert </b> variable-sized regions of a feature map into <b> fixed-sized </b> representations. This allows the network to downsample each RoI into a same sized output while displaying only its most important features.

In regular RoI pooling, we first define an input feature map x and an RoI of size <b> $w × h$ </b> with a top left-hand corner $p_0$. Essentially, RoI pooling divides the RoI into <b> $k × k$ </b> bins where $k$ is a hyperparameter. The number of pixels in each bin is defined with $n_{ij}$. The output is then a <b> $k × k$ </b> feature map $y$.

The output for each bin can be seen below. Note that this is <b> average RoI pooling </b> and other pooling types may also be used.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-9.png?raw=true)

In this equation, <b> $p$ </b> represents the positions in each bin and for this example, vector <b> $p$ </b>  holds <b> $ n_{ij} = 6 positions $ </b>. Note that in this example, the RoI perfectly splits into 3 × 3 regions, however, this is not the case for all RoI sizes, take 10 × 4 for example. This can be solved with methods such as RoI align, which are sadly outside the scope of this article.

Now, similar to what we did with deformable convolutions, we can add offsets $Δp_{ij}$ to the spatial binning positions before they are pooled.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-10.png?raw=true)

However, since RoIs can have different sizes, $Δp_{ij}$ may vary in scale.

Due to this, the network could learn <b> different offsets </b> for identical images of different scale. This is not what we want.

In order to resolve this, an <b> FC-layer </b> is implemented.

The network first performs regular RoI pooling, producing pooled feature maps. Then, these feature maps are flattened and fed into an FC-layer where <b> normalised offsets $Δ\hat p_{ij}$ </b> are then generated. These offsets are then transformed and scaled to $Δp_{ij}$. This transformation is achieved by applying a dot product over the RoI region size.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-11.png?raw=true)

In this equation, $γ$ is a predefined scalar that modulates the magnitude of the offsets and is commonly set to $γ = 0.1$.

This offset normalisation is crucial as it allows the offset learning and RoI size to be <b> invariant </b>. This is important in images where identical objects of different sizes are present.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-12.png?raw=true)

Just like in deformable convolution, the offsets are learnt via backpropagation on the bilinear operations. This is shown below.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/dcn-13.png?raw=true)

Great! This is everything you need to know about deformable RoI pooling, I hope you found it resourceful.

### 4. Summary

In this article, we have covered deformable convolutional networks and their importance in generalising and learning various transformations.

To summarise, deformable convolutions operate by adding additional offset parameters to the network. Instead of a filter being fixed in shape, fractional offsets that are learnt by the network are added. This deforms the filter and allows the same filter to reach different positions of the input feature map. This adjustment is beneficial when dealing with different geometric structures. In addition to this, deformable RoI pooling also experiences the same advantages allowing for more local and accurate object detection.

[Paper](https://arxiv.org/pdf/1703.06211.pdf)
[Code](https://github.com/msracver/Deformable-ConvNets)





