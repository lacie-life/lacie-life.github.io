---
title: Paper note - [Week 6]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2024-03-30 11:11:14 +0700
categories: [Computer vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Paper note - [Week 6]

## [A customized residual neural network and bi-directional gated recurrent unit-based automatic speech recognition model](https://doi.org/10.1016/j.eswa.2022.119293)

### Motivation

- Voice technology is currently employed in many industries, allowing businesses and consumers to facilitate digitization and automation.

- Speech recognition is one of the most challenging computer science topics due to the difficulties of separating similar phonetically sentences and smearing problems.

### Contribution

- It proposes a stacked five layers of customized ResNets and seven
layers of Bi-GRUs, each including a layer normalization based on
a learnable element-wise affine parameters approach without the
requirement of external language models.

- The inclusion of the Gaussian error linear unit (GELU) layer and
the dense and dropout layers for the classification tasks showed
its worthiness in performance enhancement.

- It demonstrates that the volume of the training data significantly
affects the model’s output.

### Method

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-1.png?raw=true)

#### Mel spectrogram

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-2.png?raw=true)

#### Residual neural network

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-3.png?raw=true)


#### Bi-directional Gated Recurrent Units

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-4.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-5.png?raw=true)


#### Speech recognition model

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-6.png?raw=true)

### Experiments

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-7.png?raw=true)

### Conclusion

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/paper-note/week-6-8.png?raw=true)




