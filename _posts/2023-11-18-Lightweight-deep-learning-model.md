---
title: Lightweight deep learning model
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-11-18 11:11:14 +0700
categories: [Machine Learning]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

## Lightweight deep learning model

### 1. Lightweight deep learning model

Deep learning models show good performance in the field of video analysis. However, it has the problem of low efficiency as it requires a lot of memory space and calculation amount.

In reality, in mobile environments such as robots, autonomous vehicles, and smartphones that require image analysis using deep learning, there are many situations where hardware performance (memory size, processor performance) is limited.

For example, deep learning models often have large capacities because they store the values ​​of numerous parameters. In order to load a deep learning model on a mobile device, parameter values ​​must be loaded into memory, but it may be difficult to load it directly due to a lack of memory.

Additionally, deep learning requires numerous calculations using stored weights, and if the processor's performance is low, image processing may take a long time.

Therefore, lightweight technology is essential to use deep learning technology in real life.

### 2. Lightweight method

Lightweighting methods can be roughly divided into two types: methods of reducing the size of the learned model and methods of efficiently designing the network structure itself.

#### 2.1. Pruning

Pruning, which can be interpreted as pruning, is a method of reducing model parameters by removing connections between weights of low importance among the weights of the model.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-1.png?raw=true)

If each weight value has a different impact on the results, even if a weight with relatively less influence is deleted, the impact on the results will be small.

Therefore, if you remove weights that have little influence on the final result, you will be able to create a model with fewer parameters but showing similar performance.

<b> Pruning can be defined as a method of making weights as sparse as possible without significantly degrading network performance. </b>

Each layer of a deep learning network consists of weight and bias. Parameters refer to both weight and bias values.

In general, bias accounts for a relatively small proportion of parameters, so pruning is not worth it because it does not make a big difference. So generally no pruning.

#### Definition of Sparsity

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-2.png?raw=true)

Of the three 16x16 matrices above, the leftmost one is considered a sparse matrix because all elements except one are 0. The middle matrix also appears to be a sparse matrix because 0 accounts for the majority.

So, is the matrix on the right a sparse matrix? Although 0 clearly occupies more than half of the matrix, it does not appear to be a sparse matrix.

If <b> most </b> of the weights are 0, it can be considered sparse. So how exactly can we set the standard for <b> most </b> ?

Sparsity is a concept used to confirm <b> most things with specific numbers </b>. <b> Sparsity is a measure of how many weights in the entire network are exactly 0 (rather than a small number) </b>.

The simplest way to measure sparsity is $l$. This is using norm.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-3.png?raw=true)


In the above formula, each element is 1 or 0, $l$ norm is the number of values ​​in which the element's value is not 0.

Now compared to the total number of weights $l$ You can find sparsity by checking the value of norm.

#### Things to set in pruning

<b> 1. Pruning granularity </b>
When pruning, pruning each element is called element-wise pruning or fine-grained pruning.

Grouping elements into groups and pruning the entire group is called coarse-grained pruning, structured pruning, group pruning, or block pruning.
Groups can be of many shapes and sizes.

As shown above, it is necessary to determine the elements to be pruned.

<b> 2. Prunning criteria </b>
Pruning requires a standard to determine which elements of the weights to prune and how. These criteria are called pruning criteria.

One example of the most commonly used pruning criteria is to determine what threshold to use for the weight value (whether to change it to 0 or leave it as is).
Among the weights, weights with small absolute values ​​will have a small impact on the results, so the previous example is one of the pruning criteria that can actually be used.

Another factor to consider when setting pruning criteria is how much accuracy is acceptable compared to the dense-model (the original model before pruning).

<b> 3. Pruning schedule </b>
The most straightforward pruning method is to prune after learning. Pruning once after completing learning is called one-shot pruning.

Re-training a sparse network after going through a learning and pruning process is called iterative pruning.
It is known that performance increases further when retraining a pruned network.

At this time, information such as whether to do one-shot pruning or iterative pruning, how many times to repeat if iterative pruning, how to set prunning criteria for each iteration, what weight to prune, and when to stop are collectively called the pruning schedule. do.

The timing of stopping pruning can also be expressed as a schedule. For example, it can be viewed as a schedule to stop when a certain sparsity level is reached or to stop when the target computational amount is reached.

#### Sensitivity analysis

The most difficult part when pruning is setting the threshold value that determines whether to prune weights and setting the target sparsity level.

Sensitivity analysis is a way to analyze which weights or layers are most affected when pruning, and is useful when setting a threshold or sparsity level.

An example of sensitivity analysis is as follows.

-  First, set the pruning level (sparsity level) of a specific weight or group, and evaluate performance after pruning.
-  Perform step 1 for all weights or groups.

From the results obtained through this process, we can know how sensitive a weight or group is, that is, how much it affects the results.
The more structures you prune, the more likely it is to analyze the sensitivity of the model structure.

In summary, >b to understand the sensitivity of each weight to the results, you can prune and evaluate each weight and see how the accuracy changes to find out how important the weight is </b>.

As an example, looking at AlexNet's pruning sensitivity, it is shown in the table below.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-4.png?raw=true)

#### Pruning method

<b> 1. Magnitude Pruner </b>
Maginiture Pruning is the most basic method and is a method of thresholding weights based on a certain value.

If the weight value is below the standard value, set it to 0, and if it is greater than the standard value, leave it as is.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-5.png?raw=true)

<b> 2. Sensitivity Pruner </b>
It is not easy to find the threshold value for each layer.

The sensitivity pruning method utilizes the fact that the convolution layer and fully connected layer have Gaussian distribution.

For example, if you look at the weight distribution of the first convolution layer and fully-connected layer of the pre-trained AlexNet, you can see that it is similar to a Gaussian distribution.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-6.png?raw=true)

Therefore, the pruning criteria for thresholding in the Sensitivity Pruner are set as follows.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-5.png?raw=true)


$λ=s∗σ$ where $σ$ is std of layer $l$ as measured on the dense model

If the values ​​of 68% weights of a layer are less than the standard deviation, the standard $λ$ is $s∗$ It becomes 64 %.

then How can I finds ?

Professor Song Han's paper, <i> Learning both Weights and Connections for Efficient Neural Networks </i>, utilizes the sensitivity discussed in the previous chapter.
In the paper, iterative pruning was used, and the sentitivity parameter the $s$ value has been modified.

Therefore, the operation of the Sensitivity Pruner is as follows.

- Perform sensitivity analysis on the model.
- The sensitivity parameter derived from the analysis is multiplied by the standard deviation of the weight distribution and used as a threshold.

<b> 3. Level Pruner </b>
Level Pruner does not threshold based on a specific value, but rather prunes based on the sparsity level of a specific layer. In other words, rather than judging by looking at the value of each weight, pruning is done so that the sparsity of the layer is a specific value.

For example, assuming that the sparsity level of a layer is pruned to 0.5 (50% of the weights are 0), the top 50% of the weight values ​​in the layer are left as is and the bottom 50% are pruned to 0.

The operation of Level Pruner is as follows.

- Sort based on the absolute value of the layer you want to prune.
- In the sorted results, from the smallest to the sparsity level, convert the number of weights to 0 that are the target value.

<b> 4. Other prunning methods </b>
In addition to the pruning techniques described, there are various pruning methods such as Splicing Pruner, Automated Gradual Pruner (AGP), RNN Pruner, Structure Pruners, and Hybrid Pruning.

The contents of the pruning algorithm that are not summarized here are well organized in the document of <i> Neural Network Distiller, an open-source for network compression </i>.

#### Use of Sparse Matrix

Through pruning, a layer with high sparsity is created. However, because it still contains information about weights pruned to 0, it is necessary to remove information about 0 to reduce the capacity of the model.

In fact, when the weight layer is viewed as a single matrix, the pruned result will appear in the form of a sparse matrix.

Methods for efficiently storing such sparse matrices include Coordinate Format (COO) and Compressed Sparse Row (CSR).

COO is a method of storing non-zero values, rows, and columns in separate memory. The number of non-zero values ​​in a sparse matrix isIn the case of COO when there is a only 3 a memory spaces are required.

CSR means that it has value and row information like COO, but has column information in the form of a pointer (address value).

In the example below, the data and column indices are stored as COO.

Rows go through a process of checking how much data is in each row and then adding that much. For example, the first row contains two pieces of data . So the column pointer gets +2 .

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-7.png?raw=true)

The length of the rows of Matrix is When there are n , the number of memory spaces required by CSR is 2a _+( n+1 ) It’s a dog. CSRs are known to use less data than COOs, although this is not always the case.

In contrast to CSR, which is organized by rows, CSC is organized by columns. The storage method is the same as CSR.

Using this <b> method of storing the sparse matrix, the pruned weight layer can be stored efficiently and the amount of memory required for saving can be reduced. </b>

#### 2.1. Quantization

#### What is Quantization?

Model Quantization is a method of increasing computational efficiency by reducing the number of bits used by parameters.

In deep learning, FP32 (32-bit floating point) is mostly used when processing numbers. However, if numbers are expressed using lower bits without losing accuracy, computational efficiency will increase.

Quantization is a method of reducing the size of the model by expressing the weight value by lowering the bitband to FP16 (16-bit floating point) or INT8 (8-bit integer) rather than FP32.

#### Advantages of Quantization

If a model saved as FP32 is reduced to FP16, the memory usage is halved (32-bit → 16-bit), and if it is reduced to INT8, the memory usage is reduced $\frac{One}{4}$. This happens.

Additionally, because fewer bits are used during calculation, operation time can be shortened when inference is performed using a learned model, and energy efficiency increases.

Looking at [Professor Dally's educational materials](https://media.nips.cc/Conferences/2015/tutorialslides/Dally-NIPS-Tutorial-2015.pdf),
using INT8 can significantly save energy and storage space in the add and multiply operations as shown below.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/lightweight-8.png?raw=true)

Recently, most deep learning frameworks such as Tensorflow and MXNet support quantization such as FP16 and INT8, so you can use it easily and quickly.

#### Precautions for Quantization

Nowadays, most deep learning frameworks support quantization, so there is no need to implement quantization directly. However, in theory, it would be good to know what to pay attention to, so I have summarized the information below.









