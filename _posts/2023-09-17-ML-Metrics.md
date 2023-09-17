---
title: Common Model Evaluation Metrics for Machine Learning
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-09-13 11:11:14 +0700
categories: [Computer Vision]
tags: [Paper]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

Evaluating your machine learning algorithm is an essential part of any project. Your model may give you satisfying results when evaluated using a metric say accuracy_score but may give poor results when evaluated against other metrics such as logarithmic_loss or any other such metric. Most of the times we use classification accuracy to measure the performance of our model, however it is not enough to truly judge our model. In this post, we will cover different types of evaluation metrics available.

# Types of Predictive Models

When we talk about predictive models, we are talking either about a regression model (continuous output) or a classification model (nominal or binary output). The evaluation metrics used in each of these models are different.

In classification problems, we use two types of algorithms (dependent on the kind of output it creates):

- <b> Class output: </b> Algorithms like SVM and KNN create a class output. For instance, in a binary classification problem, the outputs will be either 0 or 1. However, today we have algorithms that can convert these class outputs to probability. But these algorithms are not well accepted by the statistics community.

- <b> Probability output: </b> Algorithms like Logistic Regression, Random Forest, Gradient Boosting, Adaboost, etc., give probability outputs. Converting probability outputs to class output is just a matter of creating a threshold probability.

In regression problems, we do not have such inconsistencies in output. The output is always continuous in nature and requires no further treatment.

# 1. Classification Accuracy

Classification Accuracy is what we usually mean, when we use the term accuracy. It is the ratio of number of correct predictions to the total number of input samples.

![Classification Accuracy](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-1.png?raw=true)

It works well only if there are equal number of samples belonging to each class.

For example, consider that there are 98% samples of class A and 2% samples of class B in our training set. Then our model can easily get <b> 98% training accuracy </b> by simply predicting every training sample belonging to class A.

When the same model is tested on a test set with 60% samples of class A and 40% samples of class B, then the <b> test accuracy would drop down to 60% </b>. 

=> <i> Classification Accuracy is great, but gives us the false sense of achieving high accuracy </i>.

The real problem arises, when the cost of misclassification of the minor class samples are very high. If we deal with a rare but fatal disease, the cost of failing to diagnose the disease of a sick person is much higher than the cost of sending a healthy person to more tests.

# 2. Logarithmic Loss

Logarithmic Loss or Log Loss, works by penalising the false classifications. It works well for multi-class classification. When working with Log Loss, the classifier must assign probability to each class for all the samples. Suppose, there are N samples belonging to M classes, then the Log Loss is calculated as below :

![Logarithmic Loss](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-2.png?raw=true)

Where:

- $y_{ij}$, indicates wheter sample $i$ belong to class $j$ off not.

- $p_{ij}$, indicates the probability of sample $i$ belonging to class $j$.

Log Loss has no upper bound and it exists on the range $[0, ∞)$. Log Loss nearer to 0 indicates higher accuracy, whereas if the Log Loss is away from 0 then it indicates lower accuracy.

In general, minimising Log Loss gives greater accuracy for the classifier.

# 3. Confusion Matrix

Confusion Matrix as the name suggests gives us a matrix as output and describes the complete performance of the model.

Lets assume we have a binary classification problem. We have some samples belonging to two classes : YES or NO. Also, we have our own classifier which predicts a class for a given input sample. On testing our model on 165 samples ,we get the following result.

![Confusion matrix](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-3.png?raw=true)

There are 4 important terms :

- <b> True Positives: </b> The cases in which we predicted YES and the actual output was also YES.

- <b> True Negatives: </b> The cases in which we predicted NO and the actual output was NO.

- <b> False Positives: </b> The cases in which we predicted YES and the actual output was NO.

- <b> False Negatives: </b> The cases in which we predicted NO and the actual output was YES.

Accuracy for the matrix can be calculated by taking average of the values lying across the “main diagonal” i.e

![Confusion matrix](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-4.png?raw=true)

Confusion Matrix forms the basis for the other types of metrics.

# 4. Area Under the ROC Curve (AUC – ROC) 

## 4.1. Area Under Curve

Area Under Curve(AUC) is one of the most widely used metrics for evaluation. It is used for binary classification problem. AUC of a classifier is equal to the probability that the classifier will rank a randomly chosen positive example higher than a randomly chosen negative example. Before defining AUC, let us understand two basic terms:

- <b> True Positive Rate (Sensitivity): </b> True Positive Rate is defined as TP/ (FN+TP). True Positive Rate corresponds to the proportion of positive data points that are correctly considered as positive, with respect to all positive data points.

![AUC-1](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-5.png?raw=true)

- <b> True Negative Rate (Specificity): </b> True Negative Rate is defined as TN / (FP+TN). False Positive Rate corresponds to the proportion of negative data points that are correctly considered as negative, with respect to all negative data points.

![AUC-2](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-6.png?raw=true)

- <b> False Positive Rate: </b> False Positive Rate is defined as FP / (FP+TN). False Positive Rate corresponds to the proportion of negative data points that are mistakenly considered as positive, with respect to all negative data points.

![AUC-3](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-7.png?raw=true)

<i> False Positive Rate </i> and <i> True Positive Rate </i> both have values in the range <b> [0, 1] </b>. FPR and TPR both are computed at varying threshold values such as (0.00, 0.02, 0.04, …., 1.00) and a graph is drawn. AUC is the area under the curve of plot <i> False Positive Rate </i> vs <i> True Positive Rate </i> at different points in <b> [0, 1] </b>.

![AUC-4](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-8.png?raw=true)

As evident, AUC has a range of [0, 1]. The greater the value, the better is the performance of our model.

## 4.2. ROC Curve

This is again one of the popular evaluation metrics used in the industry. The biggest advantage of using the ROC curve is that it is independent of the change in the proportion of responders. This statement will get clearer in the following sections.

Let’s first try to understand what the ROC (Receiver operating characteristic) curve is. If we look at the confusion matrix below, we observe that for a probabilistic model, we get different values for each metric.

![ROC-1](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-14.png?raw=true)

Hence, for each sensitivity, we get a different specificity. The two vary as follows:

![ROC-2](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-15.png?raw=true)

The ROC curve is the plot between sensitivity and (1- specificity). (1- specificity) is also known as the false positive rate, and sensitivity is also known as the True Positive rate. Following is the ROC curve for the case in hand.

![ROC-3](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-16.png?raw=true)

Let’s take an example of threshold = 0.5 (refer to confusion matrix). Here is the confusion matrix:

![ROC-4](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-17.png?raw=true)

As you can see, the sensitivity at this threshold is 99.6%, and the (1-specificity) is ~60%. This coordinate becomes on point in our ROC curve. To bring this curve down to a single number, we find the area under this curve (AUC).

Note that the area of the entire square is 1*1 = 1. Hence AUC itself is the ratio under the curve and the total area. For the case in hand, we get AUC ROC as 96.4%. Following are a few thumb rules:

- .90-1 = excellent (A)
- .80-.90 = good (B)
- .70-.80 = fair (C)
- .60-.70 = poor (D)
- .50-.60 = fail (F)

We see that we fall under the excellent band for the current model. But this might simply be over-fitting. In such cases, it becomes very important to do in-time and out-of-time validations.

<b> Points to Remember </b>

1. For a model which gives class as output will be represented as a single point in the ROC plot.

2. Such models cannot be compared with each other as the judgment needs to be taken on a single metric and not using multiple metrics. For instance, a model with parameters (0.2,0.8) and a model with parameters (0.8,0.2) can be coming out of the same model; hence these metrics should not be directly compared.

3. In the case of the probabilistic model, we were fortunate enough to get a single number which was AUC-ROC. But still, we need to look at the entire curve to make conclusive decisions. It is also possible that one model performs better in some regions and other performs better in others.

<b> Advantages of Using ROC </b>
Why should you use ROC and not metrics like the lift curve?

Lift is dependent on the total response rate of the population. Hence, if the response rate of the population changes, the same model will give a different lift chart. A solution to this concern can be a true lift chart (finding the ratio of lift and perfect model lift at each decile). But such a ratio rarely makes sense for the business.

The ROC curve, on the other hand, is almost independent of the response rate. This is because it has the two axes coming out from columnar calculations of the confusion matrix. The numerator and denominator of both the x and y axis will change on a similar scale in case of a response rate shift.

# 5. F1 Score

<i> F1 Score is used to measure a test's accuracy. </i>

F1 Score is the Harmonic Mean between precision and recall. The range for F1 Score is [0, 1]. It tells you how precise your classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances).

High precision but lower recall, gives you an extremely accurate, but it then misses a large number of instances that are difficult to classify. The greater the F1 Score, the better is the performance of our model. Mathematically, it can be expressed as :

![F1-1](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-9.png?raw=true)

F1 Score tries to find the balance between precision and recall.

- <b> Precision: </b> It is the number of correct positive results divided by the number of positive results predicted by the classifier.

![F1-2](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-10.png?raw=true)

- <b> Recall: </b> It is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).

![F1-3](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-11.png?raw=true)

# 6. Mean Absolute Error

Mean Absolute Error is the average of the difference between the Original Values and the Predicted Values. It gives us the measure of how far the predictions were from the actual output. However, they don’t gives us any idea of the direction of the error i.e. whether we are under predicting the data or over predicting the data. Mathematically, it is represented as:

![MAE](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-12.png?raw=true)

# 7. Mean Squared Error

## 7.1. Mean Squared Error
Mean Squared Error(MSE) is quite similar to Mean Absolute Error, the only difference being that MSE takes the average of the square of the difference between the original values and the predicted values. The advantage of MSE being that it is easier to compute the gradient, whereas Mean Absolute Error requires complicated linear programming tools to compute the gradient. As, we take square of the error, the effect of larger errors become more pronounced then smaller error, hence the model can now focus more on the larger errors.

![MSE](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-13.png?raw=true)

## 7.2. Root Mean Squared Error

RMSE is the most popular evaluation metric used in regression problems. It follows an assumption that errors are unbiased and follow a normal distribution. Here are the key points to consider on RMSE:

1. The power of ‘square root’ empowers this metric to show large number deviations.

2. The ‘squared’ nature of this metric helps to deliver more robust results, which prevent canceling the positive and negative error values. In other words, this metric aptly displays the plausible magnitude of the error term.

3. It avoids the use of absolute error values, which is highly undesirable in mathematical calculations.

4. When we have more samples, reconstructing the error distribution using RMSE is considered to be more reliable.

5. RMSE is highly affected by outlier values. Hence, make sure you’ve removed outliers from your data set prior to using this metric.

6. As compared to mean absolute error, RMSE gives higher weightage and punishes large errors.

RMSE metric is given by:

![RMSE](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-18.png?raw=true)

where N is the Total Number of Observations.

## 7.3. Root Mean Squared Logarithmic Error

In the case of Root mean squared logarithmic error, we take the log of the predictions and actual values. So basically, what changes are the variance that we are measuring? RMSLE is usually used when we don’t want to penalize huge differences in the predicted and the actual values when both predicted, and true values are huge numbers.

![RMSE](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/metric-19.png?raw=true)

1. If both predicted and actual values are small: RMSE and RMSLE are the same.

2. If either predicted or the actual value is big: RMSE > RMSLE

3. If both predicted and actual values are big: RMSE > RMSLE (RMSLE becomes almost negligible)




