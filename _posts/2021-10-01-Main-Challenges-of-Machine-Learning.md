# Main Challenges of Machine Learning

## "Bad Data"

### Insufficient Quantity of Training Data

Machine Learning takes a lot of data for most Machine Learning algorithms to work properly. Even for very simple problems you typically need thousands of examples, and for complex problems such as image or speech recognition you may need millions of examples(unless you can reuse parts of an existing model).

[Ref](https://dl.acm.org/doi/10.3115/1073012.1073017)

### Nonrepresentative Training Data

In order to machine learning algorithm works well, it is crucial that your training data be representative of the new cases you want to generalize to. This is correct whether you use instance-based learning or model-based learning.

By using a non representative training set, trained model is unlikely to make accurate predictions.

It is crucial to use a training set that is representative of the cases you want to train the data. If the sample is too small, you will have sampling noise (i.e., nonrepresentative data as a result of chance), but even very large samples can be non representative if the sampling method is flawed. This is called <b> sampling bias </b>.

### Poor-Quality Data

In reality, we don’t directly start training the model, analyzing data is the most important step. But the data we collected might not be ready for training, some samples are abnormal from others having outliers or missing values for instance. The truth is, most data scientists spend a significant part of their time doing just that.

If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually. Then, if some instances are missing a few features you must decide whether you want to ignore this attribute all of it, ignore these instances, fill in the missing values with the median or mean, or train one model with the feature and drop the feature.

### Irrelevant Features

If the training data contains a large number of irrelevant features and enough relevant features, the machine learning system will not give the results as expected. A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process, called feature engineering, involves :

- Feature selection: selecting the most useful features to train on among existing features.
- Feature extraction: combining existing features to produce a more useful one
- Creating new features by gathering new data.

### Organizing Experiment

Machine learning is an iterative process. You need to experiment with multiple combinations of data, learning algorithms and model parameters, and keep track of the impact these changes have on predictive performance. Over time this iterative experimentation can result in thousands of model training runs and model versions. This makes it hard to track the best performing models and their input configurations.

## "Bad Algorithms"

### Overfitting

Imagine you breakup with your boyfriend or girlfriend. Then you said all man or woman are jerk. Overgeneralizing is something that we humans do all too often, and unfortunately machines can fall into the same trap if we are not careful. In Machine Learning this is called <b> overfitting </b>. it means that the model performs well on the training data, but it does not generalize well.

There is a way to avoid overfitting. Constraining a model to make it simpler and reduce the risk of overfitting is called <b> regularization </b>. You want to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well.

The amount of regularization to apply during learning can be controlled by a <b> hyperparameter </b>. A hyperparameter is a parameter of a learning algorithm. it must be set prior to training and remains constant during training. If you set the regularization hyperparameter to a very large value, you will get an almost flat model. The learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good result.

At the point when the model is excessively unpredictable when compared to the noisiness of the training dataset, overfitting occurs. We can avoid it by:

- Gathering more training data.
- Selecting a model with fewer features, a higher degree polynomial model is not preferred compared to the linear model.
- Fix data errors, remove the outliers, and reduce the number of instances in the training set.

### Underfitting

Underfitting which is opposite to overfitting generally occurs when the model is too simple to understand the base structure of the data. It’s like trying to fit into undersized pants. It generally happens when we have less information to construct an exact model and when we attempt to build or develop a linear model with non-linear information.

Main options to reduce underfitting are:

- Feature Engineering — feeding better features to the learning algorithm.
- Removing noise from the data.
- Increasing parameters and selecting a powerful model.

### Testing and Validating

The only way to know how well a model will generalize to new cases is to actually try it out on new cases. One way to do that is to put your model in production and monitor how well it performs. This works well, but if your model is horribly bad, your users will complain—not the best idea. 

A better option is to split your data into two sets: the training set and the test set. As these names imply, you train your model using the training set, and you test it using the test set. The error rate on new cases is called the generalization error (or out-of-sample error), and by evaluating your model on the test set, you get an estimate of this error. This value tells you how well your model will perform on instances it has never
seen before.

If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the training data.

### Hyperparameter Tuning and Model Selection

Evaluating a model is simple enough: just use a test set. But suppose you are hesitating between two types of models (say, a linear model and a polynomial model): how can you decide between them? One option is to train both and compare how well they generalize using the test set.

Now suppose that the linear model generalizes better, but you want to apply some regularization to avoid overfitting. The question is, how do you choose the value of the regularization hyperparameter? One option is to train 100 different models using 100 different values for this hyperparameter. Suppose you find the best hyperparameter value that produces a model with the lowest generalization error—say, just 5% error. You launch this model into production, but unfortunately it does not perform as well as expected and produces 15% errors. What just happened?

The problem is that you measured the generalization error multiple times on the test set, and you adapted the model and hyperparameters to produce the best model for that particular set. This means that the model is unlikely to perform as well on new data.

### Data Mismatch

In some cases, it’s easy to get a large amount of data for training, but this data probably won’t be perfectly representative of the data that will be used in production.

### Models deployment to production

A lot of machine learning practitioners can perform all steps but can lack the skills for deployment, bringing their cool applications into production has become one of the biggest challenges due to lack of practice and dependencies issues, low understanding of underlying models with business, understanding of business problems, unstable models.

There are multiple factors to consider when deciding how to deploy a machine learning model:

- How frequently predictions should be generated.
- Whether predictions should be generated for a single instance at a time or a batch of instances.
- The number of applications that will access the model.
- The latency requirements of these applications.

Generally, many of the developers collect data from websites like Kaggle and start training the model. But in reality, we need to make a source for data collection, that varies dynamically. Offline learning or Batch learning may not be used for this type of variable data. The system is trained and then it is launched into production, runs without learning anymore.

It is always preferred to build a pipeline to collect, analyze, build/train, test and validate the dataset for any machine learning project and train the model in batches.

## No Free Lunch Theorem

A model is a simplified version of the observations. The simplifications are meant to discard the superfluous details that are unlikely to generalize to new instances. To decide what data to discard and what data to keep, you must make assumptions. For example, a linear model makes the assumption that the data is fundamentally linear and that the distance between the instances and the straight line is just noise, which can safely be ignored.

