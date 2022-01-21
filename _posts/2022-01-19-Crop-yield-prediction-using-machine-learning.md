# Crop yield prediction using machine learning: A systematic literature review


## Machine Learning

Machine learning (ML) technologies are employed in a variety of fields, from evaluating customer behavior in supermarkets to predicting customer phone usage.

Agriculture has been using machine learning for several years. Crop yield prediction is one of precision agriculture's most difficult problems, and numerous models have been suggested and confirmed so far. Because crop production is affected by a range of factors such as climate, weather, soil, fertilizer use, and seed variety, this challenge necessitates the use of many datasets. This suggests that predicting agricultural yields is not a simple operation; rather, it entails a series of complex stages. Crop yield prediction methods can now reasonably approximate the actual yield, but better yield prediction performance is still desired.

Machine learning is a practical strategy that can deliver superior yield prediction based on numerous parameters. It is a branch of Artificial Intelligence (AI) that focuses on learning. Machine learning (ML) may uncover knowledge from datasets by identifying patterns and connections. The models must be trained with datasets that depict the outcomes based on previous experience. The predictive model is developed using a variety of features, and the parameters of the models are set during the training phase using previous data. Part of the previous data that was not used for training is used for performance evaluation during the testing phase.

Depending on the study challenge and research objectives, an ML model might be descriptive or predictive. Predictive models are used to make forecasts in the future, whereas descriptive models are used to obtain knowledge from the collected data and describe what has transpired. When attempting to create a high-performance prediction model, ML studies provide a variety of problems. It's critical to choose the correct algorithms to tackle the problem at hand, and the algorithms and underlying platforms must be able to handle large amounts of data.

Crop yield prediction is an essential task for the decision-makers at national and regional levels (e.g., the EU level) for rapid decisionmaking. An accurate crop yield prediction model can help farmers to decide on what to grow and when to grow. There are different approaches to crop yield prediction.

## Research questions

1. Which machine learning algorithm have been used in the literature for crop yield prediction?

2. Which features have been used in literature for crop yield prediction using machine learning?

3. Which evaluation parameters and evaluation appoaches have been used in literature for crop yield prediction?

4. What are challenges in the field of crop yield prediction using machine learning?

## Which machine learning algorithm have been used in the literature for crop yield prediction?

- <b> Deep Neural Network (DNN) </b> : These DNN algorithms are very similar to the traditional Artificial Neural Networks (ANN) algorithms except the number of hidden layers. In DNN networks, there are many hidden layers that are mostly fully connected, as in the case of ANN algorithms. However, for other kinds of deep learning algorithms such as CNN, there are also different types of layers, such as the convolutional layer and the pooling layer.

- <b> Convolutional Neural Networks (CNN) </b> :  Compared to a fully connected network, CNN has fewer parameters to learn. There are three types of layers in a CNN model, namely convolutional layers, pooling layers, and fully-connected layers. Convolutional layers consist of filters and feature maps. Filters are the neurons of the layer, have weighted inputs, and create an output value. A feature map can be considered as the output of one filter. Pooling layers are applied to down-sample the feature map of the previous layers, generalize feature representations, and reduce the overfitting. Fully-connected layers are mostly used at the end of the network for predictions. The general pattern for CNN models is that one or more convolutional layers are followed by a pooling layer, and this structure is repeated several times, and finally, fully connected layers are applied. 

- <b> Long-Short Term Memory (LSTM) </b> : LSTM networks were designed specifically for sequence prediction problems. There are several LSTM architectures, namely vanilla LSTM, stacked LSTM, CNN-LSTM, Encoder-Decoder LSTM, Bidirectional LSTM, and Generative LSTM. There are several limitations of Multi-Layer Perceptron (MLP) feedforward ANN algorithms, such as being stateless, unaware of temporal structure, messy scaling, fixed sized inputs, and fixed-sized outputs. Compared to the MLP network, LSTM can be considered as the addition of loops to the network. Also, LSTM is a special type of Recurrent Neural Network (RNN) algorithm. Since LSTM has an internal state, is aware of the temporal structure in the inputs, can model parallel input series, can process variable-length input to generate variable-length output, they are very different than the MLP networks. The memory cell is the computational unit of the LSTM. These cells consist of weights (i.e., input weights, output weights, and internal state) and gates (i.e., forget gate, input gate, and output gate). 

- <b> 3D CNN </b> : This network is a special type of CNN model in which the kernels move through height, length, and depth. As such, it produces 3D activation maps. This type of model was developed to improve the identification of moving, as in the case of security cameras and medical scans. 3D convolutions are performed in the convolutional layers of CNN

- <b> Faster R-CNN </b> : The Region-Based Convolutional Neural Network (RCNN) is a family of CNN models that were designed specifically for object detection. There are four variations of RCNN, namely R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN. In Faster R-CNN, a Region Proposal Network is added to interpret features extracted from CNN. 

- <b> Autoencoder </b> :  Autoencoders are unsupervised learning approaches that consist of the following four main parts: encoder, bottleneck, decoder, and reconstruction loss. The architecture of autoencoders can be designed based on simple feedforward neural networks, CNN, or LSTM networks.

- <b> Hybrid networks </b> : It is possible to combine the power of different deep learning algorithms. As such, researchers combine different algorithms in a different way. Chu and Yu (2020) combined Back-Propagation Neural Networks (BPNNs) and Independently Recurrent Neural Network (IndRNN) and applied this model for crop yield prediction. Sun et al. (2019) combined Convolutional Neural Networks and Long-Short Term Memory Networks (CNN-LSTM) for soybean yield prediction. Khaki et al. (2020) combined Convolutional Neural Networks and Recurrent Neural Networks (CNN-RNN) for yield prediction. Wang et al. (2020) combined CNN and LSTM (CNN-LSTM) networks for the wheat yield prediction problem. 

- <b> Multi-Task Learning (MTL) </b> : In multi-task learning, we share representations between tasks to improve the performance of our models developed for these tasks. It has been applied in many different domains, such as drug discovery, speech recognition, and natural language processing. The aim is to improve the performance of all the tasks involved instead of improving the performance of a single task. Zhang and Yang (2017) reviewed several multi-task learning approaches for supervised learning tasks and also explained how to combine multi-task learning with other learning categories, such as semi-supervised learning and reinforcement learning. They divided supervised MTL approaches into the following categories: feature learning approach, low-rank approach, task clustering approach, task relation learning approach, and decomposition approach.

- <b> Deep Recurrent Q-Network (DQN) </b> :  In reinforcement learning, agents observe the environment and act based on some rules and the available data. Agents get rewards based on their actions (i.e., positive or negative reward) and try to maximize this reward. The environment and agents interact with each other continuously. DQN algorithm was developed in 2015 by the researchers of DeepMind acquired by Google in 2014. This DQN algorithm that combines the power of reinforcement learning and deep neural networks solved several Atari games in 2015. The classical Q-learning algorithm was enhanced with deep neural networks, and also, the experience replay technique was integrated (Mnih et al., 2015). Elavarasan and Vincent (2020) applied this algorithm for crop yield prediction.
 
## Which features have been used in literature for crop yield prediction using machine learning?

- <b> Soil information </b> : soil maps(soil type, pH value, cation exchange capacity) and area of production.

- <b> Solar information </b> : gamma radiometric, temperature, photoperiod, shortwave radiation, degree-days, and solar radiation.

- <b> Humidity </b> : rainfall, humidity, forecasted rainfall, and precipitation.

- <b> Nutrients </b> : nitrogen, magnesium, potassium, sulphur, zinc, boron, calcium,
manganese, and phosphorus.

- <b> Crop information </b> :  weight, growth during the growth-process, variety of plants, and crop density.

- <b> Field management </b> : irrigation and fertilization.

- <b> Other </b> : wind speed, pressure, and images (used for calculated features MODIS Enhanced Vegetation Index, Normalized Vegetation Index, Enhanced Vegetation Index).

## Which evaluation parameters and evaluation appoaches have been used in literature for crop yield prediction?

- RMSE - Root mean square error 
- R2 - R-squared 
- MAE - Mean absolute error 
- MSE - Mean square error 
- MAPE - Mean absolute percentage error 
- RSAE - Reduced simple average ensemble 
- LCCC - Lin’s concordance correlation coefficient 
- MFE - Multi factored evaluation 
- SAE - Simple average ensemble 
- rcv - Reference change values 
- MCC - Matthew’s correlation coefficient 

## What are challenges in the field of crop yield prediction using machine learning?

The publications were read to see if they stated any problems or improvements for future models. In several studies, insufficient availability of data (too few data) was mentioned as a problem. The studies stated that their systems worked for the limited data that they had at hand, and indicated data with more variety should be used for further testing. This means data with different climatic circumstances, different vegetation, and longer timeseries of yield data. Another suggested improvement is that more data sources should be integrated. Finally, the publication indicated that the use of machine learning in farm management systems should be explored. If the models work as requested, software applications must be created that allow the farmer to make decisions based on the models.

## Discussion

-  Linear Regression is the second most used algorithms. Linear Regression is used as a benchmarking algorithm in most cases to check whether the proposed algorithm is better than Linear Regression or not. Therefore, although it is shown in many articles, it does not mean that it is the best performing algorithm. In fact, Deep Learning (DL), which is a sub-branch of Machine Learning, has been used for the crop yield prediction problem recently and is believed to be very promising. In this study, we also identified several deep learning-based studies. There are several additional promising aspects of DL methods, such as automatic feature extraction and superior performance. We expect that more research will be conducted on the use of DL approaches in crop yield prediction in the near future due to the superior performance of DL algorithms in other problem domains. 

- Groups are created for features and algorithms to visualize the main features and algorithms. Due to this decision, detailed information is lost, but clarity has been maintained. The most used features are soil type, rainfall, and temperature. Apart from those features that are used in several studies, there are also features that were used in specific studies. Those features are gamma radiation, MODIS-EVI, forecast rainfall, humidity, photoperiod, pH-value, irrigation, leaf area, NDVI, EVI, and crop information. There are also studies that use different nutrients as features, which are magnesium, potassium, sulphur, zinc, nitrogen, boron, and calcium. The most used features are not always the same kind of data. Temperature, for example, is measured as average temperature, but more features like maximum temperature and minimum temperature are also applied.

- There are not many evaluation parameters reported in the selected papers. Almost every study used RMSE as the measurement of the quality of the model. Other evaluation parameters are MSE, R2, and MAE. Some parameters were used in specific studies, most of these parameters look like some of the previously mentioned parameters, with a small difference. These are MAPE, LCCC, MFE, SAE, rcv, RSAE, and MCC. Most of the models had outcomes with high accuracy values for their evaluation parameters, which means that the model made correct predictions. As the evaluation approach, the 10-fold cross-validation approach was preferred by researchers. 

- Challenges were reported based on the explicit statements in the articles. However, there might be additional challenges that were not stated in the identified papers. The challenges are mainly in the field of improvement of a working model. When more data is gathered to train and test, much more can be said about the precision of the model. Another challenge is the implementation of the models into the farm management systems. When applications are made that the farmer can use, then only can the models be useful to make decisions, also during the growing season. When specific parameters for that specific place are measured and added, predictions will have higher precision. 

