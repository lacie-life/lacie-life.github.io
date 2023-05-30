---
title:  Machine Learning is Fun! - How to Intentionally Trick Neural Networks
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2022-01-28 11:11:11 +0700
categories: [Machine Learning]
tags: [tutorial]
render_with_liquid: false
---

# Machine Learning is Fun!: How to Intentionally Trick Neural Networks

Almost as long as programmers have been writing computer programs, computer hackers have been figuring out ways to exploit those programs. Malicious hackers take advantage of the tiniest bugs in programs to break into systems, steal data and generally wreak havoc.

But systems powered by deep learning algorithms should be safe from human interference, right? How is a hacker going to get past a neural network trained on terabytes of data?

It turns out that even the most advanced deep neural networks can be easily fooled. With a few tricks, you can force them into predicting whatever result you want:

![Fig.1](https://images.viblo.asia/5776a478-7ffc-451e-8521-4fb97059cb85.png)

So before you launch a new system powered by deep neural networks, let’s learn exactly how to break them and what you can do to protect yourself from attackers.

## Neural Nets as Security Guards

Let’s imagine that we run an auction website like Ebay. On our website, we want to prevent people from selling prohibited items — things like live animals.

Enforcing these kinds of rules are hard if you have millions of users. We could hire hundreds of people to review every auction listing by hand, but that would be expensive. Instead, we can use deep learning to automatically check auction photos for prohibited items and flag the ones that violate the rules.

This is a typical image classification problem. To build this, we’ll train a deep convolutional neural network to tell prohibited items apart from allowed items and then we’ll run all the photos on our site through it.

First, we need a data set of thousands of images from past auction listings. We need images of both allowed and prohibited items so that we can train the neural network to tell them apart:

![Fig.2](https://images.viblo.asia/4df21893-5ffc-4da7-ac84-a79dcfb1accc.png)

To train then neural network, we use the standard back-propagation algorithm. This is an algorithm were we pass in a training picture, pass in the expected result for that picture, and then walk back through each layer in the neural network adjusting their weights slightly to make them a little better at producing the correct output for that picture:

![Fig.3](https://images.viblo.asia/b17d0853-de7d-42e4-88f9-478c87845d69.png)

We repeat this thousands of times with thousands of photos until the model reliably produces the correct results with an acceptable accuracy.

The end result is a neural network that can reliably classify images:

![Fig.4](https://images.viblo.asia/b47ef731-417b-4ece-8099-622c99c3dbf7.png)

## But things are not as reliable as they seem…

Convolutional neural networks are powerful models that consider the entire image when classifying it. They can recognize complex shapes and patterns no matter where they appear in the image. In many image recognition tasks, they can equal or even beat human performance.

With a fancy model like that, changing a few pixels in the image to be darker or lighter shouldn’t have a big effect on the final prediction, right? Sure, it might change the final likelihood slightly, but it shouldn’t flip an image from “prohibited” to “allowed”.

![Fig.5](https://images.viblo.asia/810ed1e8-5d1f-462b-bb2e-bad15ab90549.png)

But in a famous paper in 2013 called Intriguing properties of neural networks, it was discovered that this isn’t always true. If you know exactly which pixels to change and exactly how much to change them, you can intentionally force the neural network to predict the wrong output for a given picture without changing the appearance of the picture very much.

That means we can intentionally craft a picture that is clearly a prohibited item but which completely fools our neural network:

![Fig.6](https://images.viblo.asia/7be58bfd-0520-4c73-9d6b-765c70d83718.png)

Why is this? A machine learning classifier works by finding a dividing line between the things it’s trying to tell apart. Here’s how that looks on a graph for a simple two-dimensional classifier that’s learned to separate green points (acceptable) from red points (prohibited):

![Fig.7](https://images.viblo.asia/f611a324-c44f-41b0-89da-1c86588cb548.png)

Right now, the classifier works with 100% accuracy. It’s found a line that perfectly separates all the green points from the red points.

But what if we want to trick it into mis-classifying one of the red points as a green point? What’s the minimum amount we could move a red point to push it into green territory?

If we add a small amount to the Y value of a red point right beside the boundary, we can just barely push it over into green territory:

![Fig.8](https://images.viblo.asia/f6f1f7cc-4d1c-4241-9059-8ad3524d6080.png)

So to trick a classifier, we just need to know which direction to nudge the point to get it over the line. And if we don’t want to be too obvious about being nefarious, ideally we’ll move the point as little as possible so it just looks like an honest mistake.

In image classification with deep neural networks, each “point” we are classifying is an entire image made up of thousands of pixels. That gives us thousands of possible values that we can tweak to push the point over the decision line. And if we make sure that we tweak the pixels in the image in a way that isn’t too obvious to a human, we can fool the classifier without making the image look manipulated.

In other words, we can take a real picture of one object and change the pixels very slightly so that the image completely tricks the neural network into thinking that the picture is something else — and we can control exactly what object it detects instead:

![Fig.9](https://images.viblo.asia/4da557ba-dbf3-4e93-b522-09fff38aeaec.png)

## How to Trick a Neural Network

We’ve already talked about the basic process of training a neural network to classify photos:

1. Feed in a training photo.
2. Check the neural network’s prediction and see how far off the is from the correct answer.
3. Tweak the weighs of each layer in the neural network using back-propagation to make the final prediction slightly closer to the correct answer.
4. Repeat steps 1–3 a few thousand times with a few thousand different training photos.

But what if instead of tweaking the weights of the layers of the neural network, we instead tweaked the input image itself until we get the answer we want?

So let’s take the already-trained neural network and “train” it again. But let’s use back-propagation to adjust the input image instead of the neural network layers:

![Fig.10](https://images.viblo.asia/e06a2953-04a8-4663-9207-733208355afe.png)

So here’s the new algorithm:

1. Feed in the photo that we want to hack.
2. Check the neural network’s prediction and see how far off the is from the answer we want to get for this photo.
3. Tweak our photo using back-propagation to make the final prediction slightly closer to the answer we want to get.
4. Repeat steps 1–3 a few thousand times with the same photo until the network gives us the answer we want.

At end of this, we’ll have an image that fools the neural network without changing anything inside the neural network itself.

The only problem is that by allowing any single pixel to be adjusted without any limitations, the changes to the image can be drastic enough that you’ll see them. They’ll show up as discolored spots or wavy areas:

![Fig.11](https://images.viblo.asia/3b2d2df3-9396-4203-8e2c-773f7de2f07d.png)

To prevent these obvious distortions, we can add a simple constraint to our algorithm. We’ll say that no single pixel in the hacked image can ever be changed by more than a tiny amount from the original image — let’s say something like 0.01%. That forces our algorithm to tweak the image in a way that still fools the neural network without it looking too different from the original image.

Here’s what the generated image looks like when we add that constraint:

![Fig.12](https://images.viblo.asia/9c026291-524f-4514-8625-ef69dca2b626.png)

Even though that image looks the same to us, it still fools the neural network!

## Let’s Code It

To code this, first we need a pre-trained neural network to fool. Instead of training one from scratch, let’s use one created by Google.

Keras, the popular deep learning framework, comes with several pre-trained neural networks. We’ll use its copy of Google’s Inception v3 deep neural network that was pre-trained to detect 1000 different kinds of objects.

Here’s the basic code in Keras to recognize what’s in a picture using this neural network. Just make sure you have Python 3 and Keras installed before you run it:

```
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3

# Load pre-trained image recognition model
model = inception_v3.InceptionV3()

# Load the image file and convert it to a numpy array
img = image.load_img("cat.png", target_size=(299, 299))
input_image = image.img_to_array(img)

# Scale the image so all pixel intensities are between [-1, 1] as the model expects
input_image /= 255.
input_image -= 0.5
input_image *= 2.

# Add a 4th dimension for batch size (as Keras expects)
input_image = np.expand_dims(input_image, axis=0)

# Run the image through the neural network
predictions = model.predict(input_image)

# Convert the predictions into text and print them
predicted_classes = inception_v3.decode_predictions(predictions, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))
```

When we run it, it properly detects our image as a Persian cat:

```
$ python3 predict.py
This is a Persian_cat with 85.7% confidence!
```

Now let’s trick it into thinking that this cat is a toaster by tweaking the image until it fools the neural network.

Keras doesn’t have a built-in way to train against the input image instead of training the neural network layers, so I had to get a little tricky and code the training step manually.

Here’s the code:

```
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image

# Load pre-trained image recognition model
model = inception_v3.InceptionV3()

# Grab a reference to the first and last layer of the neural net
model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

# Choose an ImageNet object to fake
# The list of classes is available here: https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
# Class #859 is "toaster"
object_type_to_fake = 859

# Load the image to hack
img = image.load_img("cat.png", target_size=(299, 299))
original_image = image.img_to_array(img)

# Scale the image so all pixel intensities are between [-1, 1] as the model expects
original_image /= 255.
original_image -= 0.5
original_image *= 2.

# Add a 4th dimension for batch size (as Keras expects)
original_image = np.expand_dims(original_image, axis=0)

# Pre-calculate the maximum change we will allow to the image
# We'll make sure our hacked image never goes past this so it doesn't look funny.
# A larger number produces an image faster but risks more distortion.
max_change_above = original_image + 0.01
max_change_below = original_image - 0.01

# Create a copy of the input image to hack on
hacked_image = np.copy(original_image)

# How much to update the hacked image in each iteration
learning_rate = 0.1

# Define the cost function.
# Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
cost_function = model_output_layer[0, object_type_to_fake]

# We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
# In this case, referring to "model_input_layer" will give us back image we are hacking.
gradient_function = K.gradients(cost_function, model_input_layer)[0]

# Create a Keras function that we can call to calculate the current cost and gradient
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

cost = 0.0

# In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
# until it gets to at least 80% confidence
while cost < 0.80:
    # Check how close the image is to our target class and grab the gradients we
    # can use to push it one more step in that direction.
    # Note: It's really important to pass in '0' for the Keras learning mode here!
    # Keras layers behave differently in prediction vs. train modes!
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

    # Move the hacked image one step further towards fooling the model
    hacked_image += gradients * learning_rate

    # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    hacked_image = np.clip(hacked_image, -1.0, 1.0)

    print("Model's predicted likelihood that the image is a toaster: {:.8}%".format(cost * 100))

# De-scale the image's pixels from [-1, 1] back to the [0, 255] range
img = hacked_image[0]
img /= 2.
img += 0.5
img *= 255.

# Save the hacked image!
im = Image.fromarray(img.astype(np.uint8))
im.save("hacked-image.png")
```

If we run this, it will eventually spit out an image that will fool the neural network:

```
$ python3 generated_hacked_image.py
Model's predicted likelihood that the image is a toaster: 0.00072%
[ .... a few thousand lines of training .... ]
Model's predicted likelihood that the image is a toaster: 99.4212%
```

Note: If you don’t have a GPU, this might take a few hours to run. If you do have a GPU properly configured with Keras and CUDA, it shouldn’t take more than a couple of minutes to run.

Now let’s test the hacked image that we just made by running it through the original model again:

```
$ python3 predict.py
This is a toaster with 98.09% confidence!
```

We did it! We tricked the neural network into thinking that a cat is a toaster!

## What can we do with a Hacked Image?

Created a hacked image like this is called “generating an adversarial example”. We’re intentionally crafting a piece of data so that a machine-learning model will misclassify it. It’s a neat trick, but why does this matter in the real world?

Research has show that these hacked images have some surprising properties:

1. Hacked images can still fool neural networks even when they are printed out on paper! So you can use these hacked images to fool physical cameras or scanners, not just systems where upload an image file directly.

2. Images that fool one neural network tend to fool other neural networks with entirely different designs if they were trained on similar data.

So we can potentially do a lot with these hacked images!

But there is still a big limitation with how we create these images — our attack requires direct access to the neural network itself. Because we are actually “training” against the neural network to fool it, we need a copy of it. In the real world, no company is going to let you download their trained neural network’s code, so that means we can’t attack them… Right?

Nope! Researchers have recently shown that you can train your own substitute neural network to mirror another neural network by probing it to see how it behaves. Then you can use your substitute neural network to generate hacked images that still often fool the original network! This is called a black-box attack.

The applications of black-box attacks are limitless. Here are some plausible examples:

- Trick self-driving cars into seeing a stop sign as a green light — this could cause car crashes!
- Trick content filtering systems into letting offensive/illegal content through.
- Trick ATM check scanners into thinking the handwriting on a check says the check is for a greater amount then it actually is (with plausible deniability if you get caught!)

And these attack methodology isn’t limited to just images. You can use the same kind of approach to fool classifiers that work on other types of data. For example, you could trick virus scanners into recognizing your virus as safe code!

## How can we protect ourselves against these attacks?

So now that we know it’s possible to trick neural networks (and all other machine learning models too), how do we defend against this?

The short answer is that no one is entirely sure yet. Preventing these kinds of attacks is still an on-going area of research. The best way to keep up with the latest developments is by reading the cleverhans blog maintained by Ian Goodfellow and Nicolas Papernot, two of the most influential researchers in this area.

But there are some things we do know so far:

- If you simply create lots of hacked images and include them in your training data set going forward, that seems to make your neural network more resistant to these attacks. This is called Adversarial Training and is probably the most reasonable defense to consider adopting right now.
- There is another somewhat effective approach called Defensive Distillation where you train a second model to mimic your original model. But this approach is new and rather complicated, so I wouldn’t invest in this yet unless you have specialized needs.
- Pretty much every other idea researchers have tried so far has failed to be helpful in preventing these attacks.

Since we don’t have any final answers yet, its worth thinking about the scenarios where you are using neural networks so that you can at least lessen the risk that this kind of attack would cause damage your business.

For example, if you have a single machine learning model as the only line of defense to grant access to a restricted resource and assume it can’t be fooled, that’s probably a bad idea. But if you use machine learning as a step in a process where there is still human verification, that’s probably fine.

In other words, treat machine learning models in your architecture like any other component that can potentially be bypassed. Think through the implications of what would happen if a user intentionally sets out to fool them and think of ways to mitigate those scenarios.


