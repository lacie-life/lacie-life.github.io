---
title: NLP - Getting started [Part 5]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-01-26 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# NLP - Getting started [Part 5]

## Neural Networks
 
Neural Networks (NNs) are a foundational concept
in machine learning, inspired by the structure and
function of the human brain.

At their core, NNs consist of interconnected nodes
organized into layers.

Input layers receive data, hidden layers process
information, and output layers produce results.

The strength of NNs lies in their ability to learn from
data, adjusting internal parameters (weights) during
training to optimize performance.

### Forward Propagation

In the forward propagation phase, data travels
through the network, and computations occur at
each layer, generating predictions. It’s similar to
information flowing from input to output.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-1.png?raw=true)

### Backward Propagation

The backward propagation phase involves the
critical aspect of learning.

Through techniques like gradient descent, the
network refines its internal parameters by
calculating the gradient of the loss function with
respect to the weights.

The chain rule plays a pivotal role here, allowing
the network to attribute the loss to specific
weights, enabling fine-tuning for better accuracy.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-2.png?raw=true)

### Gradient Descent

Gradient descent is the driving force behind weight
adjustments in NNs.

It’s an optimization algorithm that minimizes the loss
function by iteratively moving toward the steepest
down direction in the multidimensional weight space.

This iterative adjustment of weights enhances the
network’s predictive accuracy.

### Chain Rule

The chain rule in calculus is the essential for backpropagation.

It enables the computation of partial derivatives, attributing the
network’s overall error to individual weights.

This decomposition is vital for making nuanced adjustments during
training.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-3.png?raw=true)

## Importance of Sequences in NLP Tasks

In Natural Language Processing (NLP),
understanding and processing sequences is vital.

Unlike traditional machine learning tasks where
data points are independent, language inherently
involves sequential information.

In NLP, the order of words in a sentence carries
meaning, and context from previous words
influences the interpretation of subsequent ones.

## Recurrent Neural Networks (RNN)

RNNs are a specialized form of NN designed to
handle sequential data.

They introduce the concept of memory, enabling the
network to retain information about previous inputs.

This memory is crucial for tasks where context
matters, such as language understanding and
generation.

In some cases when it is required to predict the next
word of a sentence, the previous words are
required and hence there is a need to remember
the previous words.

Thus RNN came into existence, which solved this
issue with the help of a Hidden Layer.

The main and most important feature of RNN is its
Hidden state, which remembers some information
about a sequence.

The state is also referred to as Memory State since
it remembers the previous input to the network. RNN
uses the same weights for each element of the
sequence.

### How RNNs Work

Sequential Processing: Unlike traditional neural
networks, RNNs are designed to process sequences of
data. They do this by taking inputs one at a time in a
sequential manner.

Recurrent Connections: The key feature of an RNN is its
recurrent connections. These connections allow the
network to retain a form of ‘memory’.

At each step in a sequence, the RNN processes the
current input along with a ‘hidden state’ from the
previous step. This hidden state contains information
learned from prior inputs.

Hidden State: The hidden state is updated at each
time step, based on both the new input and the
previous hidden state. This mechanism allows the
RNN to carry information across different steps in
the sequence.

Shared Weights: In an RNN, the weights
(parameters) are shared across all time steps. This
means the same weights are used to process each
input in the sequence, making the model more
efficient and reducing the number of parameters.

### RNN Architecture

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-4.png?raw=true)

Vector h — is the output of the hidden state after the
activation function has been applied to the hidden
nodes. As you can see at time t, the architecture takes
into account what happened at t-1 by including the h
from the previous hidden state as well as the input x at
time t. This allows the network to account for
information from previous inputs that are sequentially
behind the current input. It’s important to note that the
zeroth h vector will always start as a vector of 0’s
because the algorithm has no information preceding
the first element in the sequence.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-5.png?raw=true)

The hidden state at t=2, takes as input the output from t-1 and x at t.
Matrices Wx, Wy, Wh — are the weights of the RNN architecture which are shared
throughout the entire network. The model weights of Wx at t=1 are the exact same as the
weights of Wx at t=2 and every other time step. Vector xᵢ — is the input to each hidden state
where i=1, 2,…, n for each element in the input sequence

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-6.png?raw=true)

The hidden nodes are a concatenation of the previous state’s output weighted by the
weight matrix Wh and the input x weighted by the weight matrix Wx. The tanh function is
the activation function that we mentioned earlier, symbolized by the green block. The
output of the hidden state is the activation function applied to the hidden nodes. To make a
prediction, we take the output from the current hidden state and weight it by the weight
matrix Wy with a soft max activation.

### Challenges and Advantages

Recurrent Neural Networks (RNNs) best in
processing sequential data, making them suitable
for tasks in language processing and time series
analysis.

Their ability to remember previous inputs is a
distinct advantage for short to medium sequences.

However, RNNs struggle with the vanishing gradient
problem, hindering their ability to handle long-term
dependencies. This limitation is significant for tasks
requiring extensive historical context.

Additionally, their sequential nature limits the
exploitation of modern parallel processing
technologies, leading to longer training times.
Despite these challenges, RNNs remain a
foundational architecture for sequence data
analysis.

### Long Short-Term Memory (LSTM)

LSTMs represent an advanced evolution in the world
of Recurrent Neural Networks, specifically
engineered to address and overcome the limitations
inherent in traditional RNNs, particularly when
dealing with long-term dependencies.

Advanced Memory Handling: The defining feature
of LSTM is its complex memory cell, known as the
LSTM unit. This unit can maintain information over
extended periods, thanks to its unique structure
comprising different gates.

Gating Mechanism: LSTMs incorporate three types
of gates, each playing a crucial role in the
network’s memory management.

### LSTM Architecture

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-7.jpeg?raw=true)

Input Gate: Determines which values from the input
should be used to modify the memory.

Forget Gate: Decides what portions of the existing
memory should be discarded.

Output Gate: Controls the output flow of the
memory content to the next layer in the network.

Cell State: At the core of LSTM is the cell state, a
kind of conveyor belt that runs straight down the
entire chain of the network. It allows information to
flow relatively unchanged and ensures that the
network retains and accesses important long-term
information efficiently.

### Challenges and Advantages

LSTMs are specifically designed to avoid the long-
term dependency problem, making them more
effective for tasks that require understanding
information over extended time periods.

However,
they are more complex and
computationally intensive compared to basic RNNs
and GRUs, which can be a challenge in terms of
training time and resource allocation.

LSTMs have proven effective in various domains that
require the processing of sequences with long-term
dependencies, such as complex sentence structures
in text, speech recognition, and time-series analysis.

LSTM networks offer a sophisticated approach to
handling sequential data, particularly excelling in
tasks where understanding long-term dependencies
is crucial.

Despite their complexity, they are a powerful tool in
the arsenal of neural network architectures,
particularly suited for deep learning tasks in NLP
and beyond.

## Gated Recurrent Unit (GRU)

GRUs are an innovative variation of Recurrent
Neural Networks, designed to improve upon and
simplify the architecture of LSTMs.

They offer a more streamlined approach to
handling sequential data, particularly effective in
scenarios where long-term dependencies are vital.

Simplified Architecture: The GRU is known for its
simplified structure compared to LSTM, making it
more efficient in terms of computational resources.
This efficiency stems from its reduced number of
gates.

### GRU Architecture

Gating Mechanism: GRUs utilize two gates:

Update Gate: This gate decides the extent to which
the information from the previous state should be
carried over to the current state. It is a blend of the
forget and input gates found in LSTMs.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-5-8.png?raw=true)

Reset Gate: It determines how much of the past
information to forget, effectively allowing the model
to decide how much of the past information is
relevant for the current prediction.

No Separate Cell State: Unlike LSTMs, GRUs do not
have a separate cell state. They combine the cell
state and hidden state into a single structure,
simplifying the information flow and making them
easier to model and train.

### Challenges and Advantages

GRUs are particularly known for their efficiency
and speed in training, making them a suitable
choice for models where computational resources
are a concern.

While they are generally faster and simpler than
LSTMs, they might not be as effective in capturing
very long-term dependencies due to their simplified
structure.

GRUs have been successfully applied in various
domains, such as language modeling, machine
translation, and speech-to-text applications, where
the balance between complexity and performance
is crucial.

GRUs present a more streamlined alternative to LSTMs,
offering similar capabilities in handling sequential data
with long-term dependencies but with less
computational complexity.

This makes them an attractive choice for many practical
applications in NLP and other areas where processing
sequential data is essential.

Their ability to balance performance with
computational efficiency makes them a valuable tool in
the field of deep learning, especially in scenarios
where resources are limited or when faster training
times are desired.

## Comparison of RNN, LSTM, and GRU

| Tables   |     Strengths       |  Limitations |
|----------|:-------------:|------:|
| RNN |  Ideal for processing sequences and maintaining information over short time spans. Simple architecture makes them computationally efficient | Struggle with long-term dependencies due to the vanishing gradient problem |
| LSTM |    Highly effective in learning long-term dependencies. The addition of input, forget, and output gates allows for better control over the memory cell, making them adept at handling issues like the vanishing gradient problem.   |   More complex than RNNs with additional parameters, leading to higher computational costs. |
| GRU | Similar to LSTMs in managing long-term dependencies but with a simpler structure. GRUs merge the input and forget gates into a single update gate, reducing complexity | - |

### Points to be Note

Choose RNNs for simplicity and when dealing with
shorter sequences where long-term dependencies
are not critical.

Opposite for LSTMs when the task involves complex
dependencies over extended time periods, and
model precision is paramount.

Select GRUs for a more balanced approach,
particularly when computational efficiency is as
important as model accuracy, or when working with
limited data.

In summary, the choice between RNN, LSTM, and
GRU depends on the specific requirements of the
task, including the nature of the input sequences,
computational resources, and the importance of
capturing long-term dependencies.
