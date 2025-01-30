---
title: Reinforcement Learning Algorithms - [Part 1]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-01-26 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Reinforcement Learning Algorithms - [Part 1]

## An introduction to RL

RL is an area of machine learning that deals with sequential decision-making, aimed at
reaching a desired goal. An RL problem is constituted by a decision-maker called
an <b> Agent </b> and the physical or virtual world in which the agent interacts, is known as
the <b> Environment </b>. The agent interacts with the environment in the form of <b> Action </b> which
results in an effect. As a result, the environment will feedback to the agent a new <b> State </b>
and <b> Reward </b>. These two signals are the consequences of the action taken by the agent. In
particular, the reward is a value indicating how good or bad the action was, and the state is
the current representation of the agent and the environment.
 
![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-1-1.png?raw=true)

In this diagram the agent is represented by PacMan that based on the current state of the
environment, choose which action to take. Its behavior will influence the environment, like
its position and that of the enemies, that will be returned by the environment in the form of
a new state and the reward. This cycle is repeated until the game ends.

=> <b> The ultimate goal of the agent is to maximize the total reward accumulated during
its lifetime. </b>

To maximize the cumulative reward, the agent has to learn the best behavior in every
situation. To do so, the agent has to optimize for a long-term horizon while taking care of
every single action. In environments with many discrete or continuous states and actions,
learning is difficult because the agent should be accountable for each situation. To make the
problem harder, RL can have very sparse and delayed rewards, making the learning
process more arduous.

An important characteristic of RL is that it can deal with environments that are dynamic,
uncertain, and non-deterministic. These qualities are essential for the adoption of RL in the
real world. 

## Comparing RL and supervised learning

RL and supervised learning are similar, yet different, paradigms to learn from data. Many
problems can be tackled with both supervised learning and RL; however, in most cases,
they are suited to solve different tasks.

Supervised learning learns to generalize from a fixed dataset with a limited amount of data
consisting of examples. Each example is composed of the input and the desired output
(or label) that provides immediate learning feedback.

In comparison, RL is more focused on sequential actions that you can take in a particular
situation. In this case, the only supervision provided is the reward signal. There's no
correct action to take in a circumstance, as in the supervised settings.

RL can be viewed as a more general and complete framework for learning. The major
characteristics that are unique to RL are as follows:

- The reward could be dense, sparse, or very delayed. In many cases, the reward is
obtained only at the end of the task (for example, in the game of chess).

- The problem is sequential and time-dependent; actions will affect the next
actions, which, in turn, influence the possible rewards and states.

- An agent has to take actions with a higher potential to achieve a goal
(exploitation), but it should also try different actions to ensure that other parts of
the environment are explored (exploration). This problem is called the
exploration-exploitation dilemma (or exploration-exploitation trade-off) and it
manages the difficult task of balancing between the exploration and exploitation
of the environment. This is also very important because, unlike supervised
learning, RL can influence the environment since it is free to collect new data as
long as it deems it useful.

- The environment is stochastic and nondeterministic, and the agent has to take
this into consideration when learning and predicting the next action. In fact, we'll
see that many of the RL components can be designed to either output a single
deterministic value or a range of values along with their probability.

The third type of learning is unsupervised learning, and this is used to identify patterns in
data without giving any supervised information. Data compression, clustering, and
generative models are examples of unsupervised learning. It can also be adopted in RL
settings in order to explore and learn about the environment. The combination of
unsupervised learning and RL is called <b> unsupervised RL </b>. In this case, no reward is given
and the agent could generate an intrinsic motivation to favor new situations where they can
explore the environment.

## History of RL

The first mathematical foundation of RL was built during the 1960s and 1970s in the field of
optimal control. This solved the problem of minimizing a behavior's measure of a dynamic
system over time. The method involved solving a set of equations with the known
dynamics of the system. During this time, the key concept of a <b> Markov decision process
(MDP) </b> was introduced. <i> This provides a general framework for modeling decision-making
in stochastic situations. </i> During these years, a solution method for optimal control called
?<b> dynamic programming (DP) </b> was introduced. DP is a method that breaks down a complex
problem into a collection of simpler subproblems for solving an MDP.

Note that DP only provides an easier way to solve optimal control for systems with known
dynamics; there is no learning involved. It also suffers from the problem of the <b> curse of
dimensionality </b> because the computational requirements grow exponentially with the
number of states.

Even if these methods don't involve learning, as noted by Richard S. Sutton and Andrew G.
Barto, we must consider the solution methods of optimal control, such as DP, to also be RL
methods.

In the 1980s, the concept of learning by temporally successive predictions—the so-called
<b> temporal difference learning (TD learning) </b> method—was finally introduced. TD learning
introduced a new family of powerful algorithms.

The first problems solved with TD learning are small enough to be represented in tables or
arrays. These methods are called <b> tabular methods </b>, which are often found as an optimal
solution but are not scalable. In fact, many RL tasks involve huge state spaces, making
tabular methods impossible to adopt. In these problems, function
approximations are used to find a good approximate solution with less computational
resources.

The adoption of function approximations and, in particular, of artificial neural networks
(and deep neural networks) in RL is not trivial; however, as shown on many occasions, they
are able to achieve amazing results. The use of deep learning in RL is called <b> deep
reinforcement learning (deep RL) </b> and it has achieved great popularity ever since a deep
RL algorithm named <b> deep q network (DQN) </b> displayed a superhuman ability to play Atari
games from raw images in 2015. Another striking achievement of deep RL was with
AlphaGo in 2017, which became the first program to beat Lee Sedol, a human professional
Go player, and 18-time world champion. These breakthroughs not only showed that
machines can perform better than humans in high-dimensional spaces (using the same
perception as humans with respect to images), but also that they can behave in interesting
ways. An example of this is the creative shortcut found by a deep RL system while playing
Breakout, an Atari arcade game in which the player has to destroy all the bricks, as shown
in the following image. The agent found that just by creating a tunnel on the left-hand side
of the bricks and by putting the ball in that direction, it could destroy much more bricks
and thus increase its overall score with just one move. 

Nowadays, when dealing with high-dimensional state or action spaces, the use of deep
neural networks as function approximations becomes almost a default choice. Deep RL has
been applied to more challenging problems, such as data center energy optimization, selfdriving cars, multi-period portfolio optimization, and robotics, just to name a few. 

## Deep RL

Now you could ask yourself—why can deep learning combined with RL perform so well?
Well, the main answer is that deep learning can tackle problems with a high-dimensional
state space. Before the advent of deep RL, state spaces had to break down into simpler
representations, called <b> features </b>. These were difficult to design and, in some cases, only an
expert could do it. Now, using deep neural networks such as a <b> convolutional neural
network (CNN) </b> or a <b> recurrent neural network (RNN) </b>, RL can learn different levels of
abstraction directly from raw pixels or sequential data (such as natural language). This
configuration is shown in the following diagram:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-1-2.png?raw=true)

Furthermore, deep RL problems can now be solved completely in an end-to-end fashion.
Before the deep learning era, an RL algorithm involved two distinct pipelines: one to deal
with the perception of the system and one to be responsible for the decision-making. Now,
with deep RL algorithms, these processes are joined and are trained end-to-end, from the
raw pixels straight to the action. For example, as shown in the preceding diagram, it's
possible to train Pacman end-to-end using a CNN to process the visual component and a
fully connected neural network (FNN) to translate the output of the CNN into an action.

Nowadays, deep RL is a very hot topic. The principal reason for this is that deep RL is
thought to be the type of technology that will enable us to build highly intelligent
machines. As proof, two of the more renowned AI companies that are working to solve
intelligence problems, namely DeepMind and OpenAI, are heavily researching in RL.

Besides the huge steps achieved with deep RL, there is a long way to go. There are many
challenges that still need to be addressed, some of which are listed as follows:

- Deep RL is far too slow to learn compared to humans.

- Transfer learning in RL is still an open problem.

- The reward function is difficult to design and define.

- RL agents struggle to learn in highly complex and dynamic environments such as
the physical world.

Nonetheless, the research in this field is growing at a fast rate and companies are starting to
adopt RL in their products.

## Elements of RL

As we know, an agent interacts with their environment by the means of actions. This will
cause the environment to change and to feedback to the agent a reward that is proportional
to the quality of the actions and the new state of the agent. Through trial and error, the
agent incrementally learns the best action to take in every situation so that, in the long run,
it will achieve a bigger cumulative reward. In the RL framework, the choice of the action in
a particular state is done by a <b> policy </b>, and the cumulative reward that is achievable from
that state is called the <b> value function </b>. In brief, if an agent wants to behave optimally, then
in every situation, the policy has to select the action that will bring it to the next state with
the highest value. Now, let's take a deeper look at these fundamental concepts.

### Policy

<b> The policy defines how the agent selects an action given a state. </b> The policy chooses the
action that maximizes the cumulative reward from that state, not with the bigger
immediate reward. It takes care of looking for the long-term goal of the agent. For example,
if a car has another 30 km to go before reaching its destination, but only has another 10 km
of autonomy left and the next gas stations are 1 km and 60 km away, then the policy will
choose to get fuel at the first gas station (1 km away) in order to not run out of gas. This
decision is not optimal in the immediate future as it will take some time to refuel, but it will
be sure to ultimately accomplish the goal. 

The following diagram shows a simple example where an actor moving in a 4 x 4 grid has
to go toward the star while avoiding the spirals. The actions recommended by a policy are
indicated by an arrow pointing in the direction of the move. The diagram on the left shows
a random initial policy, while the diagram on the right shows the final optimal policy. In a
situation with two equally optimal actions, the agent can arbitrarily chooses which action to
take:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-1-3.png?raw=true)

An important distinction is between stochastic policies and deterministic policies. In the
deterministic case, the policy provides a single deterministic action to take. On the other
hand, in the stochastic case, the policy provides a probability for each action. The concept of
the probability of an action is useful because it takes into consideration the dynamicity of
the environment and helps its exploration.

One way to classify RL algorithms is based on how policies are improved during learning.
The simpler case is when the policy that acts on the environment is similar to the one that
improves while learning. Another way to say this is that the policy learns from the same
data that it generates. These algorithms are called <b> on-policy </b>. <b> Off-policy </b> algorithms, in
comparison, involve two policies—one that acts on the environment and another that
learns but is not actually used. The former is called the <b> behavior policy </b>, while the latter is
called the <b> target policy </b>. The goal of the behavior policy is to interact with and collect
information about the environment in order to improve the <b> passive target policy </b>. 

To better understand these two concepts, we can think of someone who has to learn a new
skill. If the person behaves as on-policy algorithms do, then every time they try a sequence
of actions, they'll change their belief and behavior in accordance with the reward
accumulated. In comparison, if the person behaves as an off-policy algorithm, they (the
target policy) can also learn by looking at an old video of themselves (the behavior policy)
doing the same skill—that is, they can use old experiences to help them to improve.

The <b> policy-gradient method </b> is a family of RL algorithms that learns a parametrized policy
(as a deep neural network) directly from the gradient of the performance with respect to the
policy. These algorithms have many advantages, including the ability to deal with
continuous actions and explore the environment with different levels of granularity.

### The value function

<b> The value function represents the long-term quality of a state. </b> This is the cumulative
reward that is expected in the future if the agent starts from a given state. If the reward
measures the immediate performance, the value function measures the performance in the
long run. This means that a high reward doesn't imply a high-value function and a low
reward doesn't imply a low-value function. 

Moreover, the value function can be a function of the state or of the state-action pair. The
former case is called a <b> state-value function </b>, while the latter is called an <b> action-value
function </b>:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-1-5.png?raw=true)

Here, the diagram shows the final state values (on the left side) and the corresponding
optimal policy (on the right side).

Using the same gridworld example used to illustrate the concept of policy, we can show the
state-value function. First of all, we can assume a reward of 0 in each situation except for
when the agent reaches the star, gaining a reward of +1. Moreover, let's assume that a
strong wind moves the agent in another direction with a probability of 0.33. In this case, the
state values will be similar to those shown in the left-hand side of the preceding diagram.
An optimal policy will choose the actions that will bring it to the next state with the highest
state value, as shown in the right-hand side of the preceding diagram.

Action-value methods (or value-function methods) are the other big family of RL
algorithms. These methods learn an action-value function and use it to choose the actions to
take. It's worth noting that some policy-gradient methods, in order
to combine the advantages of both methods, can also use a value function to learn the
appropriate policy. These methods are called actor-critic methods. The following diagram
shows the three main families of RL algorithms:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-1-6.png?raw=true)

### Reward

<b> At each timestep, that is, after each move of the agent, the environment sends back a
number that indicates how good that action was to the agent. This is called a reward. </b> As we
have already mentioned, the end goal of the agent is to maximize the cumulative reward
obtained during their interaction with the environment.

In literature, the reward is assumed to be a part of the environment, but that's not strictly
true in reality. The reward can come from the agent too, but never from the decisionmaking part of it. For this reason and to simplify the formulation, the reward is always sent
from the environment.

The reward is the only supervision signal injected into the RL cycle and it is essential to
design the reward in the correct way in order to obtain an agent with good behavior. If the
reward has some flaws, the agent may find them and follow incorrect behavior. For
example, Coast Runners is a boat-racing game with the goal being to finish ahead of other
players. During the route, the boats are rewarded for hitting targets. Some folks at OpenAI
trained an agent with RL to play it. They found that, instead of running to the finish line as
fast as possible, the trained boat was driving in a circle to capture re-populating targets
while crashing and catching fire. In this way, the boat found a way to maximize the total
reward without acting as expected. This behavior was due to an incorrect balance between
short-term and long-term rewards.

The reward can appear with different frequencies depending on the environment. A
frequent reward is called a <b> dense reward </b>; however, if it is seen only a few times during a
game, or only at its end, it is called a <b> sparse reward </b>. In the latter case, it could be very
difficult for an agent to catch the reward and find the optimal actions.

<b> Imitation learning and inverse RL </b> are two powerful techniques that deal with the absence
of a reward in the environment. Imitation learning uses an expert demonstration to map
states to actions. On the other hand, inverse RL deduces the reward function from an expert
optimal behavior. 

### Model

The model is an optional component of the agent, meaning that it is not required in order to
find a policy for the environment. The model details how the environment behaves,
predicting the next state and the reward, given a state and an action. If the model is known,
planning algorithms can be used to interact with the model and recommend future actions.
For example, in environments with discrete actions, potential trajectories can be simulated
using look ahead searches (for instance, using the Monte Carlo tree search).

The model of the environment could either be given in advance or learned through
interactions with it. If the environment is complex, it's a good idea to approximate it using
deep neural networks. RL algorithms that use an already known model of the environment,
or learn one, are called <b> model-based methods </b>. 

## Applications of RL

RL has been applied to a wide variety of fields, including robotics, finance, healthcare, and
intelligent transportation systems. In general, they can be grouped into three major
areas—automatic machines (such as autonomous vehicles, smart grids, and robotics),
optimization processes (for example, planned maintenance, supply chains, and process
planning) and control (for example, fault detection and quality control). 

In the beginning, RL was only ever applied to simple problems, but deep RL opened the
road to different problems, making it possible to deal with more complex tasks. Nowadays,
deep RL has been showing some very promising results. Unfortunately, many of these
breakthroughs are limited to research applications or games, and, in many situations, it is
not easy to bridge the gap between purely research-oriented applications and industry
problems. Despite this, more companies are moving toward the adoption of RL in their
industries and products.
