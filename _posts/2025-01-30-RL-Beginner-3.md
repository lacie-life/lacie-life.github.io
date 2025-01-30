---
title: Reinforcement Learning Algorithms - [Part 3]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-01-30 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Reinforcement Learning Algorithms - [Part 3]

## The OpenAI Gym API and Gymnasium

### Install requierment

```
gymnasium[atari]==0.29.1 
gymnasium[classic-control]==0.29.1 
gymnasium[accept-rom-license]==0.29.1 
moviepy==1.0.3 
numpy<2 
opencv-python==4.10.0.84 
torch==2.5.0 
torchvision==0.20.0 
pytorch-ignite==0.5.1 
tensorboard==2.18.0 
mypy==1.8.0 
ptan==0.8.1 
stable-baselines3==2.3.2 
torchrl==0.6.0 
ray[tune]==2.37.0 
pytest
```

### Gym and Gymnasium

The Python library called Gym was developed by [OpenAI](www.openai.com). The first version was released in 2017 and since
then, lots of environments were developed or adopted to this
original API, which became a de facto standard for RL.
In 2021, the team that developed OpenAI Gym moved the
development to [Gymnasium]( github.com/Farama-Foundation/Gymnasium) – the fork of the original Gym library.
Gymnasium provides the same API and is supposed to be a “dropin replacement” for Gym (you can write import gymnasium as gym
and most likely your code will work).

The main goal of Gym is to provide a rich collection of
environments for RL experiments using a unified interface. So, it is
not surprising that the central class in the library is an
environment, which is called <b> Env </b>. Instances of this class expose
several methods and fields that provide the required information
about its capabilities. At a high level, every environment provides
these pieces of information and functionality:

- A set of actions that is allowed to be executed in the
environment. Gym supports both discrete and continuous
actions, as well as their combination.

- The shape and boundaries of the observations that the
environment provides the agent with.

- A method called <b> step </b> to execute an action, which returns the
current observation, the reward, and a flag indicating that the
episode is over.

- A method called <b> reset </b>, which returns the environment to its
initial state and obtains the first observation.

#### The action space

As mentioned, the actions that an agent can execute can be
discrete, continuous, or a combination of the two. 

<b> Discrete actions </b> are a fixed set of things that an agent can do, for
example, directions in a grid like left, right, up, or down. Another
example is a push buon, which could be either pressed or
released. Both states are mutually exclusive and this is the main
characteristic of a discrete action space, where only one action from
a finite set of actions is possible at a time.

<b> Continuous action </b> has a value aached to it, for example, a
steering wheel, which can be turned at a specific angle, or an
accelerator pedal, which can be pressed with different levels of
force. A description of a continuous action includes the boundaries
of the value that the action could have. In the case of a steering
wheel, it could be from − 720 degrees to 720 degrees. For an
accelerator pedal, it’s usually from 0 to 1.

Of course, we are not limited to a single action; the environment
could take multiple actions, such as pushing multiple buons
simultaneously or steering the wheel and pressing two pedals (the
brake and the accelerator). To support such cases, Gym defines a
special container class that allows the nesting of several action
spaces into one unified action.

#### The observation space

Observations are pieces of information
that an environment provides the agent with, on every timestamp,
besides the reward. Observations can be as simple as a bunch of
numbers or as complex as several multidimensional tensors
containing color images from several cameras. An observation can
even be discrete, much like action spaces. An example of a discrete
observation space is a lightbulb, which could be in two states – on
or off – given to us as a Boolean value.

So, you can see the similarity between actions and observations,
and that is how they have been represented in Gym’s classes. Let’s
look at a class diagram:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-3-1.png?raw=true)

The basic abstract <b> Space </b> class includes one property and three
methods that are relevant to us:

- shape : This property contain the shape of the space, identical
to NumPy arrays.

- sample() : This returns a random sample from the space.

- contains(x) : This checks whether the argument, x , belongs to
the space’s domain.

- seed() : This method allows us to initialize a random number
generator for the space and all subspaces. This is useful if you
want to get reproducible environment behavior across several
runs.

All these methods are abstract and reimplemented in each of the
<b> Space </b> subclasses:

- The Discrete class represents a mutually exclusive set of items,
numbered from 0 to n-1. If needed, you can redefine the
starting index with the optional constructor argument start .
The value n is a count of the items our Discrete object
describes. For example, Discrete(n=4) can be used for an action
space of four directions to move in [left, right, up, or down].

- The Box class represents an n-dimensional tensor of rational
numbers with intervals [low, high]. For instance, this could be
an accelerator pedal with one single value between 0.0 and 1.0,
which could be encoded by Box(low=0.0, high=1.0, shape=(1,),
dtype=np.float32) . Here, the shape argument is assigned a
tuple of length 1 with a single value of 1, which gives us a onedimensional tensor with a single value. The dtype parameter
specifies the space’s value type, and here, we specify it as a
NumPy 32-bit float. Another example of Box could be an Atari
screen observation (we will cover lots of Atari environments later), which is an RGB (red, green, and blue) image of size 210
× 160: Box(low=0, high=255, shape=(210, 160, 3),
dtype=np.uint8) . In this case, the shape argument is a tuple of
three elements: the first dimension is the height of the image,
the second is the width, and the third equals 3, which all
correspond to three color planes for red, green, and blue,
respectively. So, in total, every observation is a threedimensional tensor with 100 , 800 bytes.

- The final child of Space is a Tuple class, which allows us to
combine several Space class instances together. This enables
us to create action and observation spaces of any complexity
that we want. For example, imagine we want to create an action
space specification for a car. The car has several controls that
can be changed at every timestamp, including the steering
wheel angle, brake pedal position, and accelerator pedal
position. These three controls can be specified by three float
values in one single Box instance. Besides these essential
controls, the car has extra discrete controls, like a turn signal
(which could be off, right, or left) or horn (on or off). 

There are other Space subclasses defined in Gym, for example,
Sequence (representing variable-length sequences), Text (strings),
and Graph (where space is a set of nodes with connections between
them). But the three that we have described are the most useful
ones.

Every environment has two members of type Space : the action
_space and observation _space . This allows us to create generic
code that could work with any environment. Of course, dealing with
the pixels of the screen is different from handling discrete
observations (as in the former case, we may want to preprocess
images with convolutional layers or with other methods from the
computer vision toolbox); so, most of the time, this means
optimizing the code for a particular environment or group of
environments, but Gym doesn’t prevent us from writing generic
code.

#### The environment

The environment is represented in Gym by the Env class, which
has the following members:

- action _space : This is the field of the Space class and provides
a specification for allowed actions in the environment.

- observation _space : This field has the same Space class, but
specifies the observations provided by the environment.

- reset() : This resets the environment to its initial state,
returning the initial observation vector and the dict with extra
information from the environment.

- step() : This method allows the agent to take the action and returns information about the outcome of the action:
    - The next observation
    - The local reward
    - The end-of-episode flag
    - The flag indicating a truncated episode
    - A dictionary with extra information from the environment This method is a bit complicated; we will look at it in detail later in this section.

There are extra utility methods in the Env class, such as render() ,
which allows us to obtain the observation in a human-friendly form,
but we won’t use them. You can find the full list in Gym’s
documentation, but let’s focus on the core Env methods: reset()
and step() .

As reset is much simpler, we will start with it. The reset() method
has no arguments; it instructs an environment to reset into its
initial state and obtain the initial observation. Note that you have to
call reset() after the creation of the environment.

Besides the observation, reset() returns the second value – the
dictionary with extra environment-specific information. 

The step() method is the central piece in the environment’s
functionality. It does several things in one call, which are as follows:

- Telling the environment which action we will execute in the
next step

- Getting the new observation from the environment after this
action

- Getting the reward the agent gained with this step

- Getting the indication that the episode is over

- Getting the flag which signals an episode truncation (when
time limit is enabled, for example)

- Getting the dict with extra environment-specific information

The first item in the preceding list (action) is passed as the only argument to the    step()    method, and the rest are returned by this method.   More precisely, this is a tuple (Python tuple and not the    Tuple    class we discussed in the previous section) of five elements (    observation    ,    reward    ,    done    ,    truncated    , and    info    ).   They have these types and meanings:

- observation    : This is a NumPy vector or a matrix with observation data.  

- reward    : This is the float value of the reward.  

- done    : This is a Boolean indicator, which is    True    when the episode is over.   If this value is    True    , we have to call    reset()    in the environment, as no more actions are possible.  

- truncated    : This is a Boolean indicator, which is    True    when the episode is truncated.   For most environments, this is a    TimeLimit    (which is a way to limit length of episodes), but might have different meaning in some environments.   This flag is separated from    done    , because in some scenarios it might be useful to   distinguish situations ”agent reached the end of episode” and ”agent has reached the time limit of the environment.”   If    truncated    is    True    , we also have to call    reset()    in the environment, the same as with the    done    flag.

- info    : This could be anything environment-specific with extra information about the environment.   The usual practice is to ignore this value in general RL methods.

We call the    step()    method with an action to perform until the    done    or    truncated    flags become    True    .   Then, we can call    reset()    to start over.

#### Creating an environment

Gymnasium comes with an impressive list of 198 unique environments, which can be divided into several groups:

- Classic control problems    : These are toy tasks that are used in optimal   control theory and RL papers as benchmarks or demonstrations.   They are usually simple, with low-dimension observation and action spaces, but they are useful as quick checks when implementing algorithms.   Think about them as the ”MNIST for RL”.

- Atari 2600    : These are games from the classic game platform from the 1970s.   There are 63 unique games.  

- Algorithmic    : These are problems that aim to perform small computation tasks, such as copying the observed sequence or adding numbers.  

- Box2D    : These are environments that use the Box2D physics simulator to learn walking or car control.  

- MuJoCo    : This is another physics simulator used for several continuous control problems.  

- Parameter tuning    : This is RL being used to optimize NN parameters.  

- Toy text    : These are simple grid world text environments.

Of course, the total   number of RL environments supporting the Gym API is much larger.   For example, The Farama Foundation maintains several repositories related to special RL topics like multi-agent RL, 3D navigation, robotics, and web automation.   In addition, there are lots of [third-party repositories](https://gymnasium.farama.org/environments/third_party_environments).

### The random CartPole agent

```python
import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    total_reward = 0.0
    total_steps = 0
    obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if is_done:
            break

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
```

Here, we have imported the    gymnasium    package and created an environment called    CartPole    .   This environment is from the classic control group and its gist is to control the platform with a stick attached to its bottom part (see the following figure).  

The trickiness is that this stick tends to fall right or left and you need to balance it by moving the platform to the right or left at every step.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-3-2.png?raw=true)

The observation of this environment is four floating-point numbers containing information about the x coordinate of the stick’s center of mass, its speed, its angle to the platform, and its angular speed.   Of course, by applying some math and physics knowledge, it won’t be complicated to convert these numbers into actions when we need to balance the stick, but our problem is different – how do we learn how to balance this system    without     knowing    the exact meaning of the observed numbers and only by getting the reward?   The reward in this environment is 1, and it is given on every time step.   The episode continues until the stick falls, so to get a more accumulated reward, we need to balance the platform in a way to avoid the stick falling.

Here, we created the environment and initialized the counter of
steps and the reward accumulator. On the last line, we reset the
environment to obtain the first observation.

In the preceding loop, after sampling a random action, we asked
the environment to execute it and return to us the next observation
( obs ), the reward , the is _done , and the is _trunc flags. If the
episode is over, we stop the loop and show how many steps we have
taken and how much reward has been accumulated.

### Wrapper and Visualization

#### Wrapper

```python
import gymnasium as gym
import random


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, epsilon: float = 0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: gym.core.WrapperActType) -> gym.core.WrapperActType:
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            print(f"Random action {action}")
            return action
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v1"))

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print(f"Reward got: {total_reward:.2f}")
```

[Ref](https://gymnasium.farama.org/api/wrappers/)

#### Visualization

```python
import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.HumanRendering(env)
    # env = gym.wrappers.RecordVideo(env, video_folder="video")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print(f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")
    env.close()
```

## RL and Pytorch

...
