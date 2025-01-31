---
title: Reinforcement Learning Algorithms - [Part 4]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-01-30 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Reinforcement Learning Algorithms - [Part 4]

## The taxonomy of RL methods

All the methods in RL can be classified into various groups:

- Model-free or model-based

- Value-based or policy-based

- On-policy or off-policy

There are other ways that you can taxonomize RL methods, but, for
now, we are interested in the above three. Let’s define them, as the
specifics of your problem can influence your choice of a particular
method

The term <b> model-free </b> means that the method doesn’t build a model
of the environment or reward; it just directly connects observations
to actions (or values that are related to actions). In other words, the
agent takes current observations and does some computations on
them, and the result is the action that it should take. In contrast,
<b> model-based </b> methods try to predict what the next observation
and/or reward will be. Based on this prediction, the agent tries to
choose the best possible action to take, very often making such predictions multiple times to look more and more steps into the
future.

Both classes of methods have strong and weak sides, but usually
pure model-based methods are used in deterministic environments,
such as board games with strict rules. On the other hand, model-free
methods are usually easier to train because it’s hard to build good
models of complex environments with rich observations. All of the
methods described in this book are from the model-free category, as
those methods have been the most active area of research for the
past few years. Only recently have researchers started to mix the
benefits from both worlds.

Looking at this from another angle, <b> policy-based </b> methods directly
approximate the policy of the agent, that is, what actions the agent
should carry out at every step. The policy is usually represented by a
probability distribution over the available actions. Alternatively, the
method could be <b> value-based </b>. In this case, instead of the
probability of actions, the agent calculates the value of every
possible action and chooses the action with the best value. 

The third important classification of methods is <b> on-policy </b> versus
<b> off-policy </b>. off-policy as
the ability of the method to learn from historical data (obtained by a
previous version of the agent, recorded by human demonstration, or
just seen by the same agent several episodes ago). On the other
hand, on-policy methods require fresh data for training, generated
from the policy we’re currently updating. They cannot be trained on
old historical data because the result of the training will be wrong.
This makes such methods much less data-efficient (you need much
more communication with the environment), but in some cases, this
is not a problem (for example, if our environment is very lightweight
and fast, so we can quickly interact with it).


So, our cross-entropy method is model-free, policy-based, and onpolicy, which means the following:

- It doesn’t build a model of the environment; it just says to the
agent what to do at every step

- It approximates the policy of the agent

- It requires fresh data obtained from the environment

## The cross-entropy method

### Simple praticle

The explanation of the cross-entropy method can be split into two
unequal parts: practical and theoretical. The practical part is
intuitive in nature, while the theoretical explanation of why the
cross-entropy method works and what happens, is more
sophisticated.

You may remember that the central and trickiest thing in RL is the
agent, which tries to accumulate as much total reward as possible by
communicating with the environment. In practice, we follow a
common machine learning ( ML ) approach and replace all of the complications of the agent with some kind of nonlinear trainable
function, which maps the agent’s input (observations from the
environment) to some output. The details of the output that this
function produces may depend on a particular method or a family of
methods (such as value-based or policy-based methods), as
described in the previous section. As our cross-entropy method is
policy-based, our nonlinear function ( neural network ( NN ))
produces the policy , which basically says for every observation
which action the agent should take. In research papers, policy is
denoted as $ π ( a | s )$, where a are actions and s is the current state.
This is illustrated in the following diagram:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-4-1.png?raw=true)

In practice, the policy is usually represented as a probability
distribution over actions, which makes it very similar to a
classification problem, with the number of classes being equal to the
number of actions we can carry out.

This abstraction makes our agent very simple: it needs to pass an
observation from the environment to the NN, get a probability
distribution over actions, and perform random sampling using the
probability distribution to get an action to carry out. This random
sampling adds randomness to our agent, which is a good thing
because at the beginning of the training, when our weights are
random, the agent behaves randomly. As soon as the agent gets an
action to issue, it fires the action to the environment and obtains the
next observation and reward for the last action.

During the agent’s lifetime, its experience is presented as episodes.
Every episode is a sequence of observations that the agent has got
from the environment, actions it has issued, and rewards for these
actions. Imagine that our agent has played several such episodes.
For every episode, we can calculate the total reward that the agent
has claimed. It can be discounted or not discounted; for simplicity,
let’s assume a discount factor of $γ = 1$, which just means an
undiscounted sum of all local rewards for every episode. This total
reward shows how good this episode was for the agent. It is
illustrated below, which contains four episodes (note that
different episodes have different values for $o_i$, $a_i$, and $r_i$):

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-4-2.png?raw=true)


Every cell represents the agent’s step in the episode. Due to
randomness in the environment and the way that the agent selects
actions to take, some episodes will be beer than others. The core of
the cross-entropy method is to throw away bad episodes and train
on beer ones. So, the steps of the method are as follows:

1. Play N episodes using our current model and environment.

2. Calculate the total reward for every episode and decide on a
reward boundary. Usually, we use a percentile of all rewards,
such as the 50th or 70th.

3. Throw away all episodes with a reward below the boundary.

4. Train on the remaining ”elite” episodes (with rewards higher
than the boundary) using observations as the input and issued
actions as the desired output.

5. Repeat from step 1 until we become satisfied with the result.

So, that’s the cross-entropy method’s description. With the
preceding procedure, our NN learns how to repeat actions, which
leads to a larger reward, constantly moving the boundary higher and
higher. Despite the simplicity of this method, it works well in basic
environments, it’s easy to implement, and it’s quite robust against
changing hyperparameters, which makes it an ideal baseline
method to try. Let’s now apply it to our CartPole environment.

```python
#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]


def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.tensor(obs, dtype=torch.float32)
        act_probs_v = sm(net(obs_v.unsqueeze(0)))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch: tt.List[Episode], percentile: float) -> \
        tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))

    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    print(net)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 475:
            print("Solved!")
            break
    writer.close()
```

### The theoretical background of the cross-entropy method

The basis of the cross-entropy method lies in the importance
sampling theorem, which states this:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-4-3.png?raw=true)

In our RL case,   $H   (   x   )$ is a reward value obtained by some policy   $x$   , and   $p   (   x   )$ is a distribution of all possible policies.   We don’t want to maximize our reward by searching all possible policies; instead, we want to find a way to approximate   $p   (   x   )   H   (   x   )$ by   $q   (   x   )$, iteratively minimizing the distance between them.   The distance between two probability distributions is calculated by    Kullback-Leibler (KL)    divergence, which is as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-4-4.png?raw=true)

The first term in KL is called entropy and it doesn’t depend on   $p_2    (   x   )$, so it could be omitted during the minimization.   The second term is called    cross-entropy    , which is a very common optimization objective in deep learning.  
Combining both formulas, we can get an iterative algorithm, which starts with   $q_0    (   x   ) =   p   (   x   )$ and on every step improves.   This is an approximation of   $p   (   x   )   H   (   x   )$ with an update:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-4-5.png?raw=true)

This is a generic cross-entropy method that can be significantly simplified in our RL case.   We replace our   $H   (   x   )$ with an indicator function, which is 1 when the reward for the episode is above the threshold and 0 when the reward is below.   Our policy update will look like this:


![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-4-6.png?raw=true)

Strictly   speaking, the preceding formula misses the normalization term, but it still works in practice without it.   So, the method is quite clear: we sample episodes using our current policy (starting with some random initial policy) and minimize the negative log likelihood of the most successful samples and our policy.

[Ref](https://link.springer.com/referenceworkentry/10.1007/978-1-4419-1153-7_131)
