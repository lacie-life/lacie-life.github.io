---
title: Reinforcement Learning Algorithms - [Part 2]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-01-29 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# Reinforcement Learning Algorithms - [Part 2]

## RL formalisms

The following diagram shows two major RL entities — <b> agent </b> and  <b>  environment  </b>  — and their communication channels —  <b>  actions  </b>  ,   <b> reward  </b>  , and   <b> observations  </b>  :

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-1.png?raw=true)

### Reward

Reward is a scalar value we obtain periodically from the environment. Reward can be positive or negative, large or small, but it’s just a number. The purpose of reward is to tell our agent how well it has behaved.   We don’t define how frequently the agent receives this reward; it can be every second or once in an agent’s lifetime, although it’s common practice to receive rewards every fixed timestamp or at every environment interaction, just for convenience.   In the case of once-in-a-lifetime reward systems, all rewards except the last one will be zero.

The purpose of reward is to give an agent feedback about its success, and it’s a central thing in RL.   Basically, the term    reinforcement    comes from the fact that reward obtained by an agent should reinforce its behavior in a positive or negative way.   Reward is local, meaning that it reflects the benefits and losses achieved by the agent so far.   Of course, getting a large reward for some action doesn’t mean that, a second later, you won’t face dramatic consequences as a result of your previous decisions.

### The agent

An agent is   somebody or something who/that interacts with the environment by executing certain actions, making observations, and receiving eventual rewards for this.   In most practical RL scenarios, the agent is our piece of software that is supposed to solve some problem in a more-or-less efficient way.

### The environment

The   environment is everything outside of an agent.   In the most general sense, it’s the rest of the universe, but this goes slightly overboard and exceeds the capacity of even tomorrow’s computers, so we usually follow the general sense here.  

<i> The agent’s communication with the environment is limited to reward (obtained from the environment), actions (executed by the agent and sent to the environment), and observations (some information besides the reward that the agent receives from the environment). </i>

### Actions

Actions are   things that an agent can do in the environment.   Actions can, for example, be piece moves on the board (if it’s a board game), or doing homework (in the case of school).   They can be as simple as    move     pawn one space forward    or as complicated as    build a profitable startup     company.

In RL, we distinguish between two types of actions — discrete or continuous.   

- Discrete actions form the finite set of mutually exclusive things an agent can do, such as move left or right.   

- Continuous actions have some value attached to them, such as a car’s    turn the wheel    action having an angle and direction of steering.   Different angles could lead to a different scenario a second later, so just    turn the     wheel    is definitely not enough.


### Observations

Observations of the environment form   the second information channel for an agent, with the first being the    reward    .   You may be wondering why we need a separate data source.   The answer is convenience.   Observations are pieces of information that the environment provides the agent with that indicate what’s going on around the agent.  

Observations may be relevant to the upcoming reward (such as seeing a bank notification about being paid) or may not be.   Observations can even include reward information in some vague or obfuscated form, such as score numbers on a computer game’s screen.   Score numbers are just pixels, but potentially, we could convert them into reward values; it’s not a very complex task for a modern computer vision techniques.  

On the other hand, reward shouldn’t be seen as a secondary or unimportant thing — reward is the main force that drives the agent’s learning process.   If a reward is wrong, noisy, or just slightly off course from the primary objective, then there is a chance that training will go in the wrong direction.  
It’s also important to distinguish between an environment’s state and observations.   The state of an environment most of the time is    internal    to the environment and potentially includes every atom in the universe, which makes it impossible to measure everything about the environment.   Even if we limit the environment’s state to be small enough, most of the time, it will be either not possible to get full information about it or our measurements will contain noise.   This is completely fine, though, and RL was created to support such cases natively.

### Example

<b> Financial trading </b>

- Reward: An amount of profit is a reward for a trader buying and selling stocks.

- Agent: A trading system or a trader making decisions about order execution (buying, selling, or doing nothing).

- Actions: Actions are decisions to buy or sell stock.   “Do nothing and wait” also is an action.

- Observations: the environment is the whole financial market and everything that influences it.   This is a huge list of things, such as the latest news, economic and political conditions, weather, food supplies, and Twitter/X trends.   Even your decision to stay home today can potentially indirectly influence the world’s financial system (if you believe in the “butterfly effect”).   However, our observations are limited to stock prices, news, and so on.   We don’t have access to most of the environment’s state, which makes financial forecasting such a nontrivial thing.

## The theoretical foundations of RL

### Markov decision processes

<b> Markov decision processes (MDPs) </b>, which   will be   described like a Russian matryoshka doll: we will start from the simplest case of a  <b>  Markov process   (MP) </b>, then extend that with rewards, which will turn it into a  <b>  Markov reward process (MRP) </b>.   Then, we will put this idea into an extra envelope by adding actions, which will lead us to an MDP.

#### The Markov process

Let’s start   with   the simplest concept in the Markov family: the MP, which is also known as the <b>   Markov chain </b>   .   Imagine that you have some system in front of you that you can only observe.   What you observe is called <b> states </b>, and the system can switch between states according to   some laws of dynamics (most of the time unknown to you).   Again, you cannot influence the system, but can only watch the states changing.   All possible states for a system form a set called the  <b>  state     space  </b>  .   For MPs, we require this set of states to be finite (but it can be extremely large to compensate for this limitation).   Your observations form a sequence of states or a  <b>  chain  </b>  (that’s why MPs are also called Markov chains).

For example, looking at the simplest model of the weather in some city, we can observe the current day as sunny   or rainy, which is our state space.   A sequence of observations over time forms a chain of states, such as [    sunny    ,    sunny    ,    rainy    ,    sunny    , ...], and this is called    history    .   To call such a system an MP, it needs to fulfill the    Markov property    , which means that the future system dynamics from any state have to depend on this state only.   The main point of the Markov property is to make every observable state self-contained to describe the future of the system.   In other words, the Markov property requires the states of the system to be distinguishable from each other and unique.   In this case, only one state is required to model the future dynamics of the system and not the whole history or, say, the last   N   states.

In the case of our toy weather example, the Markov property limits our model to represent only the cases when a sunny day can be followed by a rainy one with the same probability, regardless of the number of sunny days we’ve seen in the past.   It’s not a very realistic model as, from common sense, we know that the chance of rain tomorrow depends not only on the current conditions but on a large number of other factors, such as the season, our latitude, and the presence of mountains and sea nearby.   It was recently proven that even solar activity has a major influence on the weather.   So, our example is really naïve, but it’s important to understand the limitations and make conscious decisions   about them.

Of course, if we want to make our model more complex, we can
always do this by extending our state space, which will allow us to
capture more dependencies in the model at the cost of a larger state
space. For example, if you want to capture separately the
probability of rainy days during summer and winter, then you can
include the season in your state.

In this case, your state space will be [ sunny+summer , sunny+winter ,
rainy+summer , rainy+winter ] and so on.

As your system model complies with the Markov property, you can capture transition probabilities with a  <b>  transition matrix  </b>  , which is a square matrix of the size   N   ×   N   , where   N   is the number of states in our model.   Every cell in a row,   i   , and a column,   j   , in the matrix contains the probability of the system to transition from state   i   to state   j   .

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-2.png?raw=true)

In this case, if we have a sunny day, then there is an 80% chance that the next day will be sunny and a 20% chance that the next day will be rainy.   If we observe a rainy day, then there is a 10% probability that the weather will become better and a 90% probability of the next day being rainy.

So, that’s it.   The formal definition of an MP is as follows:

- A set of states (   S   ) that a system can be in

- A transition matrix (   T   ), with transition probabilities, which defines the system dynamics

A useful visual representation of an MP is a graph with nodes corresponding to system states and edges, labeled with probabilities representing a possible transition from state to state.   If the probability of a transition is 0, we don’t draw an edge (there is no way to go from one state to another).   This kind of representation is also widely used in finite state machine representation, which is studied in automata theory.   For our sunny/rainy weather model, the graph is as shown here:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-3.png?raw=true)

Again, we’re   talking about observation only.   There is no way for us to influence the weather, so we just observe it and record our observations.

To give you a more complicated example, let’s consider another model called    Office Worker    (Dilbert, the main character in Scott Adams’ famous cartoons, is a good example).   His state space in our example has the following states:

- Home    : He’s not at the office  

- Computer    : He’s working on his computer at the office  

- Coffee    : He’s drinking coffee at the office  

- Chat    : He’s discussing something with colleagues at the office

The state transition graph is shown in the following figure:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-4.png?raw=true)

We assume that   our office worker’s weekday usually starts from the    Home    state and that he starts his day with    Coffee    without exception (no    Home    →    Computer    edge and no    Home    →    Chat    edge).   The preceding diagram also shows that workdays always end (that is, going to the    Home    state) from the    Computer    state.

The transition matrix for the diagram above is as follows:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-5.png?raw=true)

The transition probabilities could be placed directly on the state transition graph below:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-6.png?raw=true)

In practice, we rarely have the luxury of knowing the exact transition matrix.   A much more real-world situation is when we only have observations of our system’s states, which are also called  <b>  episodes  </b>  :

-  Home    →    Coffee    →    Coffee    →    Chat    →    Chat    →    Coffee    →    Computer    →    Computer    →    Home   

-  Computer    →    Computer    →    Chat    →    Chat    →    Coffee    →    Computer    →    Computer    →    Computer   

-  Home    →    Home    →    Coffee    →    Chat    →    Computer    →    Coffee    →    Coffee

It’s not complicated to estimate the transition matrix from our observations — we just count all the transitions from every state and normalize them to a sum of 1.   The more observation data we have, the closer our   estimation will be to the true underlying model.

It’s also   worth noting   that the Markov property implies  <b>  stationarity  </b>  (which means, the underlying transition distribution for any state does not change over time).  <b>  Non-stationarity  </b>  means that there is some hidden factor that influences our system dynamics, and this factor is not included in observations.   However, this contradicts the Markov property, which requires the underlying probability distribution to be the same for the same state regardless of the transition history.

It’s important to understand the difference between the actual transitions observed in an episode and the underlying distribution given in the transition matrix.   Concrete episodes that we observe are randomly sampled from the distribution of the model, so they can differ from episode to episode.   However, the probability of the concrete   transition to be sampled remains the same.   If this is not the case, Markov chain formalism becomes non-applicable.

#### Markov reward processes

To introduce   reward, we need to extend our MP model a bit.   First, we need to add value to our transition from state to state.   We already have probability, but probability is being used to capture the dynamics of the system, so now we have an extra scalar number without extra burden.  

Rewards can be represented in various forms.   The most general way is to have another square matrix, similar to the transition matrix, with a reward given for transitioning from state   i   to state   j   , which reside in row   i   and column   j   .

As mentioned, rewards can be positive or negative, large or small.   In some cases, this representation is redundant and can be simplified.   For example, if a reward is given for reaching the state regardless of the origin state, we can keep only <b> (    state    ,    reward    ) </b> pairs, which is a more compact representation.   However, this is applicable only if the reward value depends solely on the target state, which is not always the case.

The second thing we’re adding to the model is the discount factor   $γ$   (Greek letter “gamma”), which is a single number from 0 to 1 (inclusive).   The meaning of this will be explained after the extra characteristics of our MRP have been defined.

As you will remember, we observe a chain of state transitions in an MP.   This is still the case for a MRP, but for every transition, we have our extra quantity — reward.   So now, all our observations have a reward value attached to every transition of the system.

For every episode, we define  <b>  return  </b>  at the time   t   as   $G_t$    :


![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-7.png?raw=true)

The   $γ$   in the preceding formula is very important in RL.   For now, think about it as a measure of how far into the future we look to estimate the future return.   The closer its value is to 1, the more steps ahead of us we will take into account.

Now let’s try to understand what the formula for    return    means.   For every time point, we calculate return as a sum of subsequent rewards, but more distant rewards are multiplied by the discount factor raised to the power of the number of steps we are away from the starting point at   t   .   The discount factor stands for the foresightedness of the agent.   If   $γ   = 1$, then return,   $G_t$    , just equals a sum of all subsequent rewards and corresponds to the agent that has perfect visibility of any subsequent rewards.   If   $γ   = 0$,   $G_t$    will be just immediate reward without any subsequent state and will correspond to absolute short-sightedness.

These extreme values are useful only in corner cases, and most of the time,   $γ$   is set to something in between, such as 0   .   9 or 0   .   99.   In this case, we will look into future rewards, but not too far.   The value of   $γ   = 1$ might be applicable in situations of short finite episodes.

This return quantity is not very useful in practice, as it was defined for every   specific chain we observed from our MRP, so it can vary   widely, even for the same state.   However, if we go to the extreme and calculate the mathematical expectation of return for any state (by averaging a large number of chains), we will get a much more practical quantity, which is called the  <b>  value of the     state  </b>  :

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-8.png?raw=true)

This interpretation is simple—for every state,   s   , the value,  $ V   (   s   )$, is the average (or expected) return we get by following the   Markov reward process.

To represent this theoretical knowledge practically, let’s extend our office worker (Dilbert) process with a reward and turn it into a    Dilbert reward process     (DRP)    .   Our reward values will be as follows:

- Home    →    Home    : 1 (as it’s good to be home)  

- Home    →    Coffee    : 1  

- Computer    →    Computer    : 5 (working hard is a good thing)  

- Computer    →    Chat    :   −   3 (it’s not good to be distracted)  

- Chat    →    Computer    : 2  

- Computer    →    Coffee    : 1  

- Coffee    →    Computer    : 3  

- Coffee    →    Cofee    : 1  

- Coffee    →    Chat    : 2  

- Chat    →    Coffee    : 1  

- Chat    →    Chat    : -1 (long conversations become boring)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-9.png?raw=true)

Let’s return to our $γ$ parameter and think about the values of states
with different values of $γ$ . We will start with a simple case: $γ = 0$.
How do you calculate the values of states here? To answer this
question, let’s fix our state to Chat . What could the subsequent
transition be? The answer is that it depends on chance. According
to our transition matrix for the Dilbert process, there is a 50%
probability that the next state will be Chat again, 20% that it will be
Coffee , and 30% that it will be Computer . When $γ = 0$, our return is
equal only to a value of the next immediate state. So, if we want to
calculate the value of the Chat state in the preceding diagram, then
we need to sum all transition values and multiply that by their
probabilities:

V   (chat)  =  −   1   ⋅   0   .   5 + 2   ⋅   0   .   3 + 1   ⋅   0   .   2 = 0   .   3  

V   (coffee)  =  2   ⋅   0   .   7 + 1   ⋅   0   .   1 + 3   ⋅   0   .   2 = 2   .   1  

V   (home)  =  1   ⋅   0   .   6 + 1   ⋅   0   .   4 = 1   .   0  

V   (computer)  =  5   ⋅   0   .   5 + (   −   3)   ⋅   0   .   1 + 1   ⋅   0   .   2 + 2   ⋅   0   .   2 = 2   .   8

So,    Computer    is the most valuable state to be in (if we care only about immediate reward), which is not surprising as    Computer    →    Computer    is frequent, has a large reward, and the ratio of interruptions is not too high.  
Now a trickier question — what’s the value when   $γ   = 1$?   Think about this carefully.   The answer is that the value is infinite for all states.   Our diagram doesn’t contain  <b>  sink states  </b>  (states without outgoing transitions), and when our discount equals 1, we care about a potentially infinite number of transitions in the future.   As you’ve seen in the case of   $γ   = 0$, all our values are positive in the short term, so the sum of the infinite number of positive values will give us an infinite value, regardless of the starting state.

This infinite result shows us one of the reasons to introduce   $γ$   into a MRP instead of just summing all future rewards.   In most cases, the process can have an infinite (or large) amount of transitions.   As it is not very practical to deal with infinite values, we would like to limit the horizon we calculate values for.   Gamma with a value less than 1 provides such a limitation, and we will discuss this later in this book.   On the other hand, if you’re dealing with finite-horizon environments (for example, the tic-tac-toe game, which is limited by at most nine steps), then it will be fine to use   $γ   = 1$.

As I already mentioned about the MRP,   $γ$   is usually set to a value between 0 and 1.   However, with such values, it becomes almost impossible to calculate them accurately by hand, even for MRPs as small as our Dilbert example, because it will require   summing hundreds of values.   Computers are good at tedious tasks such as this, and there are several simple methods that can quickly calculate values for MRPs for given transition and reward matrices.

### Adding actions to MDP

You may already   have ideas about how to extend our MDP to include actions.   Firstly, we must add a set of actions $(   A   )$, which has to be finite.   This is our agent’s    action space    .   Secondly, we need to condition our transition matrix with actions, which basically means that our matrix needs an extra action dimension, which turns it into a cuboid of shape  $ |   S   |×|   S   |×|   A   | $  , where   S   is an our state space and   A   is an action space.  

If you remember, in the case of MPs and MRPs, the transition matrix had a square form, with the source state in rows and target state in columns.   So, every row,   i   , contained a list of probabilities to jump to every state.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-10.png?raw=true)

In case of an   MDP, the agent no longer passively observes state transitions, but can actively choose an action to take at every state transition.   So, for every source state, we don’t have a list of numbers, but we have a matrix, where the <b>   depth  </b>  dimension contains actions that the agent can take, and the other dimension is what the target state system will jump to after actions are performed by the agent.   The following diagram shows our new transition table, which became a cuboid with the source state as the  <i>  height  </i>  dimension (indexed by   i   ), the target state as the  <i>  width  </i>  (   j   ), and the action the agent can take as the  <b>  depth  </b>  (   k   ) of the transition table:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-11.png?raw=true)

So, in general, by choosing an action, the agent can affect the probabilities of the target states, which is a useful ability.  
To give you an idea of why we need so many complications, let’s imagine a small robot that lives in a 3   ×   3 grid and can execute the actions    turn left    ,    turn right    , and    go forward    .   The state of the world is the robot’s position plus orientation (up, down, left, and right), which gives us 3   ×   3   ×   4 = 36 states (the robot can be at any location in any orientation).  

Also, imagine that the robot has imperfect motors (which is frequently the case in the real world), and when it executes    turn left    or    turn right    , there is a 90% chance that the desired turn happens, but sometimes, with a 10% probability, the wheel slips and the robot’s position stays the same.   The same happens with    go     forward    — in 90% of cases it works, but for the   rest (10%) the robot stays at the same position.

In    Figure          below     , a small part of a transition diagram is shown, displaying the possible transitions from the state    (1, 1), up    , when the robot is in the center of the grid and facing up.   If the robot tries to move forward, there is a 90% chance that it will end up in the state    (0, 1), up    , but there is a 10% probability that the wheels will slip and the target position will remain    (1, 1),     up    .

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-12.png?raw=true)

To properly capture all these details about the environment and possible reactions to the agent’s actions, the general MDP has a 3D transition matrix with the dimensions source state, action, and target state.  
Finally, to turn our MRP into an MDP, we need to add actions to our reward matrix in the same way that we did with the transition matrix.   Our reward matrix will depend not only on the state but also on the action.   In other words, the reward the agent obtains will now   depend not only on the state it ends up in but also on the action that leads to this state.   Now, with a formally defined MDP, we’re finally ready to cover the most important thing for MDPs and RL:  <b>  policy  </b>  .

### Policy

The simple   definition of policy is that it is some set of rules that defines the agent’s behavior.   Even for fairly simple environments, we can have a variety of policies.   For example, in the preceding example with the robot in the grid world, the agent can have different policies, which will lead to different sets of visited states.   For example, the robot can perform the following actions:  

- Blindly move forward regardless of anything  

- Try to go around obstacles by checking whether that previous    forward    action failed  

- Funnily spin around by always turning right to entertain its creator  

- Choose an action randomly regardless of position and orientation, modeling a drunk robot in the grid world scenario  

You may remember that the main objective of the agent in RL is to gather as much return as possible.   So, again, different policies can give us different amounts of return, which makes it important to find a good policy.   This is why the notion of policy is important.

Formally, policy is defined as the probability distribution over actions for every possible state:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/rl/part-2-14.png?raw=true)

This is defined as probability and not as a concrete action to introduce randomness into an agent’s behavior.   Deterministic policy is a special case of probabilistics with the needed action having 1 as its probability.  

Another useful notion is that if our policy is fixed and not changing during training (i.e., when the policy always returns the same actions for the same states), then our MDP becomes a MRP, as we can reduce the transition and reward matrices with a policy’s probabilities and get rid of the action dimensions.

## The anatomy of the agent python implemetation

There are several fundamental concepts in RL:

- <b> The agent : </b> A thing, or person, that takes an active role. In
practice, the agent is some piece of code that implements some
policy. Basically, this policy decides what action is needed at
every time step, given our observations.

- <b> The environment : </b> Everything that is external to the agent and
has the responsibility of providing observations and giving
rewards. The environment changes its state based on the
agent’s actions.

Let’s explore how both can be implemented in Python for a simple
situation. We will define an environment that will give the agent
random rewards for a limited number of steps, regardless of the
agent’s actions. This scenario is not very useful in the real world,
but it will allow us to focus on specific methods in both the
environment and agent classes.

```python
import random
from typing import List


class Environment: # Environment 
    def __init__(self):
        self.steps_left = 10

    def get_observation(self) -> List[float]: # Return  the current environment’s observation to the agent
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]: # Allows the agent to query the set of actions it can execute
        return [0, 1]

    def is_done(self) -> bool: # Signals the end of the episode to the agent
        return self.steps_left == 0

    def action(self, action: int) -> float: # Handles an agent’s action and returns the reward for this action
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent: # Agent
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        current_obs = env.get_observation() # Observe the environment
        actions = env.get_actions() # Make a decision about the action to take based on the observations
        reward = env.action(random.choice(actions)) # Submit the action to the environment
        self.total_reward += reward # Get the reward for the current step


if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)
```

The simplicity of the preceding code illustrates the important basic
concepts of the RL model. The environment could be an extremely
complicated physics model, and an agent could easily be a large
neural network that implements the latest RL algorithm, but
the basic paern will stay the same – at every step, the agent will
take some observations from the environment, do its calculations,
and select the action to take. The result of this action will be a
reward and a new observation.


