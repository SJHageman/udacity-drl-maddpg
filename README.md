# udacity-drl-maddpg

This is a project as part of the Udacity Deep Reinforcement Learning nanodegree

## Introdcution
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Environment setup
Some steps are required to set up the environment. This is what was used to setup the environment on my Windows 10 computer.

1. Clone this repository and the [DRL nanodegree repo](https://github.com/udacity/deep-reinforcement-learning)
2. Download the Unity environment from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip) for Windows x86-64.
3. Unzip the contents and place that in the p3_collab-compet folder in the cloned DRL nanodegree repo
4. Create conda environment using the YAML file in this repo using the command
```
conda env create --name drlnd --file=environment-drlnd.yml
```
5. Execute the Tennis_training.ipynb notebook to train the agents and save the models (note: some paths might need to be update to reflect where the relevant repos and files are in your filesystem)
6. Execute the Tennis_demo.ipynb notebook to watch the trained agents in action! 
