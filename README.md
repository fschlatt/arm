# Advantage Based Regret Minimization

This repository contains a pytorch implementation of advantage based regret minimization from Jin et. al. [1], with some further additions. These include combining the advantage and value networks into one and adding the option for simple curriculum learning and future prediction.

## How to train

The algorithm is split into three main parts. The replay buffer, the policy and the actual arm algorithm. To train an agent, collect observations, actions, rewards etc... using the policy into a replay buffer. Then pass the replay buffer with the network used in the policy to the arm algorithm.

If you would like to use future prediction, the network must include a future toggle and include the action in its forward function `def forward(self, obs, action, future=False)`. If future is true, the forward pass must return an output with the same dimensions as a single observation.

If you would like to use curriculum learning, pass curriculum bounds via the `vectorize()` function of the replay buffer. E.g. to train on only the last 20 steps of an episode, pass curriculum bounds of `(-20, 0)`. 

To record the training results and losses in a tensorboard, use at least tensorflow 2.0. Install using `pip install -q tf-nightly-2.0-preview`.

## Pong Example
A working example with hyperparameters for Pong is included. The recommended way to run the example is to use this Colab notebook https://colab.research.google.com/drive/1XAi-le8EYaG1LzFnzov7e_u6WAXOiGeY. Switch to a GPU runtime for increased performance. Otherwise, clone the repository and run `example.py` (pytorch, gym, atari-gym required). If you are using windows, atari-py can be installed using `pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py`.

[1]: Jin, Peter, Kurt Keutzer, and Sergey Levine. "Regret minimization for partially observable deep reinforcement learning." arXiv preprint arXiv:1710.11424 (2017).
