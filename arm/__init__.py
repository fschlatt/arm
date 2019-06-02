"""Pytorch implementation of advantage based regret minimization [1]

[1]: Jin, Peter, Kurt Keutzer, and Sergey Levine.
\"Regret minimization for partially observable deep reinforcement learning.\"
arXiv preprint arXiv:1710.11424 (2017).
"""
from .arm import Arm
from .buffer import ReplayBuffer
from .policy import Policy
