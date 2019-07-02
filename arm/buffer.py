"""Replay buffer used to record environment state and agent behavior
"""
import numpy as np


class ReplayBuffer():
    """Replay buffer used to record observations, next observations, actions,
    rewards and if a node is terminal. Can be passed to the
    arm algorithm for training.

    Keyword Arguments:
        curriculum {tuple} -- range of indices to train on (default: {()})
        curriculum_mode {str} -- either 'done' or 'reward', curriculum
                                 learning based on episodes or rewards
                                 (default: {'done'})
        frame_buffer {int} -- number of frames per observation (default: {1})
        n_step_size {int} -- number of steps to accumulate rewards (default: {1})
        gamma {float} -- discount factor per reward step (default: {0.9})
    """

    def __init__(self,
                 curriculum=(),
                 curriculum_mode='done',
                 frame_buffer=1,
                 n_step_size=1,
                 gamma=0.9):
        self.obs = []
        self.vec_obs = np.array([])
        self.next_obs = []
        self.vec_next_obs = np.array([])
        self.actions = []
        self.vec_actions = np.array([])
        self.rewards = []
        self.vec_rewards = np.array([])
        self.done = []
        self.vec_done = np.array([])

        self.n_step = np.array([])
        self.est_rew_weights = np.array([])

        self.idcs = np.array([])
        self.obs_idcs = np.array([])
        self.curriculum_idcs = np.array([])

        self.curriculum = curriculum
        self.curriculum_mode = curriculum_mode
        self.frame_buffer = frame_buffer
        self.n_step_size = n_step_size
        self.gamma = gamma

    def __getitem__(self, idcs):
        obs_idcs = self.obs_idcs[idcs]
        return (self.vec_obs[obs_idcs],
                self.vec_next_obs[idcs],
                self.vec_actions[idcs],
                self.vec_rewards[idcs],
                self.vec_done[idcs])

    def __add__(self, other):
        if not isinstance(other, ReplayBuffer):
            raise TypeError('only two replay buffers can be added together')
        self.obs += other.obs
        self.next_obs += other.next_obs
        self.actions += other.actions
        self.rewards += other.rewards
        self.done += other.done
        return self

    def __len__(self):
        return len(self.rewards)

    def __vec_idcs(self, mode):
        # init index array
        self.idcs = np.arange(len(self))

        if self.frame_buffer > 1:
            # episode start indices
            epi_start_idcs = np.insert(np.nonzero(
                self.vec_done)[0][:-1] + 1, 0, 0)
            idcs = np.repeat(self.idcs, self.frame_buffer)
            idcs = idcs.reshape(len(self), self.frame_buffer)
            # subtract frame buffer length
            idx_sub = np.arange(self.frame_buffer-1, -1, -1)
            idx_sub = np.tile(idx_sub, len(
                self)).reshape(-1, self.frame_buffer)
            idcs = idcs - idx_sub

            # compute frame buffer overlap insert and corresponding idcs
            insert = np.cumsum(
                np.ones((self.frame_buffer - 1, self.frame_buffer)), 1).astype(int)
            # add consecutive larger number to each row
            insert = (insert.T + np.arange(self.frame_buffer - 1)).T
            # subtract frame buffer and clip to zero
            insert = np.clip(insert - self.frame_buffer, 0, None)
            # repeat for number of episodes
            insert = np.tile(insert.T, epi_start_idcs.shape[0]).T
            # add episode index to start insert
            rep_epi_start = np.repeat(epi_start_idcs, self.frame_buffer - 1)
            insert = (insert.T + rep_epi_start).T

            # add frame buffer range onto insert idcs
            insert_idcs = rep_epi_start + np.tile(np.arange(self.frame_buffer - 1),
                                                  epi_start_idcs.shape[0])

            # insert frame buffer overlap at indices
            idcs[insert_idcs] = insert
            self.obs_idcs = idcs
        else:
            self.obs_idcs = self.idcs

        self.__compute_curriculum(mode)

    def __n_step_reward(self):
        # split trajactories
        rewards = self.vec_rewards
        idcs = np.nonzero(self.vec_done)[0] + 1
        traj_rewards = np.split(rewards, idcs)[:-1]
        # compute n step discount array
        n_step_discount = np.power(
            np.full(self.n_step_size, self.gamma), np.arange(self.n_step_size))
        # compute n step rewards
        n_step = np.array([np.sum(trajectory[step_idx:step_idx+self.n_step_size] *
                                  n_step_discount[:trajectory.shape[0]-step_idx])
                           for trajectory in traj_rewards for step_idx in range(trajectory.shape[0])])
        # create array of ones for all trajectories, make last n_step_size entries 0
        est_rew_weights = np.concatenate([np.concatenate((np.ones(
            trajectory.shape[0] - self.n_step_size), np.zeros(self.n_step_size)))
            for trajectory in traj_rewards])
        self.n_step = n_step.astype(np.float32)
        self.est_rew_weights = est_rew_weights.astype(np.float32)

    def __compute_curriculum(self, mode):
        if self.curriculum:
            if mode == 'done':
                epi_start_idcs = np.insert(
                    np.nonzero(self.vec_done)[0][:-1] + 1, 0, 0)
            elif mode == 'reward':
                epi_start_idcs = np.insert(np.nonzero(
                    self.vec_rewards)[0][:-1] + 1, 0, 0)
            curriculum_idcs = np.arange(*self.curriculum)
            bounded_length = curriculum_idcs.shape[0]
            if curriculum_idcs[0] < 0:
                epi_idcs = np.append(epi_start_idcs[1:], len(self))
            else:
                epi_idcs = epi_start_idcs
            curriculum_idcs = np.tile(curriculum_idcs, epi_idcs.shape[0])
            curriculum_idcs = curriculum_idcs + \
                np.repeat(epi_idcs, bounded_length)
            curriculum_idcs = np.unique(curriculum_idcs)
            self.curriculum_idcs = self.idcs[curriculum_idcs]
        else:
            self.curriculum_idcs = self.idcs

    def __vectorize(self):
        if self.vec_obs.shape[0] != len(self):
            self.vec_obs = np.stack(self.obs).astype(np.float32)
        if self.vec_next_obs.shape[0] != len(self):
            self.vec_next_obs = np.stack(self.next_obs).astype(np.float32)
        if self.vec_actions.shape[0] != len(self):
            self.vec_actions = np.stack(self.actions).astype(np.int64)
        if self.vec_rewards.shape[0] != len(self):
            self.vec_rewards = np.stack(self.rewards).astype(np.float32)
        if self.vec_done.shape[0] != len(self):
            self.vec_done = np.stack(self.done).astype(np.int64)

        return self

    def append(self, obs, next_obs, action, reward, done):
        """Adds data to replay buffer
        
        Arguments:
            obs {iterable} -- observations of state
            next_obs {iterable} -- observations of next state
            action {float} -- action of state
            reward {float} -- reward obtained by action
            done {bool} -- toggle if episode is done
        """
        self.obs.append(np.array(obs, dtype=np.float32))
        self.next_obs.append(np.array(next_obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.int64))
        self.rewards.append(np.array(reward, dtype=np.float32))
        self.done.append(np.array(done, dtype=np.int64))

    def iterate(self, batch_size=1, random=False, curriculum=False):
        """Generator to iterate over replay buffer

        Keyword Arguments:
            batch_size {int} -- number of samples per batch (default: {0})
            random {bool} -- return random samples (default: {False})
            curriculum {bool} -- only use samples from curriculum (default: {False})
        """
        batch_iters = 0
        shown_data = 0
        if curriculum:
            iter_idcs = self.curriculum_idcs
        else:
            iter_idcs = self.idcs

        while shown_data < iter_idcs.shape[0]:
            batch_iters += 1
            if random:
                idcs = np.random.choice(iter_idcs.shape[0], batch_size)
            else:
                end_idx = min(shown_data + batch_size, iter_idcs.shape[0])
                idcs = np.arange(shown_data, end_idx)
                idcs = iter_idcs[idcs]
            shown_data = batch_iters * batch_size
            yield self[idcs]

    def vectorize(self,
                  curriculum=(),
                  curriculum_mode='done',
                  frame_buffer=0,
                  n_step_size=0,
                  gamma=0):
        """Vectorizes buffer for training

        Keyword Arguments:
            curriculum {tuple} -- range of indices to train on (default: {()})
            curriculum_mode {str} -- either 'done' or 'reward', curriculum
                                     learning based on episodes or rewards
                                     (default: {'done'})
            frame_buffer {int} -- number of frames per observation (default: {1})
            n_step_size {int} -- number of steps to accumulate rewards (default: {1})
            gamma {float} -- discount factor per reward step (default: {0.9})

        Returns:
            ReplayBuffer -- vectorized buffer (no copy)
        """
        self.__vectorize()
        idcs = False or len(self) != self.idcs.shape[0]
        if frame_buffer and frame_buffer != self.frame_buffer:
            self.frame_buffer = frame_buffer
            idcs = True
        if curriculum and curriculum != self.curriculum:
            self.curriculum = curriculum
            idcs = True
        if curriculum_mode and curriculum_mode != self.curriculum_mode:
            self.curriculum_mode = curriculum_mode
            idcs = True
        if idcs:
            self.__vec_idcs(curriculum_mode)
        n_step_reward = False or len(self) != self.n_step.shape[0]
        if n_step_size and n_step_size != self.n_step_size:
            self.n_step_size = n_step_size
            n_step_reward = True
        if gamma and gamma != self.gamma:
            self.gamma = gamma
            n_step_reward = True
        if n_step_reward:
            self.__n_step_reward()
        return self
