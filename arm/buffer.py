import pickle
import time

import numpy as np


class ReplayBuffer():

    def __init__(self, curriculum=(), frame_buffer=1, n_step_size=1, gamma=0.9):
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
        self.bounded_idcs = np.array([])

        self.curriculum = curriculum
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

    def __vec_idcs(self):
        # episode start indices
        epi_start_idcs = np.insert(np.nonzero(
            self.vec_done)[0][:-1] + 1, 0, 0)

        # init index array
        self.idcs = np.arange(len(self))

        idcs = np.repeat(self.idcs, self.frame_buffer)
        idcs = idcs.reshape(len(self), self.frame_buffer)
        # subtract frame buffer length
        idx_sub = np.arange(self.frame_buffer-1, -1, -1)
        idx_sub = np.tile(idx_sub, len(self)).reshape(-1, self.frame_buffer)
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

        self.__compute_curriculum()

    def __n_step_reward(self):
        rewards = self.vec_rewards
        idcs = np.nonzero(self.vec_done)[0] + 1
        traj_rewards = np.split(rewards, idcs)[:-1]
        n_step_discount = np.power(
            np.full(self.n_step_size, self.gamma), np.arange(self.n_step_size))
        n_step = np.array([np.sum(trajectory[step_idx:step_idx+self.n_step_size] *
                                  n_step_discount[:trajectory.shape[0]-step_idx])
                           for trajectory in traj_rewards for step_idx in range(trajectory.shape[0])])
        # create array of ones for all trajectories, make last n_step_size entries 0
        est_rew_weights = np.concatenate([np.concatenate((np.ones(
            trajectory.shape[0] - self.n_step_size), np.zeros(self.n_step_size)))
            for trajectory in traj_rewards])
        self.n_step = n_step.astype(np.float32)
        self.est_rew_weights = est_rew_weights.astype(np.float32)

    def __compute_curriculum(self):
        epi_start_idcs = np.insert(np.nonzero(
            self.done)[0][:-1] + 1, 0, 0)
        if self.curriculum:
            curriculum_idcs = np.arange(*self.curriculum)
            bounded_length = curriculum_idcs.shape[0]
            if curriculum_idcs[0] < 0:
                epi_idcs = np.append(epi_start_idcs[1:], len(self))
            else:
                epi_idcs = epi_start_idcs
            curriculum_idcs = np.tile(curriculum_idcs, epi_idcs.shape[0])
            curriculum_idcs = curriculum_idcs + \
                np.repeat(epi_idcs, bounded_length)
            self.curriculum_idcs = self.idcs[curriculum_idcs]
        else:
            self.curriculum_idcs = self.idcs

    def append(self, obs, next_obs, action, reward, done):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.next_obs.append(np.array(next_obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.int64))
        self.rewards.append(np.array(reward, dtype=np.float32))
        self.done.append(np.array(done, dtype=np.int64))

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

    def iterate(self, batch_size, random=False, curriculum=False):

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

    def vectorize(self, frame_buffer=0, curriculum=(), n_step_size=0, gamma=0):
        self.__vectorize()
        idcs = False or len(self) != self.idcs.shape[0]
        if frame_buffer and frame_buffer != self.frame_buffer:
            self.frame_buffer = frame_buffer
            idcs = True
        if curriculum and curriculum != self.curriculum:
            self.curriculum = curriculum
            idcs = True
        if idcs:
            self.__vec_idcs()
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
