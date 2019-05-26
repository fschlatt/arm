"""Arm module, used to train arm algorithm
"""
import copy

import numpy as np
import torch
try:
    import tensorflow as tf
except ImportError:
    TENSORFLOW = False


class Arm(torch.nn.Module):
    """Arm algorithm - initialized with arbitrary network
    that maps observations to vector of size one greater than
    the size of the action space. If one step future prediction
    is used, the network needs to implement a future toggle in
    the forward function and return an output whose dimensions
    are equal to the dimensions of a single observation.
    Can be trained by passing a replay buffer to the `train_batch`
    function.

    Arguments:
        network {torch.nn.Module} -- arbitrary pytorch network
        iters {int} -- number of training iterations per batch
        mini_batch_size {int} -- number of samples per iteration
        tau {float} -- target network update offset

    Keyword Arguments:
        future {bool} -- toggle to use one step future prediction (default: {False})
    """

    def __init__(self, network, iters, mini_batch_size, tau, future=False):
        super(Arm, self).__init__()

        self.network = network
        self.target_network = copy.deepcopy(network)
        self.iters = iters
        self.mini_batch_size = mini_batch_size
        self.tau = tau
        self.future = future
        self.device = network.device

        self.epochs = 0
        self.steps = 0

    def __compute_targets(self, replay_buffer):

        first_batch = self.epochs == 0
        first_batch = False

        # precompute all v and q target values
        if first_batch:
            q_plus = torch.zeros([replay_buffer.curriculum_idcs.shape[0], 1])
        else:
            evs = torch.tensor([])
            cfvs = torch.tensor([])
            for obs, _, actions, *_ in replay_buffer.iterate(512, random=False, curriculum=True):
                obs = torch.from_numpy(obs).to(self.device)
                actions = torch.from_numpy(
                    actions).unsqueeze(1).to(self.device)
                with torch.no_grad():
                    if self.future:
                        b_evs, b_cfvs = torch.split(self.network(
                            obs, actions), (1, self.network.action_dim), dim=1)
                    else:
                        b_evs, b_cfvs = torch.split(self.network(
                            obs), (1, self.network.action_dim), dim=1)
                b_cfvs = torch.gather(b_cfvs, 1, actions)
                b_evs, b_cfvs = b_evs.cpu(), b_cfvs.cpu()
                evs = torch.cat((evs, b_evs.cpu()))
                cfvs = torch.cat((cfvs, b_cfvs.cpu()))
            q_plus = torch.clamp(cfvs - evs, min=0)

        n_step = torch.from_numpy(
            replay_buffer.n_step[replay_buffer.curriculum_idcs]).unsqueeze(1)

        v_tar = n_step
        q_tar = q_plus + n_step

        return v_tar, q_tar

    def __sample_mini_batch(self, replay_buffer, v_tar, q_tar):

        mb_idcs = np.random.choice(
            replay_buffer.curriculum_idcs.shape[0], self.mini_batch_size)
        mb_obs, mb_next_obs, mb_actions = replay_buffer[replay_buffer.curriculum_idcs[mb_idcs]][:3]

        val_est_mb = torch.zeros(
            (self.mini_batch_size, 1)).to(self.device)

        mb_est_rew_w = torch.from_numpy(
            replay_buffer.est_rew_weights[replay_buffer.curriculum_idcs][mb_idcs])
        mb_est_non_zero = mb_est_rew_w.nonzero().squeeze()
        if mb_est_non_zero.numel():
            mb_est_rew_idcs = (mb_idcs[mb_est_non_zero] +
                               replay_buffer.n_step_size).reshape(-1)
            mb_v_prime_obs, _, mb_v_prime_actions = replay_buffer[mb_est_rew_idcs][:3]
            mb_v_prime_obs = mb_v_prime_obs
            mb_v_prime_actions = mb_v_prime_actions.astype(np.int64)
            mb_v_prime_obs = torch.from_numpy(
                mb_v_prime_obs).to(self.device)
            mb_v_prime_actions = torch.from_numpy(
                mb_v_prime_actions).to(self.device)

            with torch.no_grad():
                if self.future:
                    val_est = self.target_network(
                        mb_v_prime_obs, mb_v_prime_actions)[:, :1]
                else:
                    val_est = self.target_network(
                        mb_v_prime_obs)[:, :1]
            val_est = val_est * replay_buffer.gamma**replay_buffer.n_step_size
            val_est_mb.index_add_(0, mb_est_non_zero.to(self.device), val_est)

        mb_obs = torch.from_numpy(mb_obs).to(self.device)
        mb_actions = torch.from_numpy(mb_actions).to(
            self.device).unsqueeze(1)

        if self.future:
            mb_v, mb_q = torch.split(self.network(
                mb_obs, mb_actions), (1, self.network.action_dim), dim=1)
            mb_next_obs = torch.from_numpy(mb_next_obs).to(self.device)
            mb_pred_obs = self.network(mb_obs, mb_actions, future=True)
        else:
            mb_pred_obs, mb_next_obs = None, None
            mb_v, mb_q = torch.split(self.network(
                mb_obs), (1, self.network.action_dim), dim=1)
        mb_q = torch.gather(mb_q, dim=1, index=mb_actions)
        mb_v_tar = v_tar[mb_idcs].to(self.device) + val_est_mb
        mb_q_tar = q_tar[mb_idcs].to(self.device) + val_est_mb
        return mb_v, mb_v_tar, mb_q, mb_q_tar, mb_pred_obs, mb_next_obs

    def __reset_v_tar(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def __update_target(self):
        target_params = self.target_network.parameters()
        params = self.network.parameters()

        for target, net in zip(target_params, params):
            target.data.add_(self.tau * (net.data - target.data))

    def __update_network(self, mb_v, mb_v_tar, mb_q, mb_q_tar, mb_pred_obs, mb_next_obs):

        obs_loss = torch.tensor(0.0)
        if self.future:
            self.network.optimizer.zero_grad()
            obs_loss = self.network.criterion(mb_pred_obs, mb_next_obs)
            obs_loss.backward()
            self.network.optimizer.step()
            obs_loss = obs_loss.detach().cpu()

        self.network.optimizer.zero_grad()
        v_loss = self.network.criterion(mb_v, mb_v_tar)
        v_loss.backward(retain_graph=True)
        self.network.optimizer.step()
        v_loss = v_loss.detach().cpu()

        self.network.optimizer.zero_grad()
        q_loss = self.network.criterion(mb_q, mb_q_tar)
        q_loss.backward()
        self.network.optimizer.step()
        q_loss = q_loss.detach().cpu()

        return v_loss, q_loss, obs_loss

    def train_batch(self, replay_buffer, writer=None):
        """Trains the network with samples from the replay buffer
        using the arm algorithm. If a writer is passed, losses are
        recorded.

        Arguments:
            replay_buffer {ReplayBuffer} -- replay buffer of samples

        Keyword Arguments:
            writer {tf.summary.SummaryWriter} -- optional tensorflow
                                                 summary writer
                                                 (default: {None})
        """

        self.steps += len(replay_buffer)

        print('computing target values...')
        v_tar, q_tar = self.__compute_targets(replay_buffer)

        cum_v_loss = torch.tensor(0.0)
        cum_q_loss = torch.tensor(0.0)
        cum_obs_loss = torch.tensor(0.0)

        self.__reset_v_tar()

        print('training network...')

        for batch in range(self.iters):

            mb_v, mb_v_tar, mb_q, mb_q_tar, mb_pred_obs, mb_next_obs = self.sample_mini_batch(
                replay_buffer, v_tar, q_tar)

            v_loss, q_loss, obs_loss = self.update_network(
                mb_v, mb_v_tar, mb_q, mb_q_tar, mb_pred_obs, mb_next_obs)

            self.update_target()

            cum_v_loss += v_loss
            cum_q_loss += q_loss
            cum_obs_loss += obs_loss

            if writer is not None and TENSORFLOW:
                with writer.as_default():
                    tf.summary.scalar(
                        'v_loss', v_loss.item(), self.epochs * self.iters + batch)
                    tf.summary.scalar(
                        'q_loss', q_loss.item(), self.epochs * self.iters + batch)
                    if self.future:
                        tf.summary.scalar(
                            'obs_loss', obs_loss.item(), self.epochs * self.iters + batch)

            if (batch + 1) % int(self.iters / 10) == 0:
                mean_v_loss = (cum_v_loss/int(self.iters / 10)).numpy()
                mean_q_loss = (cum_q_loss/int(self.iters / 10)).numpy()
                mean_obs_loss = (cum_obs_loss/int(self.iters / 10)).numpy()
                if self.future:
                    print('batch: {}, v_loss: {}, q_loss: {}, obs_loss: {}'.format(
                        batch + 1, mean_v_loss, mean_q_loss, mean_obs_loss))
                else:
                    print('batch: {}, v_loss: {}, q_loss: {}'.format(
                        batch + 1, mean_v_loss, mean_q_loss))
                cum_v_loss.zero_()
                cum_q_loss.zero_()
                cum_obs_loss.zero_()

        self.epochs += 1
