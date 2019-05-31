import time

import gym
import numpy as np
import torch
import torchvision
from arm.arm import Arm
from arm.buffer import ReplayBuffer
from arm.policy import Policy

try:
    import tensorflow as tf
    TENSORFLOW = True
except ImportError:
    TENSORFLOW = False


ARM_ITERS = 3000
CURRICULUM = ()
EPOCHS = 60
FUTURE = False
FRAME_BUFFER = 4
GAMMA = 0.99
GPU = True
LEARN_RATE = 1e-4
MINI_BATCH_SIZE = 32
N_STEP_SIZE = 1
REP_BUFFER_SIZE = 1
TAU = 0.01
TENSORBOARD_PATH = ''
# TENSORBOARD_PATH = '/runs/'


def run_env(env: gym.Env, policy: Policy):
    replay_buffer = ReplayBuffer()
    obs = env.reset()
    done = False
    # convert to grayscale, scale to 84x84 and scale values between 0 and 1
    pre_torchvision = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                      torchvision.transforms.Grayscale(),
                                                      torchvision.transforms.Resize(
                                                          (84, 84)),
                                                      torchvision.transforms.ToTensor()])
    # remove channels and convert to numpy
    def preprocess(img): return pre_torchvision(img)[0].numpy()
    obs = preprocess(obs)
    obs_arr = [obs] * FRAME_BUFFER
    while not done:
        torch_obs = torch.tensor(obs_arr).unsqueeze(0).to(policy.device)
        action = policy(torch_obs, env.action_space.n)
        total_reward = 0
        # only record every 4th frame
        for _ in range(4):
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        next_obs = preprocess(next_obs)
        replay_buffer.append(obs, next_obs, action, total_reward, done)
        obs = next_obs
        obs_arr = obs_arr[1:] + [obs]
    return replay_buffer


def collect_rep_buffer(env: gym.Env, policy: Policy):
    replay_buffer = ReplayBuffer()

    start = time.time()
    # accumulate replay buffer
    while len(replay_buffer) < REP_BUFFER_SIZE:
        replay_buffer += run_env(env, policy)
        print('collected {} steps, time elapsed: {}'.format(
            len(replay_buffer), time.time()-start))

    return replay_buffer


def evaluate(arm: Arm, replay_buffer: ReplayBuffer, writer=None):
    done_idcs = np.nonzero(replay_buffer.done)[0] + 1
    traj_rewards = np.split(replay_buffer.rewards, done_idcs)[:-1]
    traj_rewards = np.array([np.sum(trajectory)
                             for trajectory in traj_rewards])
    mean_epi_steps = len(replay_buffer) / done_idcs.shape[0]
    print('mean episode steps: {}, avg reward: {}, min reward: {}, max reward: {}'.format(
        mean_epi_steps, traj_rewards.mean(), traj_rewards.min(), traj_rewards.max()))
    if writer is not None:
        with writer.as_default():
            tf.summary.scalar('reward', traj_rewards.mean(), arm.epochs)


class Network(torch.nn.Module):

    def __init__(self, frame_buffer, action_dim, lr, device=torch.device('cpu')):
        super(Network, self).__init__()

        self.action_dim = action_dim
        self.device = device

        self.conv1 = torch.nn.Conv2d(frame_buffer, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 8, 4)
        self.conv3 = torch.nn.Conv2d(16, 32, 4, 2)
        self.fc1 = torch.nn.Linear(32*9*9, 256)
        self.fc2 = torch.nn.Linear(256, action_dim + 1)

        self.to(device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr)

    def forward(self, obs):
        out = torch.nn.functional.relu(self.conv1(obs))
        out = torch.nn.functional.relu(self.conv2(out))
        out = torch.nn.functional.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def train_arm():
    env = gym.make('PongNoFrameskip-v4')

    writer = None
    if TENSORFLOW and TENSORBOARD_PATH:
        writer = tf.summary.create_file_writer(TENSORBOARD_PATH)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    network = Network(FRAME_BUFFER, env.action_space.n,
                      LEARN_RATE, device=device)

    arm = Arm(network, ARM_ITERS, MINI_BATCH_SIZE, TAU)
    policy = Policy(network)

    while arm.epochs < EPOCHS:

        replay_buffer = collect_rep_buffer(env, policy)

        evaluate(arm, replay_buffer, writer=writer)

        replay_buffer = replay_buffer.vectorize(frame_buffer=FRAME_BUFFER,
                                                curriculum=CURRICULUM,
                                                n_step_size=N_STEP_SIZE,
                                                gamma=GAMMA)
        arm.train_batch(replay_buffer, writer=writer)

    return arm.network


if __name__ == "__main__":
    train_arm()
