"""Arm policy, computes action based on cumulative clipped advantage values
"""
import torch


class Policy():
    """Arm policy - initialized with arbitrary network
    that maps observations to vector of size one greater than
    the size of the action space.

    If started in debug mode, debug logs of action values
    and percentages are printed.

    Arguments:
        network {torch.nn.Module} -- arbitrary pytorch network

    Keyword Arguments:
        debug {bool} -- toggle to enable debug logs (default: {False})
    """

    def __init__(self, network, debug=False):
        self.network = network
        self.debug = debug

        self.device = network.device

    def __call__(self, obs):
        return self.forward(obs)

    def forward(self, obs):
        """Computes action from observations and action space
        dimensionality

        Arguments:
            obs {torch.Tensor} -- tensor of observations

        Returns:
            int -- action
        """
        self.network.eval()
        with torch.no_grad():
            values = self.network(obs)
        self.network.train()
        expected_values = values[:, 0]
        cf_values = values[:, 1:]
        action_values = torch.clamp(cf_values - expected_values, min=0.0)
        if torch.sum(action_values):
            action_probs = action_values / torch.sum(action_values)
        else:
            action_dim = action_values.shape[-1]
            action_probs = torch.full([action_dim], 1/action_dim)
        action = int(torch.multinomial(action_probs, 1))
        if self.debug:
            print('q_plus: ', cf_values.cpu().numpy().round(4))
            print('v: ', expected_values.cpu().numpy().round(4))
            print('action values: ', action_values.cpu().numpy().round(4))
            print('action probs: ', action_probs.cpu().numpy().round(4))
            print('action: ', action)
        return action
