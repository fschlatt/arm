"""Arm policy, computes action based on cumulative clipped advantage values
"""
import torch


class Policy():
    """Arm policy - initialized with arbitrary network
    that maps observations to vector of size one greater than
    the size of the action space. If one step future prediction
    is used, the network needs to implement a future toggle in
    the forward function and return an output whose dimensions
    are equal to the dimensions of a single observation.

    If started in debug mode, debug logs of action values
    and percentages are printed.

    Arguments:
        network {torch.nn.Module} -- arbitrary pytorch network

    Keyword Arguments:
        future {bool} -- toggle to use one step future prediction (default: {False})
        debug {bool} -- toggle to enable debug logs (default: {False})
    """

    def __init__(self, network, future=False, debug=False):
        self.network = network
        self.future = future
        self.debug = debug

        self.device = network.device

    def __call__(self, obs, action_dim):
        return self.forward(obs, action_dim)

    def forward(self, obs, action_dim):
        """Computes action from observations and action space
        dimensionality

        Arguments:
            obs {torch.Tensor} -- tensor of observations
            action_dim {int} -- dimensionality of actions space

        Returns:
            int -- action
        """
        if self.future:
            action_input = torch.arange(action_dim).unsqueeze(1)
            obs = obs.expand(action_dim, *obs.shape[1:])
            with torch.no_grad():
                value = self.network(obs, action_input)
            expected_value = value[:, 0].mean()
            cf_values = value[:, 1:][torch.arange(
                action_dim), torch.arange(action_dim)]
        else:
            value = self.network(obs)
            expected_value = value[:, 0]
            cf_values = value[:, 1:]
        action_values = torch.clamp(cf_values - expected_value, min=0.0)
        if torch.sum(action_values):
            action_probs = action_values / torch.sum(action_values)
        else:
            action_probs = torch.full([action_dim], 1/action_dim)
        action = int(torch.multinomial(action_probs, 1))
        if self.debug:
            print('q_plus: ', cf_values)
            print('v: ', expected_value)
            print('action values: ', action_values)
            print('action probs: ', action_probs)
            print('action: ', action)
        return action
