import torch

from .arm import Arm


class Policy():

    def __init__(self, network, future=False, debug=False):
        if not hasattr(network, 'forward'):
            raise AttributeError('network is not valid pytorch module')
        self.network = network
        self.future = future
        self.debug = debug

        self.device = network.device

    def update(self, network_state_dict):
        self.network.load_state_dict(network_state_dict)

    @classmethod
    def load(cls, path, debug=False, device=torch.device('cpu')):

        model = torch.load(path, map_location=device)

        if isinstance(model, Arm):
            network = model.network
        else:
            network = model

        network.eval()

        policy = cls(network, debug=debug)

        return policy

    def forward(self, obs, action_dim):
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

    def __call__(self, obs, action_dim):
        return self.forward(obs, action_dim)
