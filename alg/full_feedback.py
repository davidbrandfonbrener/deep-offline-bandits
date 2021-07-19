import torch
import torch.nn.functional as F

from bandit_learner import BanditLearner
from models import Policy

class FullFeedback(BanditLearner):
    """docstring for FullFeedback."""

    def __init__(self, **kwargs):
        super(FullFeedback, self).__init__(**kwargs)
        self.use_full_reward = True

    def init_model(self):
        self.model = Policy(self.context_dim, self.num_actions, model=self.model_type,
                        n_hidden=self.n_hidden, data_type=self.data_type)
        self.policy = self.model

    def objective(self, context, reward, **kwargs):
        log_pi = self.policy(context)
        pi = log_pi.exp()
        loss = -(pi * reward).sum(dim=1)
        return torch.mean(loss), pi, {}
