import torch
import torch.nn.functional as F

from bandit_learner import BanditLearner
from models import Policy

class BC(MetaAlgorithm):
    """docstring for BC."""

    def __init__(self, **kwargs):
        super(BC, self).__init__(**kwargs)

    def init_model(self):
        self.model = Policy(self.context_dim, self.num_actions, model=self.model_type,
                        n_hidden=self.n_hidden, data_type=self.data_type)
        self.policy = self.model

    def objective(self, context, action, **kwargs):
        log_pi = self.policy(context)
        pi = log_pi.exp()
        loss = F.nll_loss(log_pi, action)

        return torch.mean(loss), pi, {}
