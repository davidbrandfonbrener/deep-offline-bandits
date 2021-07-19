import torch
import torch.nn.functional as F

from bandit_learner import BanditLearner
from utils.policy_utils import greedy_policy
from models import Q

class QLearning(BanditLearner):
    """docstring for QLearning."""

    def __init__(self, offset = 0.0,
                        **kwargs):

        super(QLearning, self).__init__(**kwargs)
        self.offset = offset

    def init_model(self):
        self.model = Q(self.context_dim, self.num_actions, model=self.model_type,
                        n_hidden=self.n_hidden, data_type=self.data_type)
        self.policy = greedy_policy(self.model)

    def objective(self, context, action, reward, **kwargs):

        Qval = self.model(context)
        pi = self.policy(context)

        indicator = F.one_hot(action, num_classes=self.num_actions)
        r_hat = Qval + indicator * ((reward + self.offset).unsqueeze(1) - Qval)

        loss = ((Qval - r_hat)**2).sum(dim=1)

        info = {}
        return torch.mean(loss), pi, info
