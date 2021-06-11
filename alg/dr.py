import torch
import torch.nn.functional as F

from meta_algorithm import MetaAlgorithm
from models import Policy

from baseline import load_baseline

class DR(MetaAlgorithm):
    """docstring for DR."""

    def __init__(self,
                    clip = None,
                    baseline = None,
                    **kwargs):
        super(DR, self).__init__(**kwargs)

        self.clip = clip
        self.baseline_Q = load_baseline(**baseline)

    def init_model(self):
        self.model = Policy(self.context_dim, self.num_actions, model=self.model_type,
                        n_hidden=self.n_hidden, data_type=self.data_type)
        self.policy = self.model

    def objective(self, context, action, reward, propensity, label, split, **kwargs):
        log_pi = self.policy(context)
        pi = log_pi.exp()

        if self.clip is not None:
            propensity = torch.max(self.clip * torch.ones_like(propensity),
                                    propensity)
        indicator = F.one_hot(action, num_classes=self.num_actions)
        ip = torch.ones_like(propensity) / propensity

        Qval = self.baseline_Q(context, None, None, split)
        iw_term = torch.sum(indicator * ip * reward.unsqueeze(1) - indicator * Qval, dim=-1)
        dm_term = torch.sum(log_pi.exp() * Qval, dim = -1)

        loss = - (iw_term + dm_term)

        info = {}
        return torch.mean(loss), pi, info
