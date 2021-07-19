import torch
import torch.nn.functional as F

from bandit_learner import BanditLearner
from models import Policy

from baseline import load_baseline

class IW(BanditLearner):
    """docstring for IW."""

    def __init__(self, clip,
                    clip_type= "strehl",
                    offset = 0.0,
                    use_baseline = False,
                    baseline = None,
                    **kwargs
                ):
        super(IW, self).__init__(**kwargs)

        self.clip = clip
        self.clip_type = clip_type
        self.offset = offset

        self.use_baseline = use_baseline
        if use_baseline:
            self.baseline = load_baseline(**baseline)

    def init_model(self):
        self.model = Policy(self.context_dim, self.num_actions, model=self.model_type,
                        n_hidden=self.n_hidden, data_type=self.data_type)
        self.policy = self.model

    def _clip(self, propensity):
        if self.clip is None:
            weight = torch.ones_like(propensity) / propensity
        elif self.clip_type == "strehl":
            propensity = torch.max(self.clip * torch.ones_like(propensity),
                                    propensity)
            weight = torch.ones_like(propensity) / propensity
        elif self.clip_type == "bottou":
            weight = torch.ones_like(propensity) / propensity
            weight[weight > self.clip] = 0
        elif self.clip_type == "poem":
            weight = torch.ones_like(propensity) / propensity
            weight = torch.min(self.clip * torch.ones_like(weight), weight)
        elif self.clip_type == "no_prop":
            weight = torch.ones_like(propensity)
        return weight

    def objective(self, context, action, reward, propensity, label, split):

        weight = self._clip(propensity)
        log_pi = self.policy(context)
        pi = log_pi.exp()

        if self.use_baseline:
            with torch.no_grad():
                b = self.baseline(context, log_pi, propensity, split)
        else:
            b = 0.0


        indicator = F.one_hot(action, num_classes=self.num_actions)
        r_hat = indicator * weight * (reward - b + self.offset).unsqueeze(1)
        loss = - torch.sum(pi * r_hat, dim = -1)

        info = {}
        return torch.mean(loss), pi, info
