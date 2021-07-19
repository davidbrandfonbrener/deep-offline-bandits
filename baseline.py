import torch
from torch import nn
import torch.nn.functional as F

from models import Q
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Baseline(nn.Module):
    def __init__(self, 
                    context_dim, num_actions, 
                    arch='mlp', 
                    n_hidden = 512, 
                    data_type=None, 
                    **kwargs):
        super().__init__()
        self.Q1 = Q(context_dim, num_actions, n_hidden, 
                    arch, data_type)
        self.Q2 = Q(context_dim, num_actions, n_hidden, 
                    arch, data_type)

    def forward(self, context, log_pi, propensity, split):
        raise NotImplementedError

    def load(self, path1, path2):
        self.Q1.load_state_dict(torch.load(path1, map_location=DEVICE))
        self.Q2.load_state_dict(torch.load(path2, map_location=DEVICE))


def load_baseline(type, path1, path2, **kwargs):
    if type == 'vpi':
        baseline = VPiBaseline(**kwargs)
    elif type == 'vbeta':
        baseline = VBetaBaseline(**kwargs)
    elif type == 'q':
        baseline = QBaseline(**kwargs)

    baseline.load(path1, path2)

    return baseline

class VBaseline(Baseline):
    """docstring for VBaseline."""

    def __init__(self,  **kwargs):
        kwargs['num_actions'] = 1
        super().__init__(**kwargs)

    def forward(self, context, log_pi, propensity, split):

        use_b1 = 1 - split

        b1 = self.Q1(context)
        b2 = self.Q2(context)

        b = use_b1 * b1 + (1 - use_b1) * b2

        return b

class VPiBaseline(Baseline):
    """docstring for ValueBaseline."""

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def forward(self, context, log_pi, propensity, split):

        use_b1 = 1 - split

        b1 = torch.sum(log_pi.exp() * self.Q1(context), dim = -1)
        b2 = torch.sum(log_pi.exp() * self.Q2(context), dim = -1)

        b = use_b1 * b1 + (1 - use_b1) * b2

        return b

class VBetaBaseline(Baseline):
    """docstring for ValueBaseline."""

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def forward(self, context, log_pi, propensity, split):

        use_b1 = 1 - split

        b1 = torch.sum(propensity * self.Q1(context), dim = -1)
        b2 = torch.sum(propensity * self.Q2(context), dim = -1)

        b = use_b1 * b1 + (1 - use_b1) * b2

        return b

class QBaseline(Baseline):
    """docstring for ValueBaseline."""

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def forward(self, context, log_pi, propensity, split):

        use_b1 = (1 - split).unsqueeze(1)

        b1 = self.Q1(context)
        b2 = self.Q2(context)

        b = use_b1 * b1 + (1 - use_b1) * b2

        return b