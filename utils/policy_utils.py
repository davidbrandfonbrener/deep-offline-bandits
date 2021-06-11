import torch
import torch.nn.functional as F


def greedy_policy(Q):
    def policy(context):
        Q_output = Q(context)
        x = torch.argmax(Q_output, dim=1)
        x = F.one_hot(x, num_classes=Q_output.shape[-1])
        return x
    return policy

def softmax_policy(Q):
    def policy(context):
        Q_output = Q(context)
        x = F.softmax( Q_output, dim=1)
        return x
    return policy

def null_Q(context):
    return None
