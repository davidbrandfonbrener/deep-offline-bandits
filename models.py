import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock


class Q(nn.Module):
    def __init__(self, input_dim, num_actions,
                    n_hidden = 512, model='mlp', data_type=None):
        super().__init__()
        if model == 'resnet':
            self.Q = QResnet(data_type)
        elif model == 'linear':
            self.Q = QLinear(input_dim, num_actions)
        elif model == 'mlp':
            self.Q = MLP(input_dim, num_actions, n_hidden)

    def forward(self, x):
        x = self.Q(x)
        return x

class Policy(nn.Module):
    def __init__(self, input_dim, num_actions,
                    n_hidden = 512, model='mlp', data_type=None):
        super().__init__()
        self.Q = Q(input_dim, num_actions,
                    n_hidden=n_hidden, model=model, data_type=data_type)

    def forward(self, x):
        x = self.Q(x)
        x = F.log_softmax(x, dim=1)
        return x



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.l1 = nn.Linear(input_dim, n_hidden)
        self.l2 = nn.Linear(n_hidden, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1, -1)

        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)

        return x


class QResnet(ResNet):
    def __init__(self, data_type):
        if data_type == 'mnist':
            super(QResnet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3,
                                        stride=1, padding=1, bias=False)
        elif data_type == 'cifar10':
            super(QResnet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class QLinear(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.l1 = nn.Linear(input_dim, num_actions)

    def forward(self, x):
        x = torch.flatten(x, 1, -1)
        x = self.l1(x)
        return x



class V(nn.Module):
    def __init__(self, input_dim,
                        n_hidden = 512, model='mlp', data_type=None):
        super().__init__()
        if model == 'mlp':
            self.V = MLP(input_dim, 1, n_hidden)

    def forward(self, x):
        x = self.V(x)
        return x
