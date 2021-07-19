import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

from models import Q, Policy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_list(tensor):
    return list(tensor.flatten().cpu().detach().numpy())

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class BanditLearner():

    def __init__(self, context_dim, num_actions,
                    model_type,
                    n_hidden,
                    optim_type='sgd',
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=0.0,
                    use_lr_sched=True,
                    data_type = None,
                    train_Q = False,
                    estimated_behavior = None,
                    use_full_reward=False,
                    **kwargs):

        self.context_dim = context_dim
        self.num_actions = num_actions
        self.estimated_behavior = estimated_behavior
        self.train_Q = train_Q
        self.use_full_reward = use_full_reward
        self.model_type = model_type
        self.n_hidden = n_hidden
        self.data_type = data_type

        # initialize model
        self.init_model()
        self.model.to(DEVICE)

        # initialize optimizers
        if optim_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
        elif optim_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                        weight_decay=weight_decay)

        self.lr_sched = [lr / 10.0] + [lr] * 199 + [lr / 10.0] * 200 + [lr / 100.0] * 600

    def objective(self, Q, log_pi, context, action, reward, propensity, label):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def train(self, data_loader, epoch, print_idx=100, tau = 1):
        self.optimizer = set_lr(self.optimizer, self.lr_sched[epoch-1])
        self.model.train()

        for batch_idx, (context, action, full_reward,
                        propensity, label, split) in enumerate(data_loader):
            # get data
            context = context.to(DEVICE)
            action = action.to(DEVICE)
            full_reward = full_reward.to(DEVICE)
            propensity = propensity.to(DEVICE)
            label =   label.to(DEVICE)
            split = split.to(DEVICE)

            if self.use_full_reward:
                reward = full_reward
            else:
                reward = torch.gather(full_reward, dim=1, index=action.unsqueeze(1)).squeeze(1)

            if self.estimated_behavior is not None:
                propensity = self.estimated_behavior(context)

            # calculate loss and policy
            loss, pi, info = self.objective(context=context, action=action,
                                        reward=reward, propensity=propensity,
                                        label=label, split=split)

            # take step with clipped grad
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()

            if batch_idx % print_idx == 0 and batch_idx > 0:
                reward = torch.sum(1.0 * pi * full_reward, dim=1).mean()
                pred = pi.argmax(dim=1, keepdim=True)
                accuracy = (pred.eq(label.view_as(pred)).sum() / (1.0 * len(context)))

                print('Train batch: {} \t Loss: {:.6f}\t Reward: {:.6f}\t Accuracy: {:.6f}'.format(
                    batch_idx, loss.item(), reward.item(), accuracy.item()))

        return

    def eval(self, data_loader, epoch, store=None, mode="test"):
        self.model.eval()

        with torch.no_grad():
            l, r, correct, imitate, data_kept = [],[],[], [], []
            for batch_idx, (context, action, full_reward,
                            propensity, label, split) in enumerate(data_loader):
                # get data
                context = context.to(DEVICE)
                action = action.to(DEVICE)
                full_reward = full_reward.to(DEVICE)
                propensity = propensity.to(DEVICE)
                label =   label.to(DEVICE)
                split = split.to(DEVICE)

                if self.use_full_reward:
                    reward = full_reward
                else:
                    reward = torch.gather(full_reward, dim=1, index=action.unsqueeze(1)).squeeze(1)

                if self.estimated_behavior is not None:
                    propensity = self.estimated_behavior(context)

                # calculate loss
                loss, pi, info = self.objective(context=context, action=action,
                                        reward=reward, propensity=propensity,
                                        label=label, split=split)

                # log
                l.extend(make_list(loss))
                r.extend(make_list(torch.sum(pi * full_reward, dim=1)))
                pred = pi.argmax(dim=1, keepdim=True)
                correct.extend(make_list(pred.eq(label.view_as(pred))))
                im = torch.gather(pi, dim=1, index=action.unsqueeze(1)).squeeze(1)
                imitate.extend(make_list(im))

            # reduce log
            l_mean = np.mean(l)
            r_mean = np.mean(r)
            c_mean = np.mean(correct)
            im_mean = np.mean(imitate)


        if store is not None:
            store['learning_curves'].append_row({
                        'epoch': epoch,
                        'mode': mode,
                        'loss': l_mean,
                        'reward': r_mean,
                        'pct_optimal': c_mean,
                        'pct_imitate': im_mean,
                    })

        print('{}\t Loss: {:.3f},\t'\
                    'Reward: {:.3f},\t'\
                    'Pct Optimal: {:.3f},\t'\
                    'Pct Imitate: {:.3f},\t'.format(
                mode, l_mean, r_mean, c_mean, im_mean))
        return l_mean, r_mean, c_mean

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        return
