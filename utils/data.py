import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.dataset_utils import DatasetCache, DatasetSubset, TransformDataset

class BanditDataset(Dataset):
    def __init__(self, contexts, actions,
                        rewards, propensities,
                        labels, splits):
        super(BanditDataset, self).__init__
        self.contexts = contexts
        self.actions = actions
        self.rewards = rewards
        self.propensities = propensities
        self.labels = labels
        self.splits = splits

    def __getitem__(self, index):
        return (self.contexts[index],self.actions[index],
                self.rewards[index], self.propensities[index],
                self.labels[index], self.splits[index])

    def __len__(self):
        return self.contexts.shape[0]

    def save(self, path):
        with open(path, 'wb') as f:
            torch.save((self.contexts, self.actions,
                        self.rewards, self.propensities,
                        self.labels, self.splits), f)

    def load(self, path):
        c,a,r,p,l,s = torch.load(path)
        self.contexts = c
        self.actions = a
        self.rewards = r
        self.propensities = p
        self.labels = l
        self.splits = s


def get_bandit_loaders(root_path, subpath,
                        data_type, batch_size,
                        train_on_partition, partition,
                        **kwargs):
    if data_type == 'cifar10':
        path = root_path + '/data'
        class_train = CIFAR10(path, train=True, transform=None, download=True)
        class_test = CIFAR10(path, train=False, transform=None)

        action_path = root_path + '/action_data/cifar10/' + kwargs['action_dir']
        bandit_train = ClassToBandit(class_train, action_path, train=True)
        train_size = 45000
        train_set = DatasetSubset(bandit_train, start=0, stop=train_size)
        val_set = DatasetSubset(bandit_train, start=train_size, stop=None)
        test_set = ClassToBandit(class_test, action_path, train=False)

        transform_train = transforms.Compose([ transforms.Pad(4, padding_mode='reflect'),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        train_set = TransformDataset(DatasetCache(train_set), transform_train)
        val_set = TransformDataset(DatasetCache(val_set), transform_test)
        test_set = TransformDataset(DatasetCache(test_set), transform_test)

    else:
        path = root_path + '/' + subpath
        train_set = BanditDataset(None, None, None, None, None, None)
        val_set = BanditDataset(None, None, None, None, None, None)
        test_set = BanditDataset(None, None, None, None, None, None)

        if train_on_partition:
            train_set.load(path + '/train_' + str(partition) + '.pt')
        else:
            train_set.load(path + '/train.pt')
        test_set.load(path + '/test.pt')
        val_set.load(path + '/val.pt')

    loader_args = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader( train_set,
        batch_size=batch_size, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_set,
        batch_size=batch_size, shuffle=True, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=batch_size, shuffle=True, **loader_args)

    return train_loader, val_loader, test_loader


class ClassToBandit(Dataset):
    def __init__(self, class_dataset, action_dir, train=True):

        self.class_dataset = class_dataset
        self.train = train

        if self.train:
            self.actions, self.rewards, self.propensities = torch.load(action_dir + '/train.pt')
        else:
            self.actions, self.rewards, self.propensities = torch.load(action_dir + '/test.pt')

    def __getitem__(self, index):
        context, label = self.class_dataset.__getitem__(index)
        action = int(self.actions[index])
        propensity = self.propensities[index]
        reward = self.rewards[index]
        return context, action, reward, propensity, label, 0

    def __len__(self):
        return self.class_dataset.__len__()


def get_data_dim(data_type):
    if data_type == 'mnist':
        context_dim = 784
        num_actions = 10
    elif data_type == 'cifar10':
        context_dim = 3072
        num_actions = 10
    elif data_type == 'toy':
        context_dim = 5
        num_actions = 2
    elif data_type == 'world3':
        context_dim = 12
        num_actions = 2

    return context_dim, num_actions

def split_dataset(data):
    new_data = data
    split_size = int(new_data.__len__() / 2)
    split_0 = BanditDataset(np.zeros_like(data.contexts[:split_size]),
                            np.zeros_like(data.actions[:split_size]),
                            np.zeros_like(data.rewards[:split_size]),
                            np.zeros_like(data.propensities[:split_size]),
                            np.zeros_like(data.labels[:split_size]),
                            np.zeros_like(data.splits[:split_size]))
    split_1 = BanditDataset(np.zeros_like(data.contexts[:split_size]),
                            np.zeros_like(data.actions[:split_size]),
                            np.zeros_like(data.rewards[:split_size]),
                            np.zeros_like(data.propensities[:split_size]),
                            np.zeros_like(data.labels[:split_size]),
                            np.zeros_like(data.splits[:split_size]))
    idx_0 = 0
    idx_1 = 0

    idxs = np.random.choice(new_data.__len__(), size=split_size, replace=False)
    for i in range(new_data.__len__()):
        if i in idxs:
            new_data.splits[i] = 0
            split_0.contexts[idx_0] = new_data.contexts[i]
            split_0.actions[idx_0] = new_data.actions[i]
            split_0.rewards[idx_0] = new_data.rewards[i]
            split_0.propensities[idx_0] = new_data.propensities[i]
            split_0.labels[idx_0] = new_data.labels[i]
            split_0.splits[idx_0] = new_data.splits[i]
            idx_0 += 1
        else:
            new_data.splits[i] = 1
            split_1.contexts[idx_1] = new_data.contexts[i]
            split_1.actions[idx_1] = new_data.actions[i]
            split_1.rewards[idx_1] = new_data.rewards[i]
            split_1.propensities[idx_1] = new_data.propensities[i]
            split_1.labels[idx_1] = new_data.labels[i]
            split_1.splits[idx_1] = new_data.splits[i]
            idx_1 += 1

    assert idx_0 == split_size
    assert idx_1 == split_size
    assert np.sum(split_0.contexts - split_1.contexts) != 0

    return new_data, split_0, split_1




def make_mnist_dataset(data_path, action_path):

    return




# def get_bandit_loaders(path, data_type, action_type, action_seed, toy_data_params,
#                         batch_size, train_on_partition, partition, **kwargs):

#     if data_type == 'mnist' or data_type == 'cifar10':
#         action_dir = 'action_data/' + data_type + '/' + \
#                             action_type + '_' + str(action_seed)

#         if data_type == 'mnist':
#             train_size = 55000
#             transform_train = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.1307,), (0.3081,))])

#             transform_test = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.1307,), (0.3081,))])

#             class_train = MNIST(path, train=True, transform=None)
#             class_test = MNIST(path, train=False, transform=None)

#             raw_train = ClassToBandit(class_train, action_dir, train=True)
#             raw_test = ClassToBandit(class_test, action_dir, train=False)

#         elif data_type == 'cifar10':
#             train_size = 49000
#             transform_train = transforms.Compose([ transforms.Pad(4, padding_mode='reflect'),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.RandomCrop(32),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

#             transform_test = transforms.Compose([transforms.ToTensor(),
#                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

#             class_train = CIFAR10(path, train=True, transform=None)
#             class_test = CIFAR10(path, train=False, transform=None)

#             raw_train = ClassToBandit(class_train, action_dir, train=True)
#             raw_test = ClassToBandit(class_test, action_dir, train=False)

#         # split val
#         train_set = DatasetSubset(raw_train, start=0, stop=train_size)
#         val_set = DatasetSubset(raw_train, start=train_size, stop=None)

#         # cache and transform
#         train_set = TransformDataset(DatasetCache(train_set), transform_train)
#         val_set = TransformDataset(DatasetCache(val_set), transform_test)
#         test_set = TransformDataset(DatasetCache(raw_test), transform_test)

#     elif data_type == 'toy' or data_type == 'world3':
#         if data_type == 'toy':
#             data_path ='toy_data/sine_' + str(toy_data_params.size) +\
#                         '_' + str(toy_data_params.p1) + '_' + str(toy_data_params.p2) +\
#                         '_' + str(toy_data_params.p3) +\
#                         '_' + str(toy_data_params.seed)
#         elif data_type == 'world3':
#             data_path ='whynot'

#         train_set = BanditDataset(data_path + '/train.pt')
#         test_test = BanditDataset(data_path + '/test.pt')
#         val_set = BanditDataset(data_path + '/val.pt')

#     else:
#         assert False, f'wrong data type: {data_type}'


#     kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}

#     train_loader = torch.utils.data.DataLoader( train_set,
#         batch_size=batch_size, shuffle=True, **kwargs)
#     val_loader = torch.utils.data.DataLoader(val_set,
#         batch_size=batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(test_set,
#         batch_size=batch_size, shuffle=True, **kwargs)

#     return train_loader, val_loader, test_loader










# def buffer_split(train_buffer_path):
#     buffer = torch.load(train_buffer_path)
#     split_size = int(buffer.capacity / 2)
#     split0 = ReplayBuffer(buffer.obs_shape,
#                           buffer.action_shape,
#                           buffer.num_actions,
#                           split_size,
#                           buffer.full_feedback)
#     split1 = ReplayBuffer(buffer.obs_shape,
#                           buffer.action_shape,
#                           buffer.num_actions,
#                           split_size,
#                           buffer.full_feedback)

#     idxs = np.random.randint(0, buffer.capacity, size=split_size)
#     for i in range(buffer.capacity):
#         if i in idxs:
#             buffer.data_splits[i] = 0
#             split0.add(buffer.obses[i],
#                         buffer.actions[i],
#                         buffer.rewards[i],
#                         buffer.next_obses[i],
#                         buffer.propensities[i],
#                         buffer.dones[i],
#                         buffer.data_splits[i],
#                         buffer.labels[i])
#         else:
#             buffer.data_splits[i] = 1
#             split1.add(buffer.obses[i],
#                         buffer.actions[i],
#                         buffer.rewards[i],
#                         buffer.next_obses[i],
#                         buffer.propensities[i],
#                         buffer.dones[i],
#                         buffer.data_splits[i],
#                         buffer.labels[i])

#     with open(train_buffer_path, 'wb') as f:
#         torch.save(buffer, f)

#     path = train_buffer_path[:-3]
#     with open(path + '_0.pt', 'wb') as f:
#         torch.save(split0, f)
#     with open(path + '_1.pt', 'wb') as f:
#         torch.save(split1, f)

#     return


# def make_mnist_buffers(data_path, action_path, save_path):

#     return


# def class_and_action_to_buffer(class_data, action_data):

#     return buffer


# def bandit_to_buffer_ff(x, a, r, p, l):
#     obs_shape = x[0].shape
#     num_actions = torch.max(a) + 1
#     capacity = x.shape[0]

#     buffer = ReplayBuffer(obs_shape, (1,), num_actions, capacity,
#                             full_feedback=True)

#     for i in range(capacity):
#         buffer.add(x[i], a[i], r[i], np.zeros_like(x[i]),
#                      p[i], True, 0, l[i])

#     return buffer
