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


