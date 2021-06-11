import numpy as np
import torch
import argparse
import os

from data import BanditDataset, split_dataset

def gen_data(seed, action_seed, N, d, K, gamma, epsilon, nbad):
    
    # sample contexts from an isotropic gaussian
    np.random.seed(seed)
    x = np.random.randn(N, d)

    # set K to number of linear actions
    assert K > nbad
    K = K - nbad

    # rewards are a random linear function of context with gaussian noise 
    r_theta = np.random.random((d,K))
    
    r = np.matmul(x, r_theta)
    l = np.argmax(r, axis=1)

    # epsilon controls the noise level
    # only add noise after defining l 
    r = r + epsilon * np.random.randn(N, K)

    # sample actions from mixture between uniform and optimal
    # gamma determines probability of sampling from pi^*
    np.random.seed(action_seed)
    sample_optimal = (np.random.random(N) < gamma).astype(int)
    uniform_actions = np.random.choice(K, size=N)
    a = (1 - sample_optimal) * uniform_actions + sample_optimal * l

    one_hot_l = np.zeros((N, K))
    one_hot_l[np.arange(N), l] = 1
    p = gamma * one_hot_l + (1 - gamma) * np.ones((N, K)) / K
    assert np.sum(np.abs(np.sum(p, axis=1) - np.ones(N))) < 1e-8

    # add bad actions to r, p
    if nbad > 0:
        bad_r = -10.0 * np.ones((N, nbad))
        r = np.concatenate((r, bad_r), axis = 1)
        p = np.concatenate((p, np.zeros((N, nbad))), axis=1)

    x = torch.tensor(x, dtype=torch.float)
    a = torch.tensor(a, dtype=torch.int64)
    r = torch.tensor(r, dtype=torch.float)
    p = torch.tensor(p, dtype=torch.float)
    l = torch.tensor(l, dtype=torch.int64)

    return x,a,r,p,l


def gen_data_splits(seed, action_seed, N, d, K, gamma, epsilon, nbad):
    
    path = '../data/synth/n=' + str(N) + \
                '/d=' + str(d) + \
                '/K=' + str(K) + \
                '/nbad=' + str(nbad) + \
                '/gamma=' + str(gamma) + \
                '/epsilon=' + str(epsilon) + \
                '/seed=' + str(seed) + \
                '/action_seed=' + str(action_seed)
    if not os.path.exists(path):
        os.makedirs(path)
            
    x,a,r,p,l = gen_data(seed, action_seed, 7 * N, d, K, gamma, epsilon, nbad)
    
    train = BanditDataset(x[:N], a[:N], r[:N], p[:N], l[:N], np.zeros(N))
    test = BanditDataset(x[N:-N], a[N:-N], r[N:-N], p[N:-N], l[N:-N], np.zeros(5*N))
    val = BanditDataset(x[-N:], a[-N:], r[-N:], p[-N:], l[-N:], np.zeros(N))

    train, split_0, split_1 = split_dataset(train)

    train.save(path + '/train.pt')
    split_0.save(path + '/train_0.pt')
    split_1.save(path + '/train_1.pt')
    test.save(path + '/test.pt')
    val.save(path + '/val.pt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--action_seed", default=0, type=int)
    parser.add_argument("--N", default=1000, type=int)
    parser.add_argument("--d", default=10, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--gamma", default=0.0, type=float)
    parser.add_argument("--epsilon", default=0.0, type=float)
    parser.add_argument("--nbad", default=0, type=int)
    args = parser.parse_args()

    gen_data_splits(args.seed, args.action_seed,
                    args.N, args.d, 
                    args.K, args.gamma, args.epsilon,
                    args.nbad)
