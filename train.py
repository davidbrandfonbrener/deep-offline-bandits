import hydra
import os
import shutil
import numpy as np
from cox.store import Store
import torch

from utils.data import get_bandit_loaders, get_data_dim

@hydra.main(config_path='config', config_name='train')
def train(cfg):
    print('jobname: ', cfg.name)

    # set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # get data
    train_loader, val_loader, test_loader = get_bandit_loaders(**cfg.data)

    # initialize algorithm
    alg = hydra.utils.instantiate(cfg.alg)

    # set up logging
    if cfg.logging:
        log_file_path = cfg.log_path + '/' + cfg.name
        try:
            shutil.rmtree(log_file_path)
        except OSError as e:
            print("Error: %s : %s" % (log_file_path, e.strerror))
        store = Store(cfg.log_path, cfg.name)
        store.add_table_like_example('metadata',dict(cfg))
        store['metadata'].append_row(dict(cfg))
        store.add_table('learning_curves', {
                        'epoch':int,
                        'mode':str,
                        'loss':float,
                        'reward':float,
                        'pct_optimal':float,
                        'pct_imitate':float,
        })

    if not os.path.exists(cfg.model_save_path):
        os.makedirs(cfg.model_save_path)
    model_path = cfg.model_save_path + '/' + cfg.name

    # train
    for epoch in range(1, cfg.epochs+1):
        print("Epoch: ", epoch)

        alg.train(train_loader, epoch)

        alg.eval(train_loader, epoch, store, mode="train")
        alg.eval(test_loader, epoch, store, mode="test")
        alg.eval(val_loader, epoch, store, mode="val")

        if epoch % cfg.model_save_freq == 0:
            alg.save(model_path + '_' + str(epoch) + '.pt')

    if cfg.logging:
        store.close()


if __name__ == "__main__":
    train()
