import os
import argparse
from collections import OrderedDict

from subprocess import Popen


class Overrides(object):
    def __init__(self):
        self.kvs = OrderedDict()

    def add(self, key, values):
        value = ','.join(str(v) for v in values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    overrides = Overrides()
#    overrides.add('hydra/launcher', ['submitit_slurm'])
#    overrides.add('+hydra.launcher', ['gpu'])

    # paths
    overrides.add(key='root_name', values=['cifar_sweep'])
    overrides.add(key='model_save_dir', values=['models'])
    # alg params
    overrides.add(key='alg', values=['iw'])
    overrides.add(key='alg.offset', values=[-0.1, 0.1])
    overrides.add(key='alg.model_type', values=['resnet'])
    overrides.add(key='alg.lr', values=[0.1])
    overrides.add(key='alg.weight_decay', values=[0.0001])
    overrides.add(key='epochs', values=[1000])
    # data params
    overrides.add(key='data.data_type', values=['cifar10'])
    overrides.add(key='data.d', values=[3072])
    overrides.add(key='data.K', values=[10])
    overrides.add(key='data.action_dir', values=['uniform_0', 'blbf'])
    # seeds
    overrides.add(key='seed', values=[0])

    cmd = ['python', 'train.py', '-m']
    cmd += overrides.cmd()

    if args.dry:
        print(cmd)
    else:
        env = os.environ.copy()
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()


