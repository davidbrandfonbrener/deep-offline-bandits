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
#    overrides.add('+hydra.launcher', ['cpu'])

    # paths
    overrides.add(key='root_name', values=['stability_sweep'])
    overrides.add(key='model_save_dir', values=['models'])
    # alg params
    overrides.add(key='alg', values=['q', 'iw', 'ff'])
    overrides.add(key='alg.lr', values=[0.01])
    overrides.add(key='epochs', values=[1000])
    overrides.add(key='batch_size', values=[10])
    # data params
    overrides.add(key='data.n', values=[100])
    overrides.add(key='data.K', values=[2])
    overrides.add(key='data.d', values=[10])
    overrides.add(key='data.gamma', values=[0.0])
    overrides.add(key='data.epsilon', values=[0.1])
    overrides.add(key='data.seed', values=[0]) # list(range(31,51)) list(range(1,31))
    overrides.add(key='data.action_seed', values=list(range(20)))
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


