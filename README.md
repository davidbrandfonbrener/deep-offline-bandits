# deep-offline-bandits

This is the codebase accompanying our ICML 2021 paper [Offline Contextual Bandits with Overparameterized Models](https://arxiv.org/pdf/2006.15368.pdf).

---

## Dependencies

- Python 3.7
- Pytorch 1.7
- [Hydra] (https://hydra.cc/) 1.0
- [Cox](https://cox.readthedocs.io/en/latest/) 

---

## Usage

To run `train.py`, you just need to put your path to the `deep-offline-bandits` directory into the `config/train.yaml` file as the `path` variable. Then you can run `python train.py` and overwrite any of the parameters in the config file using hydra. For example, to run value-based learning instead of importance-weighted policy-based learning, you can run `python train.py alg=q`.

The `sweeps/` directory contains files to launch slurm sweeps that we ran for the paper. Results of our sweeps are found in the `cox_logs/` directory. 

For the CIFAR experiments, you just need to download the CIFAR10 dataset from `torchvision.datasets` into the `data/` directory. The actions used in our experiments are in the `action_data/` directory. Synthetic data with actions is already found in the `data/` directory. Data loading is then handled in the `utils/data.py` file. 

---
## Citing

If you use this codebase in a paper, please cite our ICML paper:

```
@inproceedings{brandfonbrener2021offline,
  title={Offline Contextual Bandits with Overparameterized Models},
  author={Brandfonbrener, David and Whitney, William and Ranganath, Rajesh and Bruna, Joan},
  booktitle={International Conference on Machine Learning},
  pages={1049--1058},
  year={2021},
  organization={PMLR}
}
```