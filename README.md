# Focused-Flip-Federated-Backdoor-Attack
This is the official code for our paper [On the Vulnerability of Backdoor Defenses for Federated Learning](https://arxiv.org/abs/2301.08170) (accepted by AAAI-2023) by Pei Fang and  [Jinghui Chen](https://jinghuichen.github.io/) .

## Prerequisites

- Python (3.8+, is a must)
- Pytorch (1.11)
- CUDA (1.10+)
- some other packages (just conda install or pip install)

## Step to run
1. get into the directory `Focused-Flip-Federated-Backdoor-Attack/`

2. get Tiny-ImageNet dataset

   ```bash
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
   unzip tiny-imagenet-200.zip
   ```

3. run Bases.py
   ```bash
   python Bases.py 
   --defense {fedavg,ensemble-distillation,mediod-distillation,fine-tuning,mitigation-pruning,robustlr,certified-robustness,bulyan,deep-sight} 
   --config {cifar,imagenet} 
   --backdoor {ff,dba,naive,neurotoxin}
   --model {simple,resnet18}
   ```

   Hyperparameters about attack and defense baselines are mostly in `Params.py`, hyperparameters about dataset are mostly in `configs/`

## Citation
Please check our paper for technical details and full results. If you find our paper useful, please cite:
   ```
   @article{fang2023vulnerability,
     title={On the Vulnerability of Backdoor Defenses for Federated Learning},
     author={Fang, Pei and Chen, Jinghui},
     journal={Proceedings of the AAAI Conference on Artificial Intelligence},
     year={2023}
   }
   ```
