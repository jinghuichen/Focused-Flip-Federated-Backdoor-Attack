# Focused-Flip-Federated-Backdoor-Attack
This is the official code for our paper [On the Vulnerability of Backdoor Defenses for Federated Learning](https://arxiv.org/abs/2301.08170)(aceepted by AAAI-2023) by Pei Fang and  [Jinghui Chen](https://jinghuichen.github.io/) .

## Prerequisites

- Python (3.8+, required)
- Pytorch (1.11)
- CUDA (1.10+)

- small other packages (just conda install or pip install)

## Step to run
1. get into the Directory `/Focused-Flip-Federated-Backdoor-Attack`

2. run command to get Tiny-ImageNet

   ```bash
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
   unzip tiny-imagenet-200.zip
   ```

3. run Bases.py
```
python Bases.py 
--defense {fedavg,ensemble-distillation,mediod-distillation,fine-tuning,mitigation-pruning,robustlr,certified-robustness,bulyan,deep-sight} 
--config {cifar,imagenet} 
--backdoor {ff,dba,naive,neurotoxin}
--model {simple,resnet18}
```

parameters about attack and defense baselines are mostly in `Params.py`, parameters about dataset are mostly in `configs/`

## citation
[TBD]
