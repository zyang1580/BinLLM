# Text-like Encoding of Collaborative Information in Large Language Models for Recommendation


**This repository is constructed based on [CoLLM](https://github.com/zyang1580/CoLLM)! Read CoLLM "readme.md" to understand the code structure!**





## Step1: Following CoLLM to create environment and prepare Vicuna.

## step2: Pre-training for Text-like Encoding:
```bash
CUDA_VISIBLE_DEVICES=6,7 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood.yaml > /log.out &
```

## step3: LoRA Tuning
   

step 1: training without collaborative info.
```bash
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood.yaml > /log.out & 
```
Note: Please download "train_collm_mf_din.py" and collm_pretrain_mf_ood.yaml form CoLLM repository

step 2: training with collaborative info.
```bash
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_binllm.py  --cfg-path=train_configs/hash_CF_ml.yaml > /log.out & 
```
