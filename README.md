# Text-like Encoding of Collaborative Information in Large Language Models for Recommendation


**This repository is constructed based on [CoLLM](https://github.com/zyang1580/CoLLM)! Read CoLLM "readme.md" to understand the code structure!**

** Our trained models can be found at [here](https://rec.ustc.edu.cn/share/ddf0ccf0-5fb3-11ef-93eb-23d2eed3b4d2).**





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

## 
If you're using CoLLM code in your research or applications, please cite our papers:
```bibtex
@inproceedings{zhang-etal-2024-text,
    title = "Text-like Encoding of Collaborative Information in Large Language Models for Recommendation",
    author = "Zhang, Yang  and Bao, Keqin  and Yan, Ming  and Wang, Wenjie  and Feng, Fuli  and He, Xiangnan",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2024",
    url = "https://aclanthology.org/2024.acl-long.497",
    pages = "9181--9191"
}
```

```bibtex
@article{zhang2023collm,
  title={CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation},
  author={Zhang, Yang and Feng, Fuli and Zhang, Jizhi and Bao, Keqin and Wang, Qifan and He, Xiangnan},
  journal={arXiv preprint arXiv:2310.19488},
  year={2023}
}
```
You may also need to cite the [MiniGPT-4 paper](https://arxiv.org/abs/2304.10592). 
