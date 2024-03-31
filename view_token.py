import argparse
import os
# import os
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

# import minigpt4.tasks as tasks
# from minigpt4.common.config import Config
# from minigpt4.common.dist_utils import get_rank, init_distributed_mode
# from minigpt4.common.logger import setup_logger
# from minigpt4.common.optims import (
#     LinearWarmupCosineLRScheduler,
#     LinearWarmupStepLRScheduler,
# )
# from minigpt4.common.registry import registry
# from minigpt4.common.utils import now

# # imports modules for registration
# from minigpt4.datasets.builders import *
# from minigpt4.models import *
# from minigpt4.processors import *
# from minigpt4.runners import *
# from minigpt4.tasks import *
# from torch.distributed.elastic.multiprocessing.errors import *


import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import os

# from minigpt4.common.registry import registry
# from minigpt4.models.rec_model import Rec2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, GenerationConfig
import re
import numpy as np
# from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict


llama_model  =  "/data/LLM/PretrainedModels/vicuna/working-v0/"

llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
llama_tokenizer.pad_token = llama_tokenizer.eos_token    
# llama_model = LlamaForCausalLM.from_pretrained(
#     llama_model,
#     torch_dtype=torch.float16,
#     load_in_8bit=True,
#     device_map={'': int(os.environ.get("LOCAL_RANK") or 0)}
# )

# m = np.random.randn(32).astype(int)
# m[m>0] = 1
# m[m<=0] = 0
# m = list(m)
# m = [str(x) for x in m]
# m = ''.join(m)
m = '192.168.122.234'
prompt_list = [m]
llama_tokenizer.padding_side = "left"
prompts_tokens = llama_tokenizer(
prompt_list,
return_tensors="pt",
padding="longest",
truncation=True,
max_length=1024,
add_special_tokens=False
)

unk_token_id = llama_tokenizer.unk_token_id
# prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
