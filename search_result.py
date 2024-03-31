import numpy as np
import pandas as pd
import ast

def find_max(log_path):
    with open(log_path,'r') as f:
        lines = f.readlines()
        configs = []
        results = []
        i = 0
        for  one_line in lines:
            # print(one_line)
            i+=1
            # if i%2!=0:
            #     continue
            # if "\'embedding_size\': 32" not in one_line:
            #     continue
            one_line = one_line.strip("train_config: ").split(" best result: ")
            try:
                configs.append(ast.literal_eval(one_line[0]))
                results.append(ast.literal_eval(one_line[1]))
            except:
                print(one_line)
                raise RuntimeError

    max_k = 0
    max_valid_auc = 0
    for k in range(len(results)):
        if results[k]['valid_auc'] > max_valid_auc:
            max_k = k
            max_valid_auc = results[k]['valid_auc']
    return configs[max_k], results[max_k]




print(find_max("log/xxxxx.log"),'\n')
