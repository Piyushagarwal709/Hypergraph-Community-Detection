# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022
FMHMO-Main: 基于模糊信息的超图社区检测
@author: WYW
"""
import copy

import networkx as nx
from community import community_louvain
# 引入外部函数
import measure as ms

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022
FMHMO-Main: 基于模糊信息的超图社区检测
@author: WYW
"""
import logging
import numpy as np
import MyLogging as myLog

# 引入外部函数
import FMHMO_utils as HG_util
import FMHMO_function as HG_func
# =============================================================================
# 基于模糊信息的超图社区检测算法
# =============================================================================
dir_path = r"other_logs/lfr/lfr_Hy_MT/"

file_path = dir_path + f"lfr_10_he.log"
with open(file_path, mode='r', encoding='UTF-8') as f:
    lines = f.read().splitlines()

mean_v_s = []
std_v_s = []
for line in lines:
    if 'FMI_mean=' in line:
        measure_s = line.strip().split('FMI_mean=')[1].split(",")
        mean_v,std_v = float(measure_s[0].strip()),float(measure_s[1].strip().split('=')[1].strip())
        mean_v_s.append(mean_v)
        std_v_s.append(std_v)

FMI_k_mean,FMI_k_std = 0.0,0.0
for k in range(9):
    FMI_k_mean = (mean_v_s[k] + mean_v_s[k+9] + mean_v_s[k+18])/3
    FMI_k_std = (std_v_s[k] + std_v_s[k+9] + std_v_s[k+18])/3
    # 初始化数据
    print('{}\t{}'.format(round(FMI_k_mean, 6), round(FMI_k_std, 6)))

# for k in range(0,27,3):
#     FMI_k_mean = (mean_v_s[k] + mean_v_s[k+1] + mean_v_s[k+2])/3
#     FMI_k_std = (std_v_s[k] + std_v_s[k+1] + std_v_s[k+2])/3
#     # 初始化数据
#     print('{}\t{}'.format(round(FMI_k_mean, 6), round(FMI_k_std, 6)))
#

