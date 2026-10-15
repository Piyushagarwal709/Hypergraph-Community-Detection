# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022
FMHMO-Main: 基于模糊信息的超图社区检测
@author: WYW
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022
FMHMO-Main: 基于模糊信息的超图社区检测
@author: WYW
"""
import numpy as np

from sklearn import metrics
import igraph as ig


from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

# 引入外部函数
import FMHMO_utils as HG_util
import FMHMO_function as HG_func


def maximize_alignment(listA, listB):
    # 在两个列表之间创建一个混淆矩阵
    confusion_mat = confusion_matrix(listA, listB)

    # 使用线性和分配来找到最优分配
    row_ind, col_ind = linear_sum_assignment(-confusion_mat)

    # 根据最佳分配创建映射
    label_mapping = dict(zip(col_ind, row_ind))

    # 映射第二个列表中的标签以最大限度地对齐
    aligned_listB = [label_mapping[label] for label in listB]

    return aligned_listB

# =============================================================================
# 基于模糊信息的超图社区检测算法
# =============================================================================
dir_path = r"other_logs/real/Hy_MT/"

network_names = ['hospital','workspace','primary_school','high_school','house_committees','cora_ca','citeseer','enron_email','enron_email','pubmed']
# network_names = ['hospital','workspace','primary_school','high_school','house_committees','cora_ca','citeseer','enron_email','gene_disease']


for network_name in network_names:
    file_path = dir_path + f"{network_name}.log"
    with open(file_path, mode='r', encoding='UTF-8') as f:
        lines = f.read().splitlines()

    membership_c_s = []
    for line in lines:
        if 'membership_c' in line:
            membership_c_str = line.split('membership_c=')[1].strip()[1:-1]
            membership_c_list = membership_c_str.split(',') if ',' in membership_c_str else membership_c_str.split(' ')
            membership_c_list = list(map(int, membership_c_list))
            membership_c_s.append(membership_c_list)

    ### 获得网络信息 & 初始化log日志
    network, network_name, groundtruth_path = HG_util.network_set(net=network_name, network_type='real')
    H = HG_func.createHyperGraph(network)
    real_mem = HG_util.real_cd_obtain(groundtruth_path)

    # 初始化数据
    FMIs = []
    for membership_c in membership_c_s:
        # 多指标计算
        partition = {}
        for cno in set(membership_c): partition[cno] = []
        for node, icno in enumerate(membership_c): partition[icno].append(node)
        X_nmi = ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False)
        X = maximize_alignment(real_mem, membership_c)
        FMI = metrics.fowlkes_mallows_score(labels_true=real_mem, labels_pred=X)
        FMIs.append(FMI)
    print('{}\t{}'.format(round(np.mean(FMIs), 6), round(np.std(FMIs), 6)))
