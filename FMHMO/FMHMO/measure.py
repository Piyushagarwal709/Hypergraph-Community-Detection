# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:28:52 2023

@author: WYW

    用于社区检测评价指标计算
"""
import numpy as np
from sklearn import metrics
import igraph as ig

import strictModularity as sm

from collections import Counter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


# =============================================================================
# hcut
# =============================================================================
def hcut(H, P):
    s = 0
    l = 0
    for i in range(len(H)):
        if len(H[i]) == 0: continue
        l = l + len(H[i])
        for j in range(len(P)):
            s += sum([x < set(P[j]) for x in H[i]])
    return (l - s) / l


# =============================================================================
# measure_f1： F1_sore计算函数
# cluster_labels： 聚类的标签
# ground_truth： 真实社区划分
# =============================================================================
def measure_f1(cluster_labels, ground_truth):
    cluster_pairs = Counter(zip(cluster_labels, ground_truth))
    TP = sum(count * (count - 1) // 2 for count in cluster_pairs.values())
    cluster_counts = Counter(cluster_labels)
    ground_truth_counts = Counter(ground_truth)
    FP = sum(count * (count - 1) // 2 for count in cluster_counts.values()) - TP
    FN = sum(count * (count - 1) // 2 for count in ground_truth_counts.values()) - TP
    if TP == 0:
        return 0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


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
# 多指标计算
# =============================================================================
def MultiIndexCalculation(log, RITA, H, real_mem, X, A, verbose=False):
    X_nmi = ig.compare_communities(real_mem, X, method='nmi', remove_none=False)
    membership_c = maximize_alignment(real_mem, X)
    c_num = len(set(X))
    # 质量性指标
    MG = sm.modularityG(H, A)
    SG = sm.modularityH(H, A)
    Hcut = hcut(H, A)

    # 精确性指标
    nmi = ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False)
    if X_nmi!=nmi: print("ERROR! X_nmi={}, nmi={}".format(X_nmi, nmi))
    ARI_score = metrics.adjusted_rand_score(real_mem, membership_c)
    # F1_score_1 = measure_f1(membership_c, real_mem)
    F1_score = metrics.f1_score(real_mem, membership_c, average='micro')
    Precision_score = metrics.precision_score(real_mem, membership_c, average='micro')
    Accuracy = metrics.accuracy_score(y_true=real_mem,y_pred=membership_c)
    Recall_score = metrics.recall_score(y_true=real_mem,y_pred=membership_c,average='micro')
    FMI = metrics.fowlkes_mallows_score(labels_true=real_mem, labels_pred=membership_c)
    AMI = metrics.adjusted_mutual_info_score(labels_true=real_mem, labels_pred=membership_c)

    RITA['MGs'].append(MG)
    RITA['SGs'].append(SG)
    RITA['Hcuts'].append(Hcut)
    RITA['nmis'].append(nmi)
    RITA['ARIs'].append(ARI_score)
    RITA['F1s'].append(F1_score)
    RITA['FMIs'].append(FMI)
    RITA['Cs'].append(c_num)
    RITA['Precisions'].append(Precision_score)
    RITA['Accuracys'].append(Accuracy)
    RITA['Recalls'].append(Recall_score)
    RITA['AMIs'].append(AMI)
    if verbose:
        log.info("\n########### 多指标计算 ###########")
        log.info("MG={}, SG={}, Hcut={}, NMI={}, ARI={}, F1={}, FMI={}".format(MG, SG, Hcut, nmi, ARI_score, F1_score, FMI))
        log.info("Precision={},Accuracy={},Recall={},AMI={}".format(Precision_score, Accuracy, Recall_score, AMI))
