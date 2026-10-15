# -*- coding: utf-8 -*-
import copy
import os
# 在导入scikit-learn之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'  # 可以根据你的CPU核心数调整这个值

import networkx as nx
import torch
from community import community_louvain
# 引入外部函数
import measure as ms
from FHGCD.HNN_kmeans_function import X_Feature_get, HNN_Kmeans
from FHGCD.utils.FHGCD_HNN_kmeans_utils import build_hypergraph_incidence_matrix

# -*- coding: utf-8 -*-

import logging
import numpy as np
import MyLogging as myLog

# 引入外部函数
import FMHMO_utils as HG_util
import FMHMO_function as HG_func
import result_show as rs
import utils.hypergraph_utils as hgut

# =============================================================================
# 基于模糊信息的超图社区检测算法
network_names = ['hospital', 'workspace', 'primary_school', 'high_school', 'house_committees', 'cora_ca',
                 'citeseer', 'enron_email', 'gene_disease', 'pubmed']

# network_names = ['enron_email']
# =============================================================================
### 获得网络信息 & 初始化log日志

# network_name = 'house_committees'
for network_name in network_names:
    gamma = 0.5

    # 移除之前的日志处理器（如果有）
    if 'log' in locals():
        for handler in log.handlers[:]:
            handler.close()
            log.removeHandler(handler)

    log = myLog.create_log(name="ros_log", level=logging.DEBUG, log_dir=f"logs/HNN_kmeans_real/{network_name}", filename=network_name + f"_{round(gamma, 1)}.log", sh_level=logging.INFO, fh_level=logging.DEBUG)
    # network_names = ['hospital', 'workspace', 'primary_school', 'high_school', 'house_committees', 'cora_ca',
    #                  'citeseer', 'enron_email', 'gene_disease', 'pubmed']

    network, network_name, groundtruth_path = HG_util.network_set(net=network_name, network_type='real')
    ### 超图创建 & 超图信息获取
    H = HG_func.createHyperGraph(network)
    HG_info = HG_util.HG_info_obtain(log, H)
    ### 超图转化 & 加权图信息获取
    WG, WG_nx, G_info = HG_util.HG_to_WG(HG_info,gamma)
    WG_info, Q_info = HG_util.WG_info_obtain(WG, G_info)
    # print(np.allclose(WG_info['HEW_adj'], WG_info['HEW_adj'].T))

    ### 算法参数设置 & 真实社区划分获取
    FHGCD_params = HG_util.FGHCD_params_set()
    real_mem = HG_util.real_cd_obtain(groundtruth_path)

    # 构建超图矩阵，满足n个节点m条超边，得到n*m矩阵，元素值为1表示当前节点在当前超边内
    H_adj_matrix = build_hypergraph_incidence_matrix(HG_info['H_n'], HG_info['H_edges'])
    # 计算卷积层 G_conv = Dv_inv_sqrt @ H @ W @ De_inv @ H.T @ Dv_inv_sqrt
    G_conv = hgut.generate_G_from_H(H_adj_matrix)

    # =============================================================================
    # 程序运行
    # =============================================================================
    log.info("####### FHGCD_Louvain network:{0}==n:{1}==e:{2}==ave_degree:{3} #######".format(network_name, Q_info['n'], len(WG_info['edge_all']), WG_info['average_degrees']))
    # 初始化数据
    RITA = {}
    RITA['Qws'], RITA['MGs'], RITA['SGs'], RITA['Hcuts'], RITA['nmis'], RITA['ARIs'], RITA['F1s'], RITA['FMIs'], RITA['Cs'] = [], [], [], [], [], [], [], [], []
    RITA['Precisions'], RITA['Accuracys'], RITA['Recalls'], RITA['AMIs'] = [], [], [],[]
    # RITA_NCM = copy.deepcopy(RITA)
    # RITA_NCR = copy.deepcopy(RITA)
    run = 0  # 本程序开始独立运行的次数
    while (run < FHGCD_params['Independent_Runs']):
        # max_QW = 0.0
        # if nx.number_connected_components(WG_nx) == 1:
        #     for i in range(1):
        #         ml = community_louvain.best_partition(graph=WG_nx, weight='weight') # 使用Louvain算法对加权图进行社区检测，获得加权图社区划分（该louvain算法只能做连通图的社区划分）
        #         membership_c = []
        #         # for i in range(len(ml)): membership_c.append(ml[i])
        #         # 方法3：先验证键是否存在
        #         for i in range(len(ml)):
        #             if i in ml:
        #                 membership_c.append(ml[i])
        #             else:
        #                 print(f"键 {i} 不存在于ml中")
        #         X_QW = WG.modularity(membership=membership_c,weights='weight')  # 加权模块度值计算
        #         if X_QW > max_QW: max_QW, best_membership_c = X_QW, membership_c
        # else:
        #     for i in range(5):
        #         ml = WG.community_multilevel(weights="weight", resolution=1.0) #(可以做非联通图的社区划分)
        #         PL = [x for x in ml]
        #         membership_c = [-1] * Q_info['n']
        #         for cno, cnodes in enumerate(PL):
        #             for i in cnodes:
        #                 membership_c[i] = cno
        #         X_QW = ml.modularity
        #         if X_QW>max_QW: max_QW,best_membership_c = X_QW,membership_c

        # log.info("\nGW_membership_c={}".format(best_membership_c))
        # log.info("\nQW={}, C={}".format(X_QW,len(set(best_membership_c))))

        # 构建输入特征，特征数为真实分区数量，元素值服从正态分布
        X_features, X_labels = X_Feature_get(HG_info, real_mem, feature_noise=0.1)

        # 输入数据，得到聚类值
        X_HNN_result, X_HNN_partition = HNN_Kmeans(X_features, X_labels, G_conv)

        # 将加权图社区划分转换为超图社区划分 & 计算超图社区划分指标
        # X_result,X_partition_result, X_NCM_result,X_NCM_partition_result, X_NCR_result,X_NCR_partition_result = HG_util.WGCD_to_HGCD(HG_info, WG_info, best_membership_c, FHGCD_params)
        # 多指标计算
        ms.MultiIndexCalculation(log, RITA, HG_info['H'], real_mem, X_HNN_result, list(X_HNN_partition.values()), True) #RITA
        # ms.MultiIndexCalculation(log, RITA_NCM, HG_info['H'], real_mem, X_NCM_result, list(X_NCM_partition_result.values()), True) #RITA_NCN
        # ms.MultiIndexCalculation(log, RITA_NCR, HG_info['H'], real_mem, X_NCR_result, list(X_NCR_partition_result.values()), True) #RITA_NCR
        log.info("\nHy_partition={}".format(X_HNN_partition))
        log.info("Hy_partition_len={}".format(len(X_HNN_partition)))
        # RITA['Qws'].append(max_QW)
        run += 1

    rs.print_result(log,RITA,network_name)
    log.info("gamma={}".format(gamma))

    # 在循环结束时明确关闭日志处理器
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)