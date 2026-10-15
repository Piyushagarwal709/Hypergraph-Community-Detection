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
import result_show as rs

# =============================================================================
# 基于模糊信息的超图社区检测算法
# =============================================================================
for NO in range(1, 5, 1):
    gamma = 0.5
    ### 获得网络信息 & 初始化log日志
    network, network_name, groundtruth_path = HG_util.network_set(net=str(NO*1000), network_type='lfr')

    log = myLog.create_log(name="ros_log", level=logging.DEBUG, log_dir="logs/louvain_lfr", filename=network_name + f"_{gamma}.log", sh_level=logging.INFO, fh_level=logging.DEBUG)
    ### 超图创建 & 超图信息获取
    H = HG_func.createHyperGraph(network)
    HG_info = HG_util.HG_info_obtain(log, H)

    ### 超图转化 & 加权图信息获取
    WG,WG_nx, WHEW_adj = HG_util.HG_to_WG(HG_info, gamma)
    WG_info, Q_info = HG_util.WG_info_obtain(WG, WHEW_adj)

    ### 算法参数设置 & 真实社区划分获取
    FMHMO_params = HG_util.FMHMO_params_set()
    real_mem = HG_util.real_cd_obtain(groundtruth_path)

    # =============================================================================
    # 程序运行
    # =============================================================================
    log.info("####### FMHMO_Louvain network:{0}==n:{1}==e:{2}==ave_degree:{3} #######".format(network_name, Q_info['n'], len(WG_info['edge_all']), WG_info['average_degrees']))
    # 初始化数据
    RITA = {}
    RITA['Qws'], RITA['MGs'], RITA['SGs'], RITA['Hcuts'], RITA['nmis'], RITA['ARIs'], RITA['F1s'], RITA['FMIs'], RITA['Cs'] = [], [], [], [], [], [], [], [], []
    RITA['Precisions'], RITA['Accuracys'], RITA['Recalls'], RITA['AMIs'] = [], [], [],[]
    RITA_NCM = copy.deepcopy(RITA)
    RITA_NCR = copy.deepcopy(RITA)
    run = 0  # 本程序开始独立运行的次数
    while (run < 5):
        max_QW = 0.0
        if nx.number_connected_components(WG_nx) == 1:
            for i in range(3):
                ml = community_louvain.best_partition(graph=WG_nx, weight='weight') # 使用Louvain算法对加权图进行社区检测，获得加权图社区划分（该louvain算法只能做连通图的社区划分）
                membership_c = []
                for i in range(len(ml)): membership_c.append(ml[i])
                X_QW = WG.modularity(membership=membership_c,weights='weight')  # 加权模块度值计算
                if X_QW > max_QW: max_QW, best_membership_c = X_QW, membership_c
        else:
            for i in range(20):
                ml = WG.community_multilevel(weights="weight", resolution=1.0) #(可以做非联通图的社区划分)
                PL = [x for x in ml]
                membership_c = [-1] * Q_info['n']
                for cno, cnodes in enumerate(PL):
                    for i in cnodes:
                        membership_c[i] = cno
                X_QW = ml.modularity
                if X_QW>max_QW: max_QW,best_membership_c = X_QW,membership_c

        log.info("\nGW_membership_c={}".format(best_membership_c))
        print("\nQW={}, C={}".format(X_QW,len(set(best_membership_c))))

        # 将加权图社区划分转换为超图社区划分 & 计算超图社区划分指标
        X_result,X_partition_result, X_NCM_result,X_NCM_partition_result, X_NCR_result,X_NCR_partition_result = HG_util.WGCD_to_HGCD(HG_info, WG_info, best_membership_c, FMHMO_params)
        # 多指标计算
        ms.MultiIndexCalculation(log, RITA, HG_info['H'], real_mem, X_result, list(X_partition_result.values()), True) #RITA
        ms.MultiIndexCalculation(log, RITA_NCM, HG_info['H'], real_mem, X_NCM_result, list(X_NCM_partition_result.values()), True) #RITA_NCN
        ms.MultiIndexCalculation(log, RITA_NCR, HG_info['H'], real_mem, X_NCR_result, list(X_NCR_partition_result.values()), True) #RITA_NCR
        log.info("\nHy_partition={}".format(X_partition_result))
        log.info("Hy_partition_len={}".format(len(X_partition_result)))
        RITA['Qws'].append(max_QW)
        run += 1

    rs.print_result(log,RITA,RITA_NCM,RITA_NCR,network_name)
    log.info("gamma={}".format(gamma))
