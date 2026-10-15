# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022
FHGCD-Main: 基于模糊信息的超图社区检测
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
FHGCD-Main: 基于模糊信息的超图社区检测
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
dir_path = r"other_logs/lfr/lfr_IRMM/lfr/"

for NO_ in range(0,3,1):
    file_path = dir_path + f"lfr_1{NO_}_he_1.log"
    with open(file_path, mode='r', encoding='UTF-8') as f:
        lines = f.read().splitlines()

    membership_c_s = []
    for line in lines:
        if 'membership_c' in line:
            membership_c_str = line.split('membership_c=')[1].strip()[1:-1]
            membership_c_list = membership_c_str.split(', ') if ',' in membership_c_str else membership_c_str.split(' ')
            membership_c_list = list(map(int, membership_c_list))
            membership_c_s.append(membership_c_list)

    for NO in range(1,10,1):
        ### 获得网络信息 & 初始化log日志
        network, network_name, groundtruth_path = HG_util.network_set(net=str(NO*10+NO_), network_type='lfr')
        log = myLog.create_log(name="ros_log", level=logging.DEBUG, log_dir="other_logs/lfr/lfr_IRMM", filename=network_name + ".log", sh_level=logging.INFO, fh_level=logging.DEBUG)
        H = HG_func.createHyperGraph(network)
        real_mem = HG_util.real_cd_obtain(groundtruth_path)

        # 初始化数据
        RITA = {}
        RITA['MGs'], RITA['SGs'], RITA['Hcuts'], RITA['nmis'], RITA['ARIs'], RITA['F1s'], RITA['FMIs'], RITA['Cs'] = [], [], [], [], [], [], [], []
        RITA['Precisions'], RITA['Accuracys'], RITA['Recalls'], RITA['AMIs'] = [], [], [],[]
        for index in range(4):
            # 多指标计算
            membership_c = membership_c_s[(NO-1)*4+index]
            partition = {}
            for cno in set(membership_c): partition[cno] = []
            for node, icno in enumerate(membership_c): partition[icno].append(node)
            ms.MultiIndexCalculation(log, RITA, H, real_mem, membership_c, list(partition.values()), True) #RITA
        log.info("########### 多指标计算 RITA {} ###########".format(network_name))
        log.info('MG_mean={},std={}, max={}'.format(round(np.mean(RITA['MGs']), 6), round(np.std(RITA['MGs']), 6), round(max(RITA['MGs']), 6)))
        log.info('SG_mean={},std={}, max={}'.format(round(np.mean(RITA['SGs']), 6), round(np.std(RITA['SGs']), 6), round(max(RITA['SGs']), 6)))
        log.info('Hcut_mean={},std={}, min={}'.format(round(np.mean(RITA['Hcuts']), 6), round(np.std(RITA['Hcuts']), 6), round(min(RITA['Hcuts']), 6)))
        log.info('nmis_mean={},std={}, max={}'.format(round(np.mean(RITA['nmis']), 6), round(np.std(RITA['nmis']), 6), round(max(RITA['nmis']), 6)))
        log.info('ARIs_mean={},std={}, max={}'.format(round(np.mean(RITA['ARIs']), 6), round(np.std(RITA['ARIs']), 6), round(max(RITA['ARIs']), 6)))
        log.info('F1_mean={},std={}, max={}'.format(round(np.mean(RITA['F1s']), 6), round(np.std(RITA['F1s']), 6), round(max(RITA['F1s']), 6)))
        log.info('FMI_mean={},std={}, max={}'.format(round(np.mean(RITA['FMIs']), 6), round(np.std(RITA['FMIs']), 6), round(max(RITA['FMIs']), 6)))
        log.info('AMI_mean={},std={}, max={}'.format(round(np.mean(RITA['AMIs']), 6), round(np.std(RITA['AMIs']), 6), round(max(RITA['AMIs']), 6)))
        log.info('Cs_mean={},std={}, max={}\n'.format(round(np.mean(RITA['Cs']), 6), round(np.std(RITA['Cs']), 6), round(max(RITA['Cs']), 6)))
        log.info('Precisions_mean={},std={}, max={}'.format(round(np.mean(RITA['Precisions']), 6), round(np.std(RITA['Precisions']), 6), round(max(RITA['Precisions']), 6)))
        log.info('Accuracys_mean={},std={}, max={}'.format(round(np.mean(RITA['Accuracys']), 6), round(np.std(RITA['Accuracys']), 6), round(max(RITA['Accuracys']), 6)))
        log.info('Recalls_mean={},std={}, max={}'.format(round(np.mean(RITA['Recalls']), 6), round(np.std(RITA['Recalls']), 6), round(max(RITA['Recalls']), 6)))
        log.info('AMIs_mean={},std={}, max={}'.format(round(np.mean(RITA['AMIs']), 6), round(np.std(RITA['AMIs']), 6), round(max(RITA['AMIs']), 6)))
