import copy

import numpy as np
import random as rd
from collections import Counter
import strictModularity as sm

import cython_function as cfunc


##########################################################

def hcut(H, P):
    s = 0
    l = 0
    for i in range(len(H)):
        if len(H[i]) == 0: continue
        l = l + len(H[i])
        for j in range(len(P)):
            s += sum([x < set(P[j]) for x in H[i]])
    return (l - s) / l


def createHyperGraph(file):
    H = {}
    with open(file, 'r') as f:
        flag_0 = False  # 超图节点是否从0开始编码
        datas = f.readlines()
        max_line_size = -1
        H_tmp = []
        for line in datas:
            if ',' in line:
                data_line_list = str(line).strip().split(',')
            else:
                data_line_list = str(line).strip().split(' ')
            if '0' in data_line_list: flag_0 = True  # 该超图节点编码从0开始
            line_size = len(data_line_list)
            if line_size > max_line_size: max_line_size = line_size
            H_tmp.append((line_size, [int(i) for i in data_line_list]))

        for i in range(max_line_size + 1): H[i] = []
        for s_d in H_tmp:
            he = s_d[1] if flag_0 else [i - 1 for i in s_d[1]]
            H[s_d[0]].append(set(he))
    H_hes = list(H.values())

    return H_hes


# =============================================================================
# 社区间节点社区划分
# =============================================================================
def NCR_NCM(GWX, GWXpartition, Xpartition, HG_info, G_info, FHGCD_params, verbose=False):
    # 如果该节点为社区内节点，则不调整
    H, H_edges, H_n, node_HE_dict, Sij_adj, Uie_dict = HG_info['H'], HG_info['H_edges'], HG_info['H_n'], HG_info['node_HE_dict'], HG_info['Sij_adj'], HG_info['Uie_dict']
    no_he_dict, he_no_dict = HG_info['no_he_dict'], HG_info['he_no_dict']
    HEW_adj, short_path_adj,node_js_dict = G_info['HEW_adj'], G_info['short_path_adj'],HG_info['node_js_dict']

    # 初始社区划分还原(不处理超边重叠社区间节点)
    X = [-1] * H_n
    for cno in Xpartition.keys():
        cno_nodes = Xpartition[cno]
        for i in cno_nodes:
            X[i] = cno

    # 初步还原
    X_NCR,X_partition = copy.deepcopy(X),{}
    if FHGCD_params['NCR_Flag']:
        iter_nodes_list = NCR(X, GWX, GWXpartition, Xpartition, node_HE_dict, he_no_dict, G_info, short_path_adj, Uie_dict) # 对X执行NCR操作

    # X = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    # 计算初步还原后的X模块度值
    for cno in set(X): X_partition[cno] = []
    for node, icno in enumerate(X): X_partition[icno].append(node)
    NCR_Xparaction_SH = sm.modularityH(H, list(X_partition.values()))
    # # 二次还原
    X_NCM,X_NCM_partition = copy.deepcopy(X),{}
    if FHGCD_params['NCR_Flag'] and FHGCD_params['NCM_Flag']:
        NCM(X, H_n, node_HE_dict, Uie_dict, X_partition, Sij_adj, node_js_dict) # 对X执行NCM操作


    for cno in set(X): X_NCM_partition[cno] = []
    for node, icno in enumerate(X): X_NCM_partition[icno].append(node)
    NCM_Xparaction_SH = sm.modularityH(H, list(X_NCM_partition.values()))
    if NCM_Xparaction_SH<NCR_Xparaction_SH: X=copy.deepcopy(X_NCM)

    return X, X_NCM, X_NCR

def NCR(X, GWX, GWXpartition, Xpartition, node_HE_dict, he_no_dict, G_info, short_path_adj, Uie_dict):
    # 获得超边重叠社区间节点集合
    inter_nodes = set()
    for cno in Xpartition.keys():
        for jcno in Xpartition.keys():
            if cno != jcno:
                for i in set(Xpartition[cno]) & set(Xpartition[jcno]):
                    inter_nodes.add(i)

    # 对超边重叠社区间节点执行基于超图节点隶属度的NCR操作
    iter_nodes_list = list(inter_nodes)
    rd.shuffle(iter_nodes_list)
    for i in iter_nodes_list:
        # 获得i节点所参与构成的超边集合
        i_hes = node_HE_dict[i]
        # 初始化i节点对邻域社区的隶属度字典
        i_jcno_mem = {}
        for he in i_hes:
            he_no = he_no_dict[tuple(he)]
            he_cno = GWX[he_no]
            i_jcno_mem[he_cno] = 0.0

        for he in i_hes:
            # 求出he超边对各个邻域社区(包含he所在社区)的隶属度
            he_no = he_no_dict[he] # 超边编号
            he_cno = GWX[he_no]  # 超边所在当前的社区号
            j_nodes = G_info['node_nei_info']["adj"][he_no] # 获得节点 he_no 的所有邻居节点 j_nodes
            j_nodes_c = [GWX[j] for j in j_nodes] # 获得邻居节点 j 所在的社区
            j_nodes_setc = set(j_nodes_c)
            j_nodes_setc.add(he_cno)
            he_no_jcnei_mem, max_cno = cfunc.mem_func(he_no, short_path_adj, G_info['HEW_adj'], GWXpartition, np.asarray(list(j_nodes_setc)), np.asarray(j_nodes_c), list(j_nodes))
            he_cno_mem = he_no_jcnei_mem[he_cno]  # he对cno社区的隶属度
            i_jcno_mem[he_cno] += (Uie_dict[i][he] * he_cno_mem)  # i对超边的隶属属
        X[i] = max(i_jcno_mem.keys(), key=i_jcno_mem.get)
    return iter_nodes_list

def NCM(X,H_n,node_HE_dict,Uie_dict, X_partition, Sij_adj, node_js_dict):
    nodes = [v for v in range(H_n)]
    nodes.sort(reverse=True)
    # nodes = iter_nodes_list
    # 使用超图节点隶属度还原节点社区划分
    counter_NCR_all_num = 0
    for _ in range(10):
        # rd.shuffle(nodes)
        counter_NCR_num = 0
        # 获得社区间节点
        for i in nodes:
            # 获得i节点所参与构成的超边集合
            i_hes = node_HE_dict[i]
            # 初始化i节点对邻域社区的隶属度字典
            i_jcno_mem = {}
            j_nodes_setc  = set()
            for he in i_hes:
                for hei in he: j_nodes_setc.add(X[hei])
            if len(j_nodes_setc) ==1: continue #社区内节点

            for jcno in j_nodes_setc: i_jcno_mem[jcno] = 0.0
            i_jcno_mem_1 = copy.deepcopy(i_jcno_mem)
            # 求出i节点对各个超边邻域社区的归属程度
            for he in i_hes:
                # 获得he中节点所在的社区号
                he_cs = [X[hei] for hei in set(he)-set([i])]
                heic_num_dict = dict(Counter(he_cs))
                # 求出超边he中节点在该社区的比例
                for cno_ in heic_num_dict.keys():
                    # 求出he超边对各个邻域社区(包含he所在社区)的隶属度
                    rv = heic_num_dict[cno_]/len(he)
                    i_jcno_mem[cno_] += (Uie_dict[i][he] * rv)  # i对超边的隶属属

            if i==7:
                print("fdsa")
            sum_v = sum(list(i_jcno_mem.values()))
            for jcno in i_jcno_mem.keys(): i_jcno_mem[jcno] = i_jcno_mem[jcno]/sum_v

            for i_j in node_js_dict[i]:
                i_jcno_mem_1[X[i_j]]+=Sij_adj[i,i_j]

            sum_v_1 = sum(list(i_jcno_mem_1.values()))
            for jcno in i_jcno_mem.keys():
                i_jcno_mem[jcno] = i_jcno_mem[jcno]*(i_jcno_mem_1[jcno]/sum_v_1)

            sum_v = sum(list(i_jcno_mem.values()))
            for jcno in i_jcno_mem.keys():
                i_jcno_mem[jcno] = i_jcno_mem[jcno]/sum_v

            i_max_cno = max(i_jcno_mem.keys(), key=i_jcno_mem.get)

            # if  X[i]!=i_max_cno and i_cno_ihecut_dict[i_max_cno]>0:
            if  X[i]!=i_max_cno:
                X_partition[X[i]].remove(i)
                X_partition[i_max_cno].append(i)
                X[i] = i_max_cno
                counter_NCR_num+=1

        if counter_NCR_num==0: break
        counter_NCR_all_num+=counter_NCR_num

    print("\ncounter_NCR_all_num=",counter_NCR_all_num)
