import copy
import os

import networkx as nx
import numpy as np
import igraph as ig


# 引入外部函数
import FMHMO_function as HG_func

## 真实网络
path = r"data/realNetwork"
zoo_network = path + r'/zoo.txt'
house_committees_network = path + r'/house_committees.txt'
sc_congress_bills_network = path + r'/sc_congress_bills.txt'
# se_congress_bills_network = path + r'/se_congress_bills.txt'
senate_committees_network = path + r'/senate_committees.txt'
football_network = path + r'/football.txt'
primary_school_network = path + r'/primary_school.txt'
high_school_network = path + r'/high_school.txt'
dblp_network = path + r'/dblp.txt'
hospital_network = path + r'/hospital.txt'
workspace_network = path + r'/workspace.txt'
enron_email_network = path + r'/enron_email.txt'
gene_disease_network = path + r'/gene_disease.txt'

cora_ca_network = path + r'/cora_ca.txt'
cora_network = path + r'/cora.txt'
citeseer_network = path + r'/citeseer.txt'
pubmed_network = path + r'/pubmed.txt'

# 示例网络
example_7_network = path + r'/example_7.txt'
example_13_network = path + r'/example_13.txt'

## lfr网络
lfr_path = r"data/lfrNetwork/linear_lfr1000_10000"
# lfr_path = r"data/lfrNetwork/strict_lfr1000"
# lfr_path = r"data/lfrNetwork/majority_lfr100"

# =============================================================================
# 算法各参数设置
# =============================================================================
def FMHMO_params_set():
    # 独立运行运行次数
    Independent_Runs = 5 # 本次实验独立运行次数
    # 是否使用NCR操作
    NCR_Flag = True
    # 是否使用NCM操作
    NCM_Flag = True
    FMHMO_params = dict()
    FMHMO_params['Independent_Runs'],FMHMO_params['NCR_Flag'],FMHMO_params['NCM_Flag']  = Independent_Runs,NCR_Flag,NCM_Flag
    return FMHMO_params

# =============================================================================
# 网络信息设置
# network set
# =============================================================================
def network_set(net, network_type):
    ### 选择网络
    if network_type == "real":  # real network
        network_name = net
        network = eval(network_name + '_network')
        groundtruth_path = path + "/real/" + network_name + '_groundtruth.txt'
    if network_type == "lfr": # lfr network
        lfr_no = net
        network_name = 'lfr_' + lfr_no + '_he'
        network = lfr_path + r'/' + network_name + '.txt'
        groundtruth_path = lfr_path + r'/lfr_' + lfr_no + '_assign.txt'

    return network, network_name, groundtruth_path

def HG_info_obtain(log, H):
    # 获得超边中的节点数量
    H_nodes = set()
    H_edges = set()
    for he_list in H:
        for he in he_list:
            H_edges.add(tuple(sorted(list(he))))
            for i in he: H_nodes.add(i)
    H_n = len(H_nodes)
    H_edges_n = len(H_edges)
    ############################### 用于社区间重叠节点划分社区 ###################################
    # 获得节点所参与的超边集合
    node_HE_dict = {}
    for i in range(H_n):
        node_HE_dict[i] = []
        for he_list in H:
            for he in he_list:
                if i in he: node_HE_dict[i].append(tuple(sorted(he)))
    # 获得最大超边度
    max_v = 0
    for edge in H_edges:
        if max_v < len(edge):
            max_v=len(edge)
    print("max_D:",max_v)


    # 获得节点的邻接节点集合
    node_js_dict = {}
    for i in range(H_n):
        jnodes = []
        jhes = node_HE_dict[i]
        for jhe in jhes: jnodes.extend(jhe)
        jnodes = list(set(jnodes) - set([i]))
        node_js_dict[i] = jnodes

    # 计算节点的平均度
    H_average_node_degree = 0.0
    for i in range(H_n):
        H_average_node_degree += len(node_HE_dict[i])
    H_average_node_degree /= H_n

    # 计算超边的平均度
    H_average_he_degree, he_num, he_classification_num = 0.0, 0, 0
    for helist in H:
        if len(helist) > 0: he_classification_num += 1
        for he in helist:
            he_num += 1
            H_average_he_degree += len(he)
    H_average_he_degree /= he_num

    log.info("##### HG_n: {0} ### HG_he: {1} #####".format(H_n, H_edges_n))
    log.info("##### HG_nodes_avg_d: {0} ### HG_he_avg_D: {1}, ### he_class_num: {2} #####\n".format(
        round(H_average_node_degree, 2), round(H_average_he_degree, 2), he_classification_num))

    # W = sum([len(set(node_HE_dict[i])) for i in range(H_n)])
    # 节点相似度矩阵
    Sij_adj = np.zeros((H_n, H_n))
    for i in range(H_n):
        HEi = set(node_HE_dict[i])
        # HEi_len = len(HEi)
        for j in range(i, H_n):
            HEj = set(node_HE_dict[j])
            # HEj_len = len(set(node_HE_dict[j]))
            intersection = HEi&HEj
            if len(intersection) > 0:
                union = HEi|HEj
                # Sij_adj[i, j] = (len(intersection) / len(union))*((HEi_len*HEj_len)/(2*W))
                Sij_adj[i, j] = (len(intersection) / len(union))
                Sij_adj[j, i] = Sij_adj[i, j]

    # 节点对超边的隶属度字典
    Uie_dict = {}
    for i in range(H_n):
        hei_list = node_HE_dict[i]
        Uie = {}
        for he in hei_list:
            Uie[he] = sum([Sij_adj[i, j] for j in he if j!=i]) / (len(he)-1)
        Uie_sum = sum(list(Uie.values()))
        for he in hei_list:  Uie[he] /= Uie_sum
        Uie_dict[i] = Uie

    HG_info = {}
    HG_info['H'], HG_info['H_edges'], HG_info['H_n'], HG_info['node_HE_dict'], HG_info['Sij_adj'], HG_info['Uie_dict'], HG_info['node_js_dict']  = H, H_edges, H_n, node_HE_dict, Sij_adj, Uie_dict, node_js_dict
    return HG_info

# =============================================================================
# 超图转化为普通加权图
# =============================================================================
def HG_to_WG(HG_info, gamma):
    print("################### HG_to_WG #######################")
    # 转化超图到普通图
    # 为超边编号: no-he
    no_he_dict, he_no = {}, 0
    for he in HG_info['H_edges']:
        no_he_dict[he_no] = he
        he_no += 1

    # 为超边编号: he-no
    he_no_dict = {}
    for i in no_he_dict.keys():
        he_no_dict[no_he_dict[i]] = i
    HG_info['no_he_dict'], HG_info['he_no_dict'] = no_he_dict, he_no_dict

    n = len(no_he_dict)
    elist, w, w2 = [], [], []
    for i in range(n):
        hei = set(no_he_dict[i])
        for j in range(i, n):
            if i == j: continue
            hej = set(no_he_dict[j])
            intersection_set = hei & hej
            hij_intersection_len = len(intersection_set)
            union_sum = sum([len(HG_info['node_HE_dict'][i]) for i in hei | hej])
            intersection_sum = sum([len(HG_info['node_HE_dict'][i]) for i in intersection_set])

            if hij_intersection_len > 0:
                elist.append(tuple(sorted({i, j})))
                simlary_hei_hej_list = []
                for hei_v in hei:
                    for hej_v in hej:
                        simlary_hei_hej_list.append(HG_info['Sij_adj'][hei_v,hej_v])
                simlary_hei_hej = np.mean(simlary_hei_hej_list)
                jaccard_simlary = (2*hij_intersection_len/intersection_sum)*(2*hij_intersection_len/union_sum)
                wij = gamma*simlary_hei_hej + (1-gamma)*jaccard_simlary  #使用混合信息的加权方法
                # 超边权重重置
                w.append(wij)

    G_info = {}
    G_info['n'],G_info['elist'],G_info['weights'],G_info['weights2'] = n,elist,w,w2
    WG = ig.Graph()
    WG.add_vertices(n)
    WG.add_edges([x for x in elist])
    WG.es['weight'] = w
    WG_nx = nx.Graph()
    WG_nx.add_weighted_edges_from([(e[0],e[1],w[index]) for index,e in enumerate(elist)])
    return WG, WG_nx, G_info

# =============================================================================
# 获取加权网络WG的信息
# =============================================================================
def WG_info_obtain(WG, G_info):
    print("################### WG_info_obtain #######################")
    n = WG.vcount()
    adj = WG.get_adjacency()
    ### 网络平均度计算
    degrees_sum = sum(WG.degree())
    average_degrees = round(degrees_sum / n, 2)

    # 初始化模体各节点的边邻域节点集
    node_nei_info, node_adj_neis = dict(), dict()
    for i in range(n):
        node_adj_neis[i] = np.nonzero(adj[i, :])[0]

    node_nei_info["adj"] = node_adj_neis

    # 构建超边加权矩阵
    HEW_adj = np.zeros((n, n))
    for index, edge in enumerate(G_info['elist']):
        HEW_adj[edge[0], edge[1]] += G_info['weights'][index]
        HEW_adj[edge[1], edge[0]] = HEW_adj[edge[0], edge[1]]

    # 最短路径长度
    short_path_adj = np.asarray(WG.shortest_paths_dijkstra(weights='weight'))

    edge_all = WG.get_edgelist()
    for e in edge_all:  # 剔除自环边
        if e[0] == e[1]: edge_all.remove(e)
    Q_info = {}
    Q_info['n'] = n

    G_info['edge_all'], G_info['node_nei_info'], G_info['adj'], G_info['HEW_adj'], G_info['short_path_adj'],  G_info['average_degrees'] = edge_all, node_nei_info, adj, HEW_adj, short_path_adj, average_degrees
    return G_info,Q_info


# =============================================================================
# 真实社区划分获取
# =============================================================================
def real_cd_obtain(groundtruth_path):
    # 初始化NMi
    # 获取真实社区划分列表
    real_mem = []
    file_flag = os.path.exists(groundtruth_path)
    if file_flag:
        with open(groundtruth_path, mode='r', encoding='UTF-8') as f:
            real_mem = list(map(int, f.read().splitlines()))
            if 0 not in real_mem: real_mem = [i - 1 for i in real_mem]
    return real_mem

# =============================================================================
# 将加权图社区划分转换为超图社区划分
# =============================================================================
def WGCD_to_HGCD(HG_info, WG_info, membership_c, FMHMO_params):
    # 转化成partition
    partition = {}
    for cno in set(membership_c): partition[cno] = []
    for node, icno in enumerate(membership_c): partition[icno].append(node)

    prtition_real = {}
    for cno in partition.keys():
        icno_list = partition[cno]
        prtition_real[cno] = []
        for i in icno_list: prtition_real[cno].extend(list(HG_info['no_he_dict'][i]))
        prtition_real[cno] = list(set(prtition_real[cno]))

    # 求社区间节点
    X, X_NCM, X_NCR = HG_func.NCR_NCM(membership_c, partition, prtition_real, HG_info, WG_info, FMHMO_params, verbose=True)


    ## 社区划分
    # 重置社区标号（将社区标号重置为0-c的标号）X
    X_membership_c = list(X)
    c_len = len(set(X))
    Xcno_cno = dict(zip(list(set(X_membership_c)), [i for i in range(c_len)]))
    for i, cno in enumerate(X_membership_c): X_membership_c[i] = Xcno_cno[cno]

    X_commnuity_partition = {}
    for cno in set(X_membership_c): X_commnuity_partition[cno] = []
    for node, icno in enumerate(X_membership_c): X_commnuity_partition[icno].append(node)

    # 重置社区标号（将社区标号重置为0-c的标号）X_NCM
    X_NCM_membership_c = list(X_NCM)
    X_NCM_c_len = len(set(X_NCM))
    X_NCM_cno = dict(zip(list(set(X_NCM_membership_c)), [i for i in range(X_NCM_c_len)]))
    for i, cno in enumerate(X_NCM_membership_c): X_NCM_membership_c[i] = X_NCM_cno[cno]

    X_NCM_commnuity_partition = {}
    for cno in set(X_NCM_membership_c): X_NCM_commnuity_partition[cno] = []
    for node, icno in enumerate(X_NCM_membership_c): X_NCM_commnuity_partition[icno].append(node)

    # 重置社区标号（将社区标号重置为0-c的标号）X_NCR
    X_NCR_membership_c = list(X_NCR)
    X_NCR_c_len = len(set(X_NCR))
    X_NCR_cno = dict(zip(list(set(X_NCR_membership_c)), [i for i in range(X_NCR_c_len)]))
    for i, cno in enumerate(X_NCR_membership_c): X_NCR_membership_c[i] = X_NCR_cno[cno]

    X_NCR_commnuity_partition = {}
    for cno in set(X_NCR_membership_c): X_NCR_commnuity_partition[cno] = []
    for node, icno in enumerate(X_NCR_membership_c): X_NCR_commnuity_partition[icno].append(node)

    return X_membership_c,X_commnuity_partition,X_NCM_membership_c,X_NCM_commnuity_partition,X_NCR_membership_c,X_NCR_commnuity_partition

