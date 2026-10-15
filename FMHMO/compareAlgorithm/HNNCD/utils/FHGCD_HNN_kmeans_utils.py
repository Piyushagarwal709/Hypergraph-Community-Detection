import numpy as np


def build_hypergraph_incidence_matrix(nodes, hyperedges):
    """
    构建超图映射矩阵（关联矩阵）

    参数:
        nodes: 节点列表，如 [0, 1, 2, 3, 4]
        hyperedges: 超边列表，每个超边是一个包含节点的列表，如 [[0, 1], [1, 2, 3], [3, 4]]

    返回:
        超图映射矩阵，形状为 (节点数, 超边数)
    """
    n = nodes  # 节点数
    m = len(hyperedges)  # 超边数

    # 初始化全零矩阵
    B = np.zeros((n, m), dtype=int)

    # 填充矩阵
    for j, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            # 找到节点在节点列表中的索引
            # i = nodes.index(node)
            B[node, j] = 1

    return B