# X_NCR_membership_c = list(X_NCR)
# X_NCR_c_len = len(set(X_NCR))
# X_NCR_cno = dict(zip(list(set(X_NCR_membership_c)), [i for i in range(X_NCR_c_len)]))
# for i, cno in enumerate(X_NCR_membership_c): X_NCR_membership_c[i] = X_NCR_cno[cno]
#
# X_NCR_commnuity_partition = {}
# for cno in set(X_NCR_membership_c): X_NCR_commnuity_partition[cno] = []
# for node, icno in enumerate(X_NCR_membership_c): X_NCR_commnuity_partition[icno].append(node)
import numpy as np
a = [0.49481292564235074, 0.8461538461538461,  0.43266601562500007]
b = [0.6618973275517651, 0.9230769230769231, 0.43266601562500007]
print(np.asarray(b)-np.asarray(a))

import numpy as np
from scipy.stats import entropy


def calculate_nmi(community1, community2):
    """
    手动计算两个社区划分之间的NMI

    参数:
    community1, community2: 社区划分列表，例如[0, 0, 1, 1, 2, 2]

    返回:
    NMI值
    """
    # 确保两个社区划分长度相同
    assert len(community1) == len(community2), "社区划分长度必须相同"

    n = len(community1)
    # 获取唯一的社区标签
    unique_c1 = np.unique(community1)
    unique_c2 = np.unique(community2)

    # 计算每个社区的大小
    sizes_c1 = np.array([np.sum(community1 == c) for c in unique_c1])
    sizes_c2 = np.array([np.sum(community2 == c) for c in unique_c2])

    # 计算互信息
    mi = 0.0
    for i, c1 in enumerate(unique_c1):
        for j, c2 in enumerate(unique_c2):
            # 计算两个社区的交集大小
            intersection = np.sum((community1 == c1) & (community2 == c2))
            if intersection == 0:
                continue

            # 计算联合概率和边缘概率
            p_ij = intersection / n
            p_i = sizes_c1[i] / n
            p_j = sizes_c2[j] / n

            # 累加互信息
            mi += p_ij * np.log2(p_ij / (p_i * p_j))

    # 计算熵
    h1 = entropy(sizes_c1 / n, base=2)
    h2 = entropy(sizes_c2 / n, base=2)

    # 计算归一化互信息
    nmi = 2 * mi / (h1 + h2)  # 使用算术平均归一化
    return nmi


# 使用示例
real_membership = [1, 5, 1, 3, 2, 2]
predicted_membership = [0, 0, 1, 1, 2, 2]
nmi = calculate_nmi(real_membership, predicted_membership)
print(f"NMI: {nmi}")
