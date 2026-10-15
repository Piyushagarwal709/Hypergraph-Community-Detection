# 1.创建输入特征X，X->n * m, n个节点，m个特征(m等于真实分区的个数）
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import optim, nn
import os

# 在导入scikit-learn之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'  # 可以根据你的CPU核心数调整这个值

from models.HGNN import HGNN


def X_Feature_get(HG_info, real_mem, feature_noise=0.1):
    num_nodes = HG_info['H_n']
    num_features = len(set(real_mem))

    features = np.zeros((num_nodes, num_features))
    labels = np.zeros((num_nodes, num_features))

    features = np.random.normal(features, feature_noise, features.shape)

    if 0 in real_mem:
        labels[np.arange(num_nodes), np.array(real_mem)] = 1
    else:
        labels[np.arange(num_nodes), np.array(real_mem) - 1] = 1

    X_features = torch.FloatTensor(features)
    X_labels = torch.FloatTensor(labels)
    return X_features, X_labels


# 2.创建HNN网络

# 3.定义Kmeans聚类算法
import numpy as np
import torch
from torch import optim, nn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import copy

import numpy as np
import torch
from torch import optim, nn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib.pyplot as plt


def HNN_Kmeans(X_features, X_labels, WG, real_mem=None, epochs=200, patience=10, lr=0.01,
               dropout=0.3, n_hid=32, use_spectral=False, spectral_n_init=10,
               kmeans_n_init=30, scale_features=False, min_clusters=2, max_clusters=None,
               metric='silhouette'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """
    超图神经网络结合聚类算法进行节点分区，自动确定最佳聚类数目

    参数:
        X_features: 节点特征矩阵
        WG: 超图权重矩阵
        real_mem: 真实分区（用于评估，可选）
        epochs: 训练轮数
        patience: 早停耐心值
        lr: 学习率
        dropout: Dropout率
        n_hid: 隐藏层维度
        use_spectral: 是否使用谱聚类替代K-means
        spectral_n_init: 谱聚类初始化次数
        kmeans_n_init: K-means初始化次数
        scale_features: 是否标准化特征
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数（默认为特征维度）
        metric: 选择最佳聚类数的指标 ('silhouette', 'calinski', 'davies')
        plot_metrics: 是否绘制评估指标随聚类数变化的曲线

    返回:
        best_labels: 最佳聚类标签
        partition: 分区字典
        best_n_clusters: 最佳聚类数目
        best_metric_value: 最佳评估指标值
        evaluation_metrics: 评估指标
        best_embedding: 最佳嵌入表示
    """
    # 1. 特征标准化（可选）
    if scale_features:
        scaler = StandardScaler()
        X_features_np = X_features.detach().cpu().numpy()
        X_features_scaled = scaler.fit_transform(X_features_np)
        X_features = torch.FloatTensor(X_features_scaled)

    # 2. 初始化模型
    if max_clusters is None:
        max_clusters = min(X_features.shape[0] // 2, X_features.shape[1] * 2)  # 默认最大聚类数
    max_clusters = max(min_clusters + 1, max_clusters)  # 确保最大聚类数大于最小聚类数

    HGNN_model = HGNN(
        in_ch=X_features.shape[1],
        out_ch=X_features.shape[1],  # 模型输出维度设为最大可能的聚类数
        n_hid=n_hid,
        dropout=dropout
    ).to(device)

    WG = torch.Tensor(WG).to(device)
    X_features = torch.Tensor(X_features).to(device)
    X_labels = torch.Tensor(X_labels).to(device)

    # 3. 优化器和损失函数
    optimizer = optim.Adam(HGNN_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()  # 使用MSE损失

    # 4. 训练模型
    HGNN_model.train()
    best_model = None
    best_loss = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        output = HGNN_model(X_features, WG)

        # 计算损失（重构损失）
        loss = criterion(output, X_labels)
        # print('Epoch: {}/{}...'.format(epoch + 1, loss))

        # 反向传播
        loss.backward()
        optimizer.step()

        # 保存最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(HGNN_model)

    # 5. 使用最佳模型获取嵌入表示
    best_model.eval()
    with torch.no_grad():
        best_embedding = best_model(X_features, WG).detach().cpu().numpy()

    # best_embedding = X_features.detach().cpu().numpy()

    # 6. 评估不同聚类数的效果
    metrics = {}
    best_n_clusters = min_clusters
    best_metric_value = -np.inf if metric != 'davies' else np.inf  # Davies指标越小越好

    for n_clusters in range(min_clusters, max_clusters + 1):
        # 聚类
        if use_spectral:
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                assign_labels='kmeans',
                random_state=42,
                n_init=spectral_n_init
            )
            labels = clustering.fit_predict(best_embedding)
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=kmeans_n_init
            )
            labels = kmeans.fit_predict(best_embedding)

        # 计算评估指标
        if len(set(labels)) > 1:  # 确保至少有2个聚类
            if metric == 'silhouette':
                score = silhouette_score(best_embedding, labels)
            elif metric == 'calinski':
                score = calinski_harabasz_score(best_embedding, labels)
            elif metric == 'davies':
                score = -davies_bouldin_score(best_embedding, labels)  # 取负值使其越大越好
            else:
                raise ValueError(f"不支持的评估指标: {metric}")

            metrics[n_clusters] = score

            # 更新最佳聚类数
            if (metric != 'davies' and score > best_metric_value) or \
                    (metric == 'davies' and score < -best_metric_value):  # 因为取了负值
                best_metric_value = score if metric != 'davies' else -score
                best_n_clusters = n_clusters

    # 7. 使用最佳聚类数进行最终聚类
    # best_n_clusters = X_features.shape[1]
    if use_spectral:
        clustering = SpectralClustering(
            n_clusters=best_n_clusters,
            assign_labels='kmeans',
            random_state=42,
            n_init=spectral_n_init
        )
        best_labels = clustering.fit_predict(best_embedding)
    else:
        kmeans = KMeans(
            n_clusters=best_n_clusters,
            random_state=42,
            n_init=kmeans_n_init
        )
        best_labels = kmeans.fit_predict(best_embedding)

    # 8. 构建分区字典
    partition = {}
    for cno in set(best_labels):
        partition[cno] = []
    for node, icno in enumerate(best_labels):
        partition[icno].append(node)

    return best_labels, partition

# def HNN_Kmeans(X_features, WG, real_mem=None, epochs=100, patience=10, lr=0.01,
#                dropout=0.3, n_hid=16, use_spectral=False, spectral_n_init=10,
#                kmeans_n_init=30, scale_features=True):
#     """
#     超图神经网络结合聚类算法进行节点分区
#
#     参数:
#         X_features: 节点特征矩阵
#         WG: 超图权重矩阵
#         real_mem: 真实分区（用于评估，可选）
#         epochs: 训练轮数
#         patience: 早停耐心值
#         lr: 学习率
#         dropout: Dropout率
#         n_hid: 隐藏层维度
#         use_spectral: 是否使用谱聚类替代K-means
#         spectral_n_init: 谱聚类初始化次数
#         kmeans_n_init: K-means初始化次数
#         scale_features: 是否标准化特征
#
#     返回:
#         best_labels: 最佳聚类标签
#         partition: 分区字典
#         best_silhouette: 最佳轮廓系数
#         evaluation_metrics: 评估指标
#         best_embedding: 最佳嵌入表示
#     """
#     # 1. 特征标准化（可选）
#     if scale_features:
#         scaler = StandardScaler()
#         X_features_np = X_features.detach().cpu().numpy()
#         X_features_scaled = scaler.fit_transform(X_features_np)
#         X_features = torch.FloatTensor(X_features_scaled)
#
#     # 2. 初始化模型
#     n_clusters = X_features.shape[1]
#     HGNN_model = HGNN(
#         in_ch=X_features.shape[1],
#         n_cluster=n_clusters,
#         n_hid=n_hid,
#         dropout=dropout
#     )
#
#     # 3. 优化器和损失函数
#     optimizer = optim.Adam(HGNN_model.parameters(), lr=lr, weight_decay=1e-5)
#     criterion = nn.MSELoss()  # 使用MSE损失
#
#     # 4. 训练模型
#     HGNN_model.train()
#     best_model = None
#     best_silhouette = -1
#     best_embedding = None
#     best_labels = None
#     patience_counter = 0
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#
#         # 前向传播
#         output = HGNN_model(X_features, WG)
#
#         # 计算损失（重构损失）
#         loss = criterion(output, X_features)
#
#         # 反向传播
#         loss.backward()
#         optimizer.step()
#
#         # 验证（每10个epoch）
#         if (epoch + 1) % 10 == 0:
#             HGNN_model.eval()
#             with torch.no_grad():
#                 output_val = HGNN_model(X_features, WG)
#                 output_np = output_val.detach().cpu().numpy()
#
#                 # 聚类
#                 if use_spectral:
#                     clustering = SpectralClustering(
#                         n_clusters=n_clusters,
#                         assign_labels='kmeans',
#                         random_state=42,
#                         n_init=spectral_n_init
#                     )
#                     labels = clustering.fit_predict(output_np)
#                 else:
#                     kmeans = KMeans(
#                         n_clusters=n_clusters,
#                         random_state=42,
#                         n_init=kmeans_n_init
#                     )
#                     labels = kmeans.fit_predict(output_np)
#
#                 # 计算轮廓系数（内部评估）
#                 if len(set(labels)) > 1:  # 确保至少有2个聚类
#                     silhouette = silhouette_score(output_np, labels)
#
#                     # 保存最佳模型
#                     if silhouette > best_silhouette:
#                         best_silhouette = silhouette
#                         best_model = copy.deepcopy(HGNN_model)
#                         best_embedding = output_np
#                         best_labels = labels
#                         patience_counter = 0
#                     else:
#                         patience_counter += 1
#
#                 # 早停
#                 if patience_counter >= patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     break
#
#             HGNN_model.train()
#
#     # 5. 使用最佳模型进行最终预测
#     if best_model is None:
#         best_model = HGNN_model
#
#     best_model.eval()
#     with torch.no_grad():
#         output_final = best_model(X_features, WG)
#         output_np = output_final.detach().cpu().numpy()
#
#         # 最终聚类
#         if use_spectral:
#             clustering = SpectralClustering(
#                 n_clusters=n_clusters,
#                 assign_labels='kmeans',
#                 random_state=42,
#                 n_init=spectral_n_init
#             )
#             best_labels = clustering.fit_predict(output_np)
#         else:
#             kmeans = KMeans(
#                 n_clusters=n_clusters,
#                 random_state=42,
#                 n_init=kmeans_n_init
#             )
#             best_labels = kmeans.fit_predict(output_np)
#
#     # 6. 构建分区字典
#     partition = {}
#     for cno in set(best_labels):
#         partition[cno] = []
#     for node, icno in enumerate(best_labels):
#         partition[icno].append(node)
#
#     # # 7. 计算评估指标（如果有真实标签）
#     # evaluation_metrics = {}
#     # if real_mem is not None:
#     #     from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
#     #     evaluation_metrics['ARI'] = adjusted_rand_score(real_mem, best_labels)
#     #     evaluation_metrics['NMI'] = normalized_mutual_info_score(real_mem, best_labels)
#
#     return best_labels, partition

#
