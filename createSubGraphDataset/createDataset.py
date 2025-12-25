# -*- coding: utf-8 -*-
"""
生成子图数据集：以正常节点（label=0）或异常节点（label=1）为核心节点，
提取其二跳邻域子图，用于后续子图级异常检测任务。

支持的数据集包括：
- dgraph
- tfinance
- elliptic
- photo
- reddit
- Amazon

输出格式：NetworkX 图列表，每个子图带有 graph['sublabel'] 属性表示中心节点标签。
"""

import os
import math
import pickle as pkl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import torch
import networkx as nx
import dgl
from dgl.data.utils import load_graphs
import warnings
warnings.filterwarnings("ignore")

def preprocess_features(features):
    """对特征矩阵进行行归一化（Row-normalize），并转换为稠密矩阵。"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def load_data(data_dir, start_ts, end_ts):
    """
    加载 Elliptic Bitcoin 数据集，并按时间戳切分。
    
    Args:
        data_dir (str): 数据目录路径。
        start_ts (int): 起始时间戳（从1开始）。
        end_ts (int): 结束时间戳（不包含）。
    
    Returns:
        adj_mats (list): 每个时间戳对应的邻接矩阵（pandas DataFrame）。
        features_labelled_ts (list): 每个时间戳对应的带标签节点特征。
        classes_ts (list): 每个时间戳对应的节点标签（'1'=非法, '2'=合法, 'unknown'=未标注）。
    """
    classes_csv = 'elliptic_txs_classes.csv'
    edgelist_csv = 'elliptic_txs_edgelist.csv'
    features_csv = 'elliptic_txs_features.csv'

    classes = pd.read_csv(os.path.join(data_dir, classes_csv), index_col='txId')
    edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv), index_col='txId1')
    features = pd.read_csv(os.path.join(data_dir, features_csv), header=None, index_col=0)

    # 仅保留有标签的交易（排除 'unknown'）
    labelled_classes = classes[classes['class'] != 'unknown']
    labelled_tx = set(labelled_classes.index)
    total_tx = list(classes.index)

    adj_mats, features_labelled_ts, classes_ts = [], [], []

    for ts in range(start_ts, end_ts):
        # 提取当前时间戳的交易
        features_ts = features[features[1] == ts + 1]
        tx_ts = set(features_ts.index)
        labelled_tx_ts = [tx for tx in tx_ts if tx in labelled_tx]

        # 构建当前时间戳的邻接矩阵（仅含带标签节点）
        adj_mat = pd.DataFrame(np.zeros((len(total_tx), len(total_tx))), index=total_tx, columns=total_tx)
        edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]

        for i in range(edgelist_labelled_ts.shape[0]):
            src = edgelist_labelled_ts.index[i]
            dst = edgelist_labelled_ts.iloc[i]['txId2']
            if dst in adj_mat.columns:
                adj_mat.loc[src, dst] = 1

        # 限制到带标签节点子集
        adj_mat_ts = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
        features_l_ts = features.loc[labelled_tx_ts]
        classes_l_ts = classes.loc[labelled_tx_ts]

        adj_mats.append(adj_mat_ts)
        features_labelled_ts.append(features_l_ts)
        classes_ts.append(classes_l_ts)

    return adj_mats, features_labelled_ts, classes_ts


def adj_to_dgl_graph(adj):
    """将 SciPy 稀疏邻接矩阵转换为 DGL 图。"""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    return dgl.from_networkx(nx_graph)


def initOriginDataset(dataset, label_type):
    """
    为指定数据集生成子图数据集。
    
    Args:
        dataset (str): 数据集名称（如 'dgraph', 'elliptic' 等）。
        label_type (str): 子图中心节点类型：
            - 'label_1': 仅异常节点（label=1）
            - 'label_0_1': 正常（0）和异常（1）节点
    
    输出：保存为 .pkl 文件，内容为 NetworkX 子图列表，每个子图含 graph['sublabel']。
    """
    print(f"正在处理数据集: {dataset}, label_type: {label_type}")
    
    if dataset == 'dgraph':
        # 加载 DGraphFin 数据
        graphItem = np.load("../graphs/dgraphfin.npz")
        x = graphItem['x']
        y = graphItem['y']
        edge_index = graphItem['edge_index']

        src = torch.tensor(edge_index[:, 0], dtype=torch.int64)
        dst = torch.tensor(edge_index[:, 1], dtype=torch.int64)
        graph = dgl.graph((src, dst))
        graph.ndata['feature'] = torch.tensor(x, dtype=torch.float32)
        graph.ndata['label'] = torch.tensor(y, dtype=torch.int64)

        labels = graph.ndata['label']
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])

        if label_type == 'label_1':
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            target_nodes = torch.nonzero((labels == 0) | (labels == 1), as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_0_1_dataset.pkl'
        else:
            raise ValueError("label_type 必须是 'label_1' 或 'label_0_1'")

        nx_graphs = []
        for node in target_nodes:
            two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=2).keys())
            subgraph = nx.Graph(nx_graph.subgraph(two_hop_nodes).copy())
            subgraph.graph['sublabel'] = int(labels[node].item())
            nx_graphs.append(subgraph)

        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
        print(f"已保存: {file_name}")

    elif dataset == 'elliptic':
        data_dir = "../graphs/elliptic_bitcoin_dataset"
        adj_mats, features_labelled_ts, classes_ts = load_data(data_dir, 1, 49)

        nx_graphs = []
        for i in range(len(adj_mats)):
            adj_mat = adj_mats[i]
            features = features_labelled_ts[i]
            labels_df = classes_ts[i]

            if adj_mat.empty:
                continue

            # 构建 DGL 图
            src, dst = np.nonzero(adj_mat.values)
            src = torch.tensor(src, dtype=torch.int64)
            dst = torch.tensor(dst, dtype=torch.int64)
            graph = dgl.graph((src, dst))

            node_features = torch.tensor(features.values, dtype=torch.float32)
            node_labels = torch.tensor(
                labels_df['class'].apply(lambda x: 1 if x == '1' else 0).values,
                dtype=torch.int64
            )

            if len(node_features) == graph.num_nodes():
                graph.ndata['feature'] = node_features
                graph.ndata['label'] = node_labels
                nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
                labels = graph.ndata['label']

                if label_type == 'label_1':
                    target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
                elif label_type == 'label_0_1':
                    target_nodes = torch.nonzero((labels == 0) | (labels == 1), as_tuple=True)[0].tolist()
                else:
                    raise ValueError("label_type 必须是 'label_1' 或 'label_0_1'")

                for node in target_nodes:
                    two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=2).keys())
                    subgraph = nx.Graph(nx_graph.subgraph(two_hop_nodes).copy())
                    subgraph.graph['sublabel'] = int(labels[node].item())
                    nx_graphs.append(subgraph)

        file_name = f'../graphs/{dataset}_init_{"1" if label_type == "label_1" else "0_1"}_dataset.pkl'
        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
        print(f"已保存: {file_name}")

    elif dataset == 'tfinance':
        graph, _ = load_graphs('../graphs/tfinance')
        graph = graph[0]
        labels = graph.ndata['label'].argmax(1)  # one-hot 转为整数标签
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])

        if label_type == 'label_1':
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            file_name = '../graphs/tfinance_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            target_nodes = torch.nonzero((labels == 0) | (labels == 1), as_tuple=True)[0].tolist()
            file_name = '../graphs/tfinance_init_0_1_dataset.pkl'
        else:
            raise ValueError("label_type 必须是 'label_1' 或 'label_0_1'")

        nx_graphs = []
        for node in target_nodes:
            try:
                two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=1).keys())
                if len(two_hop_nodes) > 1200:
                    continue  # 过滤过大子图
                subgraph = nx.Graph(nx_graph.subgraph(two_hop_nodes).copy())
                subgraph.graph['sublabel'] = int(labels[node].item())
                nx_graphs.append(subgraph)
            except Exception as e:
                print(f"跳过节点 {node}: {e}")

        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
        print(f"已保存: {file_name}")

    elif dataset in ['photo', 'reddit', 'Amazon']:
        # 通用 MAT 格式数据集处理
        data = sio.loadmat(f"../graphs/{dataset}.mat")
        label = data.get('Label', data.get('gnd'))
        attr = data.get('Attributes', data.get('X'))
        network = data.get('Network', data.get('A'))

        adj = sp.csr_matrix(network)
        feat = sp.lil_matrix(attr)
        ano_labels = np.squeeze(np.array(label))

        # 特征预处理（行归一化）
        features = preprocess_features(feat)

        # 构建 DGL 图
        graph = adj_to_dgl_graph(adj)
        graph.ndata["feature"] = torch.tensor(features, dtype=torch.float32)
        graph.ndata["label"] = torch.tensor(ano_labels, dtype=torch.int64)

        labels = graph.ndata['label']
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])

        if label_type == 'label_1':
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            target_nodes = torch.nonzero((labels == 0) | (labels == 1), as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_0_1_dataset.pkl'
        else:
            raise ValueError("label_type 必须是 'label_1' 或 'label_0_1'")

        nx_graphs = []
        for node in target_nodes:
            two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=1).keys())
            subgraph = nx.Graph(nx_graph.subgraph(two_hop_nodes).copy())
            subgraph.graph['sublabel'] = int(labels[node].item())
            nx_graphs.append(subgraph)

        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
        print(f"已保存: {file_name}")

    else:
        raise ValueError(f"不支持的数据集: {dataset}")



initOriginDataset('dgraph', 'label_1')
initOriginDataset('dgraph', 'label_0_1')
initOriginDataset('elliptic', 'label_1')
initOriginDataset('elliptic', 'label_0_1')
initOriginDataset('tfinance', 'label_1')
initOriginDataset('tfinance', 'label_0_1')
initOriginDataset('photo', 'label_1')
initOriginDataset('photo', 'label_0_1')
initOriginDataset('reddit', 'label_1')
initOriginDataset('reddit', 'label_0_1')
initOriginDataset('Amazon', 'label_1')
initOriginDataset('Amazon', 'label_0_1')