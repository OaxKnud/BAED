
# 训练子图异常检测模型的代码
import numpy as np
import math
import torch
import os 
import networkx as nx
import numpy as np
import dgl
from dgl.data.utils import load_graphs, save_graphs
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
from torch_geometric.datasets import QM9
from sklearn.model_selection import train_test_split
import pandas as pd
import torch_geometric.data
import pickle

from torch_geometric.utils import to_dense_adj
from torch_geometric import transforms as T
from torch.nn import functional as F
import torch.optim as optim


import threading
import time

# 定义一个函数包装器，用于检测函数的执行时间
class TimeoutWrapper:
    def __init__(self, func, timeout):
        self.func = func
        self.timeout = timeout
        self.result = None

    def _run_func(self, *args, **kwargs):
        try:
            self.result = self.func(*args, **kwargs)
        except Exception as e:
            self.result = e

    def __call__(self, *args, **kwargs):
        thread = threading.Thread(target=self._run_func, args=args, kwargs=kwargs)
        thread.start()
        thread.join(self.timeout)  # 等待线程完成，最长时间为 self.timeout 秒

        if thread.is_alive():
            # 如果线程仍然在运行，则超时
            return 'timeout'
        return self.result

# 对于子图数据预处理，给pyg的图增加一些属性
@torch.no_grad()
def preprocess(g, degree=False):
    if isinstance(g, nx.Graph):
        # 这里面可能会存在一些自环，这样的话转换之后只会算一次
        pyg_data = pyg.utils.from_networkx(g)
        adj = torch.from_numpy(nx.to_numpy_array(g).astype(np.int32)).long()
    elif isinstance(g, pyg.data.Data):
        pyg_data = g
        adj = to_dense_adj(g.edge_index)[0].long()
        # raise NotImplementedError()
    #augmented_features:增加augmented_features的属性
    features=pyg_data.feature
    # 检查类型
    if isinstance(features, torch.Tensor):
        # 将 Tensor 转换为列表，每个元素为长度为 64 的 Tensor
        features_list = [features[i] for i in range(features.size(0))]
    features=features_list
    # print("features:",features.shape)
    labels=pyg_data.label
    # print("labels:",labels.shape)
    num_features = len(features[0])
    # 初始化 feature_map
    feature_map = {f'feature{i}': [] for i in range(num_features)}
    # 遍历 features 列表，提取每个索引位置的值
    for feature in features:
        for i in range(num_features):
            feature_map[f'feature{i}'].append(feature[i])
    # 将列表转换为 tensor
    for key in feature_map:
        feature_map[key] = torch.tensor(feature_map[key])
    # 打印结果
    feature_map["label"]=labels
    augmented_features=feature_map.keys()
    # print("augmented_features:",augmented_features)
    # 生成一个上三角矩阵的索引，并将其存储在pyg_data.full_edge_index中 
    row, col = torch.triu_indices(pyg_data.num_nodes, pyg_data.num_nodes,1)
    pyg_data.full_edge_index = torch.stack([row, col])
    pyg_data.full_edge_attr = adj[pyg_data.full_edge_index[0], pyg_data.full_edge_index[1]]
    pyg_data.sub_label=g.graph["sublabel"]
    # print("pyg_data.sub_label:",g.graph["sublabel"])
    if(len(pyg_data.full_edge_attr)==0):
        return None
    if not hasattr(pyg_data, 'node_attr'):
        pyg_data.node_attr = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
    if degree:
        pyg_data.degree = pyg.utils.degree(pyg_data.edge_index[0],num_nodes=pyg_data.num_nodes).long() # make sure edge_index is bi-directional
    for augmented_feature in augmented_features:
        # setattr(pyg_data, augmented_feature, FEATURE_EXTRACTOR[augmented_feature]['func'](pyg_data))
        setattr(pyg_data, augmented_feature, feature_map[augmented_feature])
    return pyg_data

# 修改 preprocessDataset 函数
@torch.no_grad()
def preprocessDataset(subgraph):
    pygraphs = []
    index = len(subgraph)
    # 包装 preprocess 函数，设置超时时间为 3 秒
    preprocess_with_timeout = TimeoutWrapper(preprocess, timeout=10)
    for nx_graph in subgraph:
        # print('index:', index)
        index -= 1
        try:
            # 调用包装后的函数
            pyg_data = preprocess_with_timeout(nx_graph, degree=False)
            if pyg_data == 'timeout':
                print("Preprocess function exceeded timeout for current graph.")
                continue
            if pyg_data:
                pygraphs.append(pyg_data)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            continue
        
    return pygraphs

@torch.no_grad()
def collate_fn(pyg_datas,repeat=1):
    # 你的原始collate_fn代码
    try:
        # pyg_datas = sum([[pyg_data.clone() for _ in range(repeat)] for pyg_data in pyg_datas], [])
        batched_data = pyg.data.Batch.from_data_list(pyg_datas)
        batch_feat=[]
        batch_edge_index=[]
        for pyg_data in pyg_datas:
            if isinstance(pyg_data.feature, torch.Tensor):
                batch_feat.append(pyg_data.feature)
            else:
                batch_feat.append(torch.stack(pyg_data.feature))
            batch_edge_index.append(pyg_data.edge_index)
        batched_data.subFeature = batch_feat
        batched_data.sub_edge_index = batch_edge_index
        batched_data.nodes_per_graph = torch.tensor([pyg_data.num_nodes for pyg_data in pyg_datas])
        batched_data.edges_per_graph = torch.tensor([pyg_data.num_nodes * (pyg_data.num_nodes - 1) // 2 for pyg_data in pyg_datas])
    except Exception as e:
        # print("collate+fn:",e)
        return None
    return batched_data


@torch.no_grad()
def processGenerateSubgraph(nx_graphs,feature_num):
    generate_feat=[]
    generate_edge_index=[]
    for nx_graph in nx_graphs:
        pyg_data = pyg.utils.from_networkx(nx_graph)
        # Extract all feature attributes
        features = []
        for i in range(feature_num):  # Assuming there are 17 features from feature0 to feature16
            feature_name = f'feature{i}'
            if hasattr(pyg_data, feature_name):
                features.append(getattr(pyg_data, feature_name))
        # print("features:",features)
        # Stack tensors horizontally to form a single tensor with 4 elements, each containing 17 features
        combined_features = torch.stack(features, dim=0).transpose(0, 1)
        generate_feat.append(combined_features)
        generate_edge_index.append(pyg_data.edge_index)
    return generate_feat,generate_edge_index
