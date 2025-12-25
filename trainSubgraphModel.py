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

# 引入子图异常检测模型
from models.gnn import BWGNN, BernNet, GCN, GIN, GraphSAGE
from tqdm import tqdm
# 引入数据预处理
from datasets.graphDataset import GraphDataset
from utils import preprocess, preprocessDataset, collate_fn, processGenerateSubgraph
# 引入gae
from gae.encoder import GCNEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE
# 引入BAED
from generateData import GraphDataGenerator, GenerateDataArgs


import torch.multiprocessing
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, classification_report
import random
import copy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sample_losses = []



def load_dataset(dataset_path):
    """
    Load the dataset from the given path.

    Parameters:
    dataset_path (str): The path to the dataset file.

    Returns:
    list: The loaded dataset.
    """
    with open(dataset_path, 'rb') as file:
        nx_graphs = pickle.load(file)
    return nx_graphs


def split_dataset(nx_graphs, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.03):
    """
    Split the dataset into training, evaluation, and test sets.

    Parameters:
    nx_graphs (list): The dataset to be split.
    train_ratio (float): The ratio of the training set.
    eval_ratio (float): The ratio of the evaluation set.
    test_ratio (float): The ratio of the test set.

    Returns:
    tuple: A tuple containing the training, evaluation, and test sets.
    """
    total_length = len(nx_graphs)
    train_size = int(train_ratio * total_length)
    eval_size = int(eval_ratio * total_length)
    test_size = eval_size  # Assuming test_ratio is equal to eval_ratio

    train_nx_graphs = nx_graphs[:train_size]
    eval_nx_graphs = nx_graphs[train_size:train_size + eval_size]
    test_nx_graphs = nx_graphs[train_size + eval_size:train_size + eval_size + test_size]

    return train_nx_graphs, eval_nx_graphs, test_nx_graphs


def create_dataloaders(train_set, eval_set, test_set, batch_size, num_workers, pin_memory, collate_fn):
    """
    Create DataLoader objects for training, evaluation, and test sets.

    Parameters:
    train_set (Dataset): The training dataset.
    eval_set (Dataset): The evaluation dataset.
    test_set (Dataset): The test dataset.
    batch_size (int): The batch size for the DataLoader.
    collate_fn (function): The collate function to be used by the DataLoader.

    Returns:
    tuple: A tuple containing the DataLoader objects for training, evaluation, and test sets.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False, collate_fn=collate_fn)

    return train_loader, eval_loader, test_loader


def construct_graph(feat, edge_index):
    # print("feat:",feat.shape)
    # print("edge_index:",edge_index)
    """
    Construct a graph using networkx from node features and edge indices.

    Parameters:
    feat (torch.Tensor): Node features.
    edge_index (torch.Tensor): Edge indices.

    Returns:
    nx.Graph: The constructed graph.
    """
    # 获取所有唯一节点索引
    unique_nodes = torch.unique(edge_index)
    # 提取相应的节点特征
    unique_feat = feat[unique_nodes]
    # 创建 DGL 图
    graph = dgl.graph((edge_index[0], edge_index[1]))
    # 添加节点特征和标签
    graph.ndata['feature'] = unique_feat.to(device)
    return graph


def test_subgraph_anomaly_detector(detector, test_loader, device, model_path):
    """
    Test a subgraph anomaly detection model on the test set and return evaluation metrics.
    """
    detector.load_state_dict(torch.load(model_path, map_location=device))
    detector.to(device)
    detector.eval()

    all_labels = []
    all_probs = []      # 正类 (anomaly) 概率
    all_predicted = []

    with torch.no_grad():
        for data in test_loader:
            if not data:
                continue
            feats, labels = data.subFeature, data.sub_label
            edges_per_subgraph = data.sub_edge_index

            for i in range(len(feats)):
                feat = feats[i].to(device)
                edge_index = edges_per_subgraph[i].to(device)
                label = labels[i].item()

                if feat.shape[0] == 0:
                    continue

                graph = construct_graph(feat, edge_index)
                if graph.ndata['feature'].shape[0] == 0:
                    continue

                output = detector(graph)
                prob = F.softmax(output, dim=0).cpu().numpy()
                pred = int(np.argmax(prob))

                all_labels.append(label)
                all_probs.append(prob[1])  # P(anomaly)
                all_predicted.append(pred)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predicted = np.array(all_predicted)

    accuracy = np.mean(all_predicted == all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    metrics = {'accuracy': accuracy}

    # Only compute AUC metrics if both classes present
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        report = classification_report(all_labels, all_predicted, digits=5, target_names=["正常", "异常"])

        metrics.update({
            'auroc': auroc,
            'auprc': auprc,
            'classification_report': report
        })

        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print("Classification Report:\n", report)
    else:
        print("Warning: Only one class in test set. Skipping AUC metrics.")

    return metrics

# 训练模型
def train_subgraph_anomaly_detector(detector, generator, gae, train_loader, eval_loader, test_loader, num_epochs,
                                    optimizer, device, dataset, use_baed=True,withcondition=True):
    """ Train a subgraph anomaly detection model.
    Parameters:
    detector (torch.nn.Module): The anomaly detection model.
    generator (object): The generator for generating negative samples.
    gae (torch.nn.Module): The graph autoencoder model.
    train_loader (DataLoader): The DataLoader for the training set.
    num_epochs (int): The number of epochs to train the model.
    optimizer (torch.optim.Optimizer): The optimizer for training.
    device (torch.device): The device to run the training on.
    use_baed (bool): Whether to use BAED for generating negative samples.

    Returns:
    None
    """
    best_avg_error = float('inf')
    if use_baed:
        generatorName = "BAED"
    else:
        generatorName = "None"
    model_path = f"./subgraphWeight/{dataset}_{detector.modelName}_{generatorName}.pth"
    # 训练集训练+验证集验证

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        epoch_loss = 0.0
        num_samples = 0
        detector.train()
        batch_num = 0

        for data in tqdm(train_loader, leave=False):
            # print("data:", data)
            batch_sample_loss = []
            if data:
                feats, labels = data.subFeature, data.sub_label
                # print("labels:",labels)
                edges_per_subgraph = data.sub_edge_index
                num_normal = (labels == 0).sum().item()
                num_anomalies = (labels == 1).sum().item()
                print("difference:", num_normal - num_anomalies)
                try:
                    if use_baed and num_anomalies < num_normal:
                        if withcondition:
                            difference = num_normal - num_anomalies
                            indices = torch.where(labels == 1)
                            indices_list = indices[0].tolist() if len(indices) else []
                            abnormal_subgraph_feats = [feats[index] for index in indices_list]
                            abnormal_edge = [edges_per_subgraph[index] for index in indices_list]
                            abnormal_subgraph_loss = [sample_losses[batch_num][index] if epoch != 0 else 1 for index in
                                                            indices_list]

                            sub_node_embedding_list = []
                            if abnormal_subgraph_feats:
                                for i in range(len(abnormal_subgraph_feats)):
                                    item_feat = abnormal_subgraph_feats[i].to(device)
                                    item_edge_index = abnormal_edge[i].to(device)
                                    item_loss = abnormal_subgraph_loss[i]
                                    total_sub_node_embedding = gae.encode(item_feat.to(dtype=torch.float), item_edge_index)
                                    max_pooled = torch.max(total_sub_node_embedding, dim=0).values
                                    sub_node_embedding_list.append(max_pooled)
                                weighted_node_embedding = torch.zeros_like(sub_node_embedding_list[0])
                                for i in range(len(sub_node_embedding_list)):
                                    weight = abnormal_subgraph_loss[i] / sum(abnormal_subgraph_loss)
                                    weighted_node_embedding += weight * sub_node_embedding_list[i]
                            else:
                                weighted_node_embedding = None
                        else:
                            weighted_node_embedding = None
                                    # print("weighted_node_embedding:",weighted_node_embedding)
                        print("需要合成{}个负样本数据".format(difference))
                        if difference >= 64:
                            batch_generate_size = 64
                            total = difference
                            num_batches = (total + batch_generate_size - 1) // batch_generate_size
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_generate_size
                                end_idx = min((batch_idx + 1) * batch_generate_size, total)
                                batch_difference = end_idx - start_idx
                                generated_nxgraphs = generator.generate_data(batch_difference,
                                                                                    embedding=weighted_node_embedding)
                                generate_feat, generate_edge_index = processGenerateSubgraph(generated_nxgraphs,
                                                                                                    feats[0].size(1))
                                feats.extend(generate_feat)
                                edges_per_subgraph.extend(generate_edge_index)
                        elif difference > 0 and difference < 64:
                            generated_nxgraphs = generator.generate_data(difference, embedding=weighted_node_embedding)
                            generate_feat, generate_edge_index = processGenerateSubgraph(generated_nxgraphs,
                                                                                                feats[0].size(1))
                            feats.extend(generate_feat)
                            edges_per_subgraph.extend(generate_edge_index)
                        new_labels = torch.ones(len(generate_feat), dtype=torch.long)
                        labels = torch.cat((labels, new_labels))
                except Exception as e:
                    print("error:", e)
                    continue
                optimizer.zero_grad()
                for i in range(len(feats)):
                    try:
                        feat = feats[i].to(device)
                        edge_index = edges_per_subgraph[i].to(device)
                        # print("feat:",feat.shape)
                        # print("feat:",feat)
                        # if feat.size(0)==0:
                        #     batch_sample_loss.append(0)
                        #     num_samples += 1
                        #     continue
                        if feat.shape[0] == 0:
                            continue  # 跳过该样本
                        graph = construct_graph(feat, edge_index)
                        # print("graph:", graph.ndata['feature'].shape)
                        if graph.ndata['feature'].shape[0] == 0:
                            continue
                        output = detector(graph)
                        # print(f"Output shape: {output.shape}")
                        if output.shape[0] == 0:
                            continue
                        if i >= len(labels):
                            continue
                        item_label = torch.tensor([labels[i].item()], device=device)
                        # 计算交叉熵损失
                        loss = F.cross_entropy(output.unsqueeze(0), item_label)
                        batch_sample_loss.append(loss.item())
                        epoch_loss += loss.item()
                        num_samples += 1
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print("error:",e)
                        batch_sample_loss.append(0)
                        num_samples += 1
                        continue
                sample_losses[batch_num] = batch_sample_loss
                batch_sample_loss = []
                batch_num += 1
        avg_error = epoch_loss / num_samples
        if avg_error < best_avg_error:
            best_epoch = epoch
            best_avg_error = avg_error
            torch.save(detector.state_dict(), model_path)
        print(f"Epoch {epoch + 1}: Average Error = {avg_error}")

        # # 验证集验证
        # eval_loss = 0.0
        # num_eval_samples = 0
        # # 开启评估模式
        # detector.eval()
        # for data in tqdm(eval_loader,leave=False):
        #     if data:
        #         feats, labels = data.subFeature, data.sub_label
        #         edges_per_subgraph = data.sub_edge_index
        #         # Iterate over each sample
        #         for i in range(len(feats)):
        #             feat = feats[i].to(device)
        #             edge_index = edges_per_subgraph[i].to(device)
        #             embedding = gae.encode(feat, edge_index)
        #             output = detector(embedding)
        #             # Calculate the loss for the sample
        #             loss = F.cross_entropy(output, labels[i].unsqueeze(0))
        #             eval_loss += loss.item()
        #             num_eval_samples += 1

        # # # Calculate the average loss for the eval loop
        # avg_eval_loss = eval_loss / num_eval_samples
        # print(f"Epoch {epoch+1}: Validation Loss = {avg_eval_loss}")
        # 测试集测试：
        test_subgraph_anomaly_detector(detector, test_loader, device, model_path=model_path)


if __name__ == "__main__":
    # 修改数据集
    dataset = "reddit"
    gae = torch.load(f'./gae/weight/{dataset}_gae_relu_128.pt')
    dataset_path = f"./graphs/{dataset}_init_0_1_dataset.pkl"
    nx_graphs = load_dataset(dataset_path)
    print("数据集样本总数为:", len(nx_graphs))
    train_nx_graphs, eval_nx_graphs, test_nx_graphs = split_dataset(nx_graphs)
    # 子图样本集预处理，并构建dataset
    train_pygraphs, eval_pygraphs, test_pygraphs = preprocessDataset(train_nx_graphs), preprocessDataset(eval_nx_graphs), preprocessDataset(test_nx_graphs)

    # 初始化计数器和索引列表
    label_0_count = 0
    label_1_count = 0
    label_1_indices = []

    # 遍历 train_pygraphs 提取标签并统计数量
    for idx, pyg_data in enumerate(train_pygraphs):
        if pyg_data.sub_label == 0:
            label_0_count += 1
        elif pyg_data.sub_label == 1:
            label_1_count += 1
            label_1_indices.append(idx)

    # 打印标签数量
    print(f"标签为 0 的样本数量: {label_0_count}")
    print(f"标签为 1 的样本数量: {label_1_count}")

    print("训练集样本数:", len(train_pygraphs))
    print("验证集样本数:", len(eval_pygraphs))
    print("测试集样本数:", len(test_pygraphs))
    # 构建dataset
    train_set = GraphDataset(train_pygraphs)
    eval_set = GraphDataset(eval_pygraphs)
    test_set = GraphDataset(test_pygraphs)

    # 修改：设置GPU训练相关参数
    batch_size, num_workers, pin_memory = 128, 0, True

    # 构建dataloader
    train_loader, eval_loader, test_loader = create_dataloaders(train_set, eval_set, test_set, batch_size, num_workers,
                                                                pin_memory, collate_fn)

    # 修改节点特征数目
    nodes_feats = 64
    embedding_dim = 128
    # 修改构建模型训练参数
    num_epochs = 10
    learning_rate = 0.0001

    # 修改构建模型
    subModelName = "GraphSAGE"
    if subModelName == "BWGNN":
        detector = BWGNN(in_feats=nodes_feats,activation='Sigmoid')
    elif subModelName == "BernNet":
        detector = BernNet(in_feats=nodes_feats,activation='Sigmoid')
    # 报错
    elif subModelName == "AMNet":
        detector = AMNet(in_feats=nodes_feats)
    elif subModelName == "GCN":
        detector = GCN(in_feats=nodes_feats)
    elif subModelName == "GIN":
        detector = GIN(in_feats=nodes_feats)
    elif subModelName == "GraphSAGE":
        detector = GraphSAGE(in_feats=nodes_feats)
    detector = detector.to(device)
    # 构建优化器
    optimizer = optim.Adam(detector.parameters(), lr=learning_rate)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 初始化gae模型
    # gae = torch.load(f./gae/weight/{dataset}_gae_relu_256.pt')
    use_baed = False
    withcondition=False
    # 初始化BAED
    if use_baed:
        args = GenerateDataArgs(run_name='2024-12-06_17-50-36', dataset='elliptic', num_samples=64, seed=0, checkpoint=50)
        generator = GraphDataGenerator(args)
    else:
        generator = None
    # 训练模型
    print("epoch:{};lr:{}".format(num_epochs, learning_rate))
    sample_losses = [[] for i in range(len(train_set) // batch_size + 1)]
    train_subgraph_anomaly_detector(detector, generator, gae, train_loader, eval_loader, test_loader, num_epochs,
                                    optimizer, device, dataset, use_baed=use_baed,withcondition=withcondition)

