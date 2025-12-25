import torch
import pickle
import argparse
from diffusion.utils import add_parent_path
import time
import os
import math
import networkx as nx
import torch_geometric as pyg

# Data
add_parent_path(level=1)
from datasets.data import get_data
# Model
from model import get_model
from gae.encoder import GCNEncoder
import random



class GenerateDataArgs:
    def __init__(self, run_name='2024-11-24_17-24-26', dataset='dgraph', num_samples=1, seed=0, checkpoint=1500):
        self.run_name = run_name
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self.checkpoint = checkpoint

class GraphDataGenerator:
    def __init__(self, args):
        torch.manual_seed(args.seed)
        self.args = args
        self.log_dir = f'./wandb/{args.dataset}/multinomial_diffusion/multistep/{args.run_name}'
        self.path_args = f'{self.log_dir}/args.pickle'
        self.path_check = f'{self.log_dir}/check/checkpoint_{args.checkpoint - 1}.pt'
        
        # Load args from file
        with open(self.path_args, 'rb') as f:
            self.args = pickle.load(f)
        
        self.args.device = 'cuda:0'
        self.train_loader, self.eval_loader, self.test_loader, self.num_node_feat, self.num_node_classes, self.num_edge_classes, self.max_degree, self.augmented_feature_dict, self.initial_graph_sampler, self.eval_evaluator, self.test_evaluator, self.monitoring_statistics = get_data(self.args)
        
        # Initialize model
        self.model = get_model(self.args, initial_graph_sampler=self.initial_graph_sampler)
        self.checkpoint = torch.load(self.path_check, map_location=self.args.device)
        self.model.load_state_dict(self.checkpoint['model'])
        
        if torch.cuda.is_available():
            self.model = self.model.to(self.args.device)
        self.model.eval()
    
    def generate_data(self, num_samples,embedding=None):
        # sample
        sampled_pygraph = self.model.sample(num_samples, embedding)
        pyg_datas = sampled_pygraph.to_data_list()
        # print("已生成pyg_datas:",len(pyg_datas))
        generated_nxgraphs = []
        for pyg_data in pyg_datas:
            feature_keys = [key for key in pyg_data.keys() if key.startswith('feature')]
            feature_keys.append("label")
            g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)
            
            # 将属性添加到 NetworkX 图的节点属性中
            for i, node in enumerate(g_gen.nodes()):
                for key in feature_keys:
                    g_gen.nodes[node][key] = pyg_data[key][i]
            largest_cc = max(nx.connected_components(g_gen), key=len)
            g_gen = g_gen.subgraph(largest_cc)
            generated_nxgraphs.append(g_gen)
        
        return generated_nxgraphs
    

dataset = "elliptic"
args = GenerateDataArgs(run_name='2024-12-06_17-50-36', dataset=dataset, num_samples=64, seed=10, checkpoint=50)
generator = GraphDataGenerator(args)

# elliptic
item_num =12800 #12800
batch_size = 128
iter_num = item_num // batch_size
generate_dataset = []
file_name = f"./generateData/{dataset}_generate_{item_num}.pkl"

# 开始合成数据
start_time = time.time()  # 记录开始时间
for i in range(iter_num):
    # print("i:", i)
    generate_data = generator.generate_data(batch_size)
    generate_dataset = generate_dataset + generate_data
end_time = time.time()  # 记录结束时间
generate_dataset=generate_dataset[0:15509]
print("len", len(generate_dataset))

# 保存数据
with open(file_name, 'wb') as f:
    pickle.dump(generate_dataset, f)
print("save finish")

# 合成数据样例（demo）
combined_graph = nx.Graph()
for graph in generate_data:
    combined_graph = nx.compose(combined_graph, graph)
nodes = [{'id': str(node), 'label': int(combined_graph.nodes[node]["label"].item())} for node in combined_graph.nodes]
edges = [{'source': str(edge[0]), 'target': str(edge[1])} for edge in combined_graph.edges]

result = {
    'nodes': nodes,
    'edges': edges
}
print("result:", result)



