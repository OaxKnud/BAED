# BAED

Official pytorch implementation for ["Balanced Anomaly-guided Ego-graph Diffusion Model for Inductive Graph Anomaly Detection"]. 


## Environment Requirements

Before running the scripts, ensure your environment meets the following requirements. Install the necessary Python packages using pip:
```
pip install -r requirements.txt
```
Additionally, manually download the appropriate versions of the following libraries that are compatible with your specific CUDA and PyTorch versions:

```
torch-cluster
torch-scatter
torch-sparse
torch-spline-conv
```

## Training script
### 01 Construct Dataset
Download datasets from https://pan.baidu.com/s/1yPs6OU2tm0_jmxNHI76HAQ?pwd=5eae. Put datasets in ''graphs'' forder.
Execute the script to generate the BAED training dataset and downstream task training datasets in the graphs folder.
```
cd createSubGraphDataset
python createDataset.py
```

### 02 Pre-train GAE
Train the GAE model for generating embedding condition vectors.
```
cd gae
python saveGAEdataset.py
python trainGAE.py
```

### 03 Train BAED
Training results can be found under wandb/{dataset_name}/multinomial_diffusion/multistep/{run_name}.
```
python train.py \
        --epochs 2000 \
        --num_generation 64 \
        --diffusion_dim 64 \
        --diffusion_steps 128 \
        --device cuda:0 \
        --dataset elliptic \
        --batch_size 8 \
        --clip_value 1 \
        --lr 1e-4 \
        --optimizer adam \
        --final_prob_edge 1 0 \
        --sample_time_method importance \
        --check_every 10 \
        --eval_every 10 \
        --noise_schedule linear \
        --dp_rate 0.1 \
        --loss_type vb_ce_xt_prescribred_st \
        --arch TGNN_embedding_guided \
        --parametrization xt_prescribed_st \
        --empty_graph_sampler empirical \
        --degree \
        --num_heads 8 8 8 8 1  \
        --log_wandb False
```

### 04 Data Synthesis
```
python evaluate.py \
        --run_name 2025-11-40_25-29-36 \
        --dataset elliptic \
        --num_samples 8 \
        --checkpoints 2000
```

### 05 Downstream Task: Graph Anomaly Detection
The downstream task provides basic models for graph anomaly detection, including GCN, BernNet, BWGNN, and Graphsage models for testing and evaluation. The evaluation metrics include AUROC and AUPRC.
```
python trainSubgraphModel.py
```

For further details about the methodology, experimental setup, and results, please refer to our paper.

