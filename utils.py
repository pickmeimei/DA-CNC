import os
import torch
import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
# import matplotlib.pyplot as plt
from sklearn import metrics
from models import *
from layers import aggregator_lookup
from sklearn.decomposition import PCA
from scipy.sparse import csc_matrix
from scipy.linalg import fractional_matrix_power, inv


def top_k_preds(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)
    top_k_array = np.array(predictions, np.int64)

    return top_k_array


def cal_f1_score(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1


def batch_generator(nodes, batch_size, shuffle=True):
    num = nodes.shape[0]
    chunk = num // batch_size
    while True:
        if chunk * batch_size + batch_size > num:
            chunk = 0   
            if shuffle:
                idx = np.random.permutation(num)
        b_nodes = nodes[idx[chunk*batch_size:(chunk+1)*batch_size]]
        chunk += 1

        yield b_nodes


def eval_iterate(nodes, batch_size, shuffle=False):
    idx = np.arange(nodes.shape[0])
    if shuffle:
        idx = np.random.permutation(idx)
    n_chunk = idx.shape[0] // batch_size + 1
    for chunk_id, chunk in enumerate(np.array_split(idx, n_chunk)):
        b_nodes = nodes[chunk]

        yield b_nodes


def do_iter(emb_model, cly_model, adj, feature, labels, idx, cal_f1=False, is_social_net=False):
    embs = emb_model(idx, adj, feature)
    preds = cly_model(embs)
    if is_social_net:
        labels_idx = torch.argmax(labels[idx], dim=1)
        cly_loss = F.cross_entropy(preds, labels_idx)   
    else:
        cly_loss = F.multilabel_soft_margin_loss(preds, labels[idx])
    if not cal_f1:
        return embs, cly_loss
    else:
        targets = labels[idx].cpu().numpy()
        preds = top_k_preds(targets, preds.detach().cpu().numpy())
        return embs, cly_loss, preds, targets


def evaluate(emb_model, cly_model, adj, feature, labels, idx, batch_size, mode='val', is_social_net=False):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, feature, labels,
                                                                                         b_nodes, cal_f1=True, is_social_net=is_social_net)
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))

    return cly_loss, micro_f1, macro_f1, embs_whole, targets_whole


def get_split(labels, seed):
    idx_tot = np.arange(labels.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx_tot)

    return idx_tot

# 原来的
def make_adjacency(G, max_degree, seed):
    all_nodes = np.sort(np.array(G.nodes()))
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes, max_degree)) + (n_nodes - 1)).astype(np.int64)
    np.random.seed(seed)
    for node in all_nodes:
        # neibs = np.array(G[node])
        neibs = np.array(G.neighbors(node))
        if len(neibs) == 0:
            neibs = np.array(node).repeat(max_degree)
        elif len(neibs) < max_degree:
            neibs = np.random.choice(neibs, max_degree, replace=True)
        else:
            neibs = np.random.choice(neibs, max_degree, replace=False)
        adj[node, :] = neibs

    return adj


#

def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=np.float64)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def pre_social_net(adj, features, labels):
    features = csc_matrix(features.astype(np.uint8))
    labels = labels.astype(np.int32)

    return adj, features, labels

def get_splithead(idx_train, seed):
    # idx_tot = np.arange(labels.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx_train)

    return idx_train

def split_nodes(A_num, k=9):
    # num_links = np.sum(adj, axis=1)
    # idx_train = np.where(num_links > k)[0]
    #
    # idx_test = np.where(num_links <= k)[0]
    idx_train = np.where(A_num >= k)[0]
    idx_test = np.where(A_num < k)[0]


    return idx_train, idx_test







def load_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_dense = np.array(adj.todense())

    A_num = np.sum(adj_dense, 1)# 计算节点的度
    # nodes_with_degree_gt_9 = np.where(A_num < 9000)[0]  # 度数大于9的节点索引
    # print(nodes_with_degree_gt_9)
    # labels_9 = labels[nodes_with_degree_gt_9]
    # print(labels_9)
    # column_sums = np.sum(labels_9, axis=0)
    # print(column_sums)

    edges = np.vstack(np.where(adj_dense)).T
    Graph = nx.from_edgelist(edges)
    adj = make_adjacency(Graph, 128, seed)
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device)





def distillation_loss(student_output, teacher_output, temperature=2.0):
    teacher_soft = F.softmax(teacher_output / temperature, dim=1)
    student_log_soft = F.log_softmax(student_output / temperature, dim=1)
    return F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)


def load_data_diff(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_dense = np.array(adj.todense())
    edges = np.vstack(np.where(adj_dense)).T
    Graph = nx.from_edgelist(edges)
    # Compute the PPR diffusion matrix
    diff = compute_ppr(adj_dense, alpha=0.2)
    diff_dense = torch.from_numpy(diff)
    diff_edges = np.vstack(np.where(diff_dense)).T
    diff_graph = nx.from_edgelist(diff_edges)
    diff = make_adjacency(diff_graph, 128, seed)

    adj = make_adjacency(Graph, 128, seed)
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    diff = torch.from_numpy(diff)
    idx_tot = torch.LongTensor(idx_tot)



    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device), diff.to(device)


import numpy as np
import scipy.io as sio
import networkx as nx
import torch


def load_data_diff_delete(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']

    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)

    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_dense = np.array(adj.todense())
    sum1 = np.sum(adj_dense)
    print(sum1)

    # 删除20%的边
    # edges = np.vstack(np.where(adj_dense)).T
    # num_edges = edges.shape[0]
    # num_edges_to_remove = int(num_edges * 0.2)
    # np.random.seed(seed)
    # edges_to_remove = np.random.choice(num_edges, num_edges_to_remove, replace=False)
    # adj_dense[edges[edges_to_remove, 0], edges[edges_to_remove, 1]] = 0
    # adj_dense[edges[edges_to_remove, 1], edges[edges_to_remove, 0]] = 0  # 如果是无向图，需要对称删除

    edges = np.vstack(np.where(adj_dense)).T
    Graph = nx.from_edgelist(edges)

    # Compute the PPR diffusion matrix
    diff = compute_ppr(adj_dense, alpha=0.2)
    diff_dense = torch.from_numpy(diff)
    diff_edges = np.vstack(np.where(diff_dense)).T
    diff_graph = nx.from_edgelist(diff_edges)
    diff = make_adjacency(diff_graph, 128, seed)

    adj = make_adjacency(Graph, 128, seed)
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    diff = torch.from_numpy(diff)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device), diff.to(device)




def compute_ppr(a, alpha=0.2, self_loop=True):



    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=F.relu, input_type='dense', drop=0.0):
    """
    创建一个全连接层（线性层）。

    参数:
        input_tensor (torch.Tensor): 输入张量。
        input_dim (int): 输入特征的维度。
        output_dim (int): 输出特征的维度。
        layer_name (str): 层的名称（在 PyTorch 中主要用于变量命名，不直接使用）。
        act (callable, optional): 激活函数，默认为 ReLU。
        input_type (str, optional): 输入类型，'dense' 或 'sparse'。默认为 'dense'。
        drop (float, optional): Dropout 概率，默认为 0.0（不使用 dropout）。

    返回:
        torch.Tensor: 经过全连接层处理后的输出张量。
    """
    # 初始化权重和偏置
    weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
    bias = nn.Parameter(torch.Tensor(output_dim))

    # 使用 Xavier 初始化权重
    nn.init.xavier_uniform_(weight)

    # 初始化偏置为 0.1
    nn.init.constant_(bias, 0.1)

    # 创建线性层
    linear = nn.Linear(input_dim, output_dim, bias=True)
    linear.weight = weight
    linear.bias = bias

    # 处理输入类型
    if input_type == 'sparse':
        if not isinstance(input_tensor, torch.sparse.Tensor):
            raise ValueError("Input tensor must be sparse for input_type='sparse'")
        # 将稀疏张量转换为密集张量后进行线性变换
        activations = linear(input_tensor.to_dense())
    else:
        activations = linear(input_tensor)

    # 应用激活函数
    activations = act(activations)

    # 应用 dropout
    if drop > 0.0:
        activations = F.dropout(activations, p=drop, training=True)

    return activations