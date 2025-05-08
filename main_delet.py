import scipy.io as sio
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
import random


def load_network(file):
    net = sio.loadmat(file,mat_dtype=True)
    X,A,Y= net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.csc_matrix):
        X = csc_matrix(X)

    return A, X, Y

def delete_edges(adj_matrix, ratio):
    # 获取邻接矩阵的大小
    n = adj_matrix.shape[0]
    # 获取所有的边，每条边用一个元组表示，包含两个节点的索引
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj_matrix[i][j] == 1]
    # 计算要删除的边的数量
    m = int(len(edges) * ratio)
    # 随机选择要删除的边
    deleted_edges = random.sample(edges, m)
    # 将邻接矩阵中对应的元素置为0
    for i, j in deleted_edges:
        adj_matrix[i][j] = 0
        adj_matrix[j][i] = 0
    # 检查是否有孤立的节点，即没有任何边的节点
    isolated_nodes = [i for i in range(n) if np.sum(adj_matrix[i]) == 0]
    # 如果有孤立的节点，就随机选择一条边恢复
    for i in isolated_nodes:
        # 找到所有和该节点相连的边，包括已删除和未删除的
        connected_edges = [(i, j) for j in range(n) if j != i]
        # 随机选择一条边恢复
        i, j = random.choice(connected_edges)
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1
    # 返回修改后的邻接矩阵
    return adj_matrix

netlist=['Blog1','Blog2','dblpv7','acmv9','citationv1']
# netlist=['Citeseer1','Citeseer2']
# netlist=['Blog1']

# network = 'Blog1'
for network in netlist:
    ####################
    # Load source data
    ####################
    A, X, Y = load_network('./Datasets/' + str(network) + '.mat')
    n = X.shape[1]  # 矩阵的列数
    A=A.toarray()
    sum1=np.sum(A)
    print(sum1)
    ratio=0.8
    A=delete_edges(A,ratio)
    sum2 = np.sum(A)
    print(sum2)
    print("ratio:",sum2/sum1)
    A=csc_matrix(A)
    if 'Blog' in network:
        X = X.toarray()



    # print(np.sum(X))
    scipy.io.savemat('./Data_deleteedge/' + network+'_'+str(ratio)+'.mat', {'attrb': X, 'group': Y,'network': A},do_compression=True)

