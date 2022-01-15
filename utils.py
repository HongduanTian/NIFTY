import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

def is_symmetric(m):
    '''
    Judge whether the matrix is symmetric or not.
    :param m: Adjacency matrix(Array)
    '''
    res = np.int64(np.triu(m).T == np.tril(m))
    if np.where(res==0)[0] != []:
        raise ValueError("The matrix is not symmetric!")
    else:
        pass

def symetric_normalize(m, half: bool):
    '''
    Symmetrically normalization for adjacency matrix
    :param m: (Array) Adjacency matrix
    :param half: (bool) whether m is triu or full
    :return: (Array) An symmetric adjacency matrix
    '''
    if not half:
        is_symmetric(m)
    else:
        m = m + m.T - np.diag(np.diagonal(m))

    hat_m = m + np.eye(m.shape[0])
    D = np.sum(hat_m, axis=1)
    D = np.diag(D)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    sn_m = np.matmul(np.matmul(D, hat_m), D)
    return sn_m

def sp2sptensor(m):

    sparse_m = sp.coo_matrix(m).astype(np.float)
    indices = torch.from_numpy(np.vstack((sparse_m.row, sparse_m.col)).astype(int))
    values = torch.from_numpy(sparse_m.data)
    shape = torch.Size(sparse_m.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def spectral_norm(m):
    '''
    Calculate the spectral norm of the m
    :param m: (Tensor) parameter tensor
    :return: (Tensor) normed parameter tensor
    '''
    eigvals, _ = torch.eig(torch.mm(m.T, m))
    max_eigval = torch.max(eigvals)
    if max_eigval <= 0:
        raise ValueError("Eigen value should be a non-negative number!")
    normed_m = m / torch.sqrt(max_eigval)
    return normed_m

def sim_loss(emb_1, pred_1, emb_2, pred_2):
    '''
    Calculate the cosine similarity loss
    :param emb_1: (Tensor) the first embedding
    :param pred_1: (Tensor) the prediction of the first embedding
    :param emb_2: (Tensor) the second embedding
    :param pred_2: (Tensor) the prediction of the second embedding
    :return: the similarity loss
    '''
    l1 = -F.cosine_similarity(pred_1, emb_2.detach(), dim=-1).mean()
    l2 = -F.cosine_similarity(pred_2, emb_1.detach(), dim=-1).mean()

    return 0.5 * (l1 + l2)

def stat_parity_and_equal(sen_val, out, labels, idx):

    # get indices of s=0 and s=1
    s0_idx = np.where(sen_val[idx]==0)
    s1_idx = np.where(sen_val[idx]==1)

    cur_label = labels[idx]
    s0_ty1_idx = np.where(cur_label[s0_idx]==1)
    s1_ty1_idx = np.where(cur_label[s1_idx]==1)

    # sp = |P(y^=1|s=0) - P(Y^=1|s=1)|
    sp_out = out[idx]
    num_s0y1 = sum(np.argmax(sp_out[s0_idx], axis=1))
    num_s1y1 = sum(np.argmax(sp_out[s1_idx], axis=1))
    #num_s0y1 = sum(np.int64(np.argmax(sp_out[s0_idx], axis=1)==1))
    #num_s1y1 = sum(np.int64(np.argmax(sp_out[s1_idx], axis=1)==1))
    sp = abs(num_s0y1/len(s0_idx[0]) - num_s1y1/len(s1_idx[0]))

    # eo = |P(y^=1|y=1, s=0) - P(y^=1|y=1, s=1)|
    eo_out = out[idx]
    s0_y1_out = eo_out[s0_idx]  # s=0
    s1_y1_out = eo_out[s1_idx]  # s=1
    p1 = sum(np.int64(np.argmax(s0_y1_out[s0_ty1_idx], axis=1)))/len(s0_ty1_idx[0])
    p2 = sum(np.int64(np.argmax(s1_y1_out[s1_ty1_idx], axis=1)))/len(s1_ty1_idx[0])
    eo = abs(p1 - p2)
    '''
    pred = np.argmax(out, axis=1)[idx]
    label = labels[idx]
    vals = sen_val[idx]

    s0_idx = vals == 0
    s1_idx = vals == 1

    s0_y1_idx = np.bitwise_and(s0_idx, label == 1)
    s1_y1_idx = np.bitwise_and(s1_idx, label == 1)

    sp = abs(sum(pred[s0_idx])/sum(s0_idx) - sum(pred[s1_idx])/sum(s1_idx))
    eo = abs(sum(pred[s0_y1_idx])/sum(s0_y1_idx) - sum(pred[s1_y1_idx])/sum(s1_y1_idx))
    '''
    return sp, eo