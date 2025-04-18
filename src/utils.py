import pyscipopt as scip
import torch
import os
import random
import numpy as np
import ecole

def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_cpu_num(cpu_num):
    """
    Set the number of used cpu kernals.
    """
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def instance_to_tensor(file_path):
    instance = ecole.scip.Model.from_file(file_path)
    obs = ecole.observation.MilpBipartite().extract(instance, True)
    b = torch.FloatTensor(obs.constraint_features)
    A_i = torch.LongTensor(np.array(obs.edge_features.indices, dtype=int))
    A_e = torch.FloatTensor(obs.edge_features.values.reshape(-1))
    c = torch.FloatTensor(obs.variable_features[:,0].reshape(-1,1))
    A = torch.sparse.FloatTensor(A_i, A_e).to_dense()
    
    return A, b, c

def instance2graph(path):
    model = ecole.scip.Model.from_file(path)
    obs = ecole.observation.MilpBipartite().extract(model, True)
    constraint_features = obs.constraint_features
    edge_indices = np.array(obs.edge_features.indices, dtype=int)
    edge_features = obs.edge_features.values.reshape((-1,1))
    variable_features = obs.variable_features
    graph = [constraint_features, edge_indices, edge_features, variable_features]

    return graph


def gumbel_sample(logits: torch.Tensor, N: int, tau: float=1.0):
    logits = logits.reshape(-1, 1)
    logits = logits.repeat(N, 1, 1)
    # logits = torch.cat([-logits, logits], dim=-1)
    logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)
    return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)[:,:,1]