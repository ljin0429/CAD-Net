import torch
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import softmax, sort_edge_index, degree
from torch_scatter import scatter_add
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class AdaCAD(MessagePassing):
    def __init__(self, K, beta, dropout=0.5, **kwargs):
        super(AdaCAD, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.beta = beta
        self.dropout = dropout

    def forward(self, x, edge_index, train_mask, is_debug=False):

        # Step 1: Class Distribution & Entropy Regularization
        cd = F.softmax(x, dim=-1)
        EPS = 1e-15
        entropy = -(cd * torch.log(cd + EPS)).sum(dim=-1)

        # Step 2: Compute a transition matrix: transP
        transP, sum_pipj = self.compute_transP(cd, edge_index)

        # Step 3: gamma
        with torch.no_grad():
            deg = degree(edge_index[0])
            deg[deg==0] = 1
            cont_i = sum_pipj / deg

            gamma = self.beta + (1 - self.beta) * cont_i

        # Step 4: Aggregate features
        x = F.dropout(x, p=self.dropout, training=self.training)
        H = x

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, transP=transP)

        x = (1 - gamma.unsqueeze(dim=-1)) * H + gamma.unsqueeze(dim=-1) * x

        if is_debug:
            debug_tensor = []
            with torch.no_grad():
                debug_tensor.append(sort_edge_index(edge_index, transP))
                debug_tensor.append(cd)
                debug_tensor.append(sum_pipj)
                debug_tensor.append(gamma)
        else:
            debug_tensor = None

        return x, entropy, debug_tensor

    def compute_transP(self, cd, edge_index):
        """

        :param cd: class distribution [N, D]
        :param edge_index: [2, E]
        :return: transition probability (transP) [E, 1]
        """

        # edge_index: [2, E]
        row, col = edge_index  # row, col: [1, E] each

        # Indexing: [N, D] -> [E, D]
        p_i = cd[row]
        p_j = cd[col]

        # Transition Probability
        pipj = (p_i * p_j).sum(dim=-1)  # [E, 1]
        transP = softmax(pipj, row, cd.size(0))

        with torch.no_grad():
            sum_pipj = scatter_add(pipj, row)

        return transP, sum_pipj

    def message(self, x_j, transP):
        return x_j * transP.view(-1, 1)

    def __repr__(self):
        return '{}(K = {}, beta={})'.format(self.__class__.__name__, self.K, self.beta)







