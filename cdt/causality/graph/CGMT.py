"""Causal Generative Neural Networks.

Author : Olivier Goudet & Diviyan Kalainathan
Ref : Causal Generative Neural Networks (https://arxiv.org/abs/1711.08936)
Date : 09/5/17

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
import itertools
import warnings
from typing import Optional

import torch as th
import pandas as pd
from copy import deepcopy
from tqdm import trange
from torch.utils.data import DataLoader
from sklearn.preprocessing import scale
from ..pairwise.GNN import GNN
from ...utils.loss import MMDloss
from ...utils.Settings import SETTINGS
from ...utils.graph import dagify_min_edge
from ...utils.parallel import parallel_run
from .model import GraphModel
import torch
from numpy.linalg import linalg
from torch import nn
import numpy as np
import networkx as nx


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class CGMT_model:

    def __init__(self,
                 adjacency_matrix: np.ndarray,
                 batch_size,
                 num_encoders=3,
                 d_input=4,
                 d_model=4,
                 n_head=1,
                 dim_feedforward=16,
                 dropout=0.1,
                 n_hidden=5,
                 positional_encoding="laplacian",
                 pos_enc_dim=None,
                 pos_enc_sign_flip=False,
                 device=None,
                 *args,
                 **kwargs):
        dag = nx.convert_matrix.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)
        self.model = GraphTransformer(
            dag=dag,
            num_encoders=num_encoders,
            d_input=d_input,
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_hidden=n_hidden,
            positional_encoding=positional_encoding,
            pos_enc_dim=pos_enc_dim,
            pos_enc_sign_flip=pos_enc_sign_flip,
        )
        self.dag = dag
        self.model.to(device)
        self.batch_size = batch_size
        self.device = device
        self.d_input = d_input

    def to(self, device):
        self.model.to(device)

    def reset_parameters(self):
        self.model.init_weights()

    def run(self, dataset, train_epochs=1000, test_epochs=1000, verbose=None,
            idx=0, lr=0.01, dataloader_workers=0, **kwargs) -> float:
        data_size = dataset.shape[0]
        batch_size = self.batch_size
        if self.batch_size is None:
            batch_size = data_size
        n_nodes = len(self.dag.nodes)
        if n_nodes != dataset.shape[1]:
            raise ValueError("Number of nodes in skeleton must be "
                             "equal to number of variables in data")
        device = self.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        elif device not in {'cuda', 'cpu'}:
            raise ValueError("Unknown device (must be 'cuda' or 'cpu')")

        optimizer = th.optim.Adam(self.model.parameters(), lr=lr)
        self.score = 0
        criterion = MMDloss(batch_size).to(device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, drop_last=True,
                                num_workers=dataloader_workers)
        with trange(train_epochs + test_epochs, disable=not verbose) as t:
            for epoch in t:
                for i, data in enumerate(dataloader):
                    optimizer.zero_grad()
                    data_indices = np.random.choice(
                        range(data_size), batch_size, replace=False)
                    target = dataset[data_indices].to(device)
                    random_input = torch.randn((
                        batch_size, n_nodes, self.d_input)).to(device)

                    generated = self.model.forward(random_input)
                    generated = generated.view(generated.size()[:2])
                    loss = criterion(generated, target)
                    if not epoch % 200 and i == 0:
                        t.set_postfix(idx=idx, loss=loss.item())
                    loss.backward()
                    optimizer.step()
                    if epoch >= test_epochs:
                        self.score += float(loss.data)

        return self.score / test_epochs


class GraphTransformer(nn.Module):
    """
    Generates data from a random input according to the provided DAG.
    Masks attentions layers of the transformer according
    to adjacency matrix of the DAG.
    """

    def __init__(self,
                 dag: nx.DiGraph,
                 num_encoders=3,
                 d_input=4,
                 d_model=4,
                 n_head=1,
                 dim_feedforward=16,
                 dropout=0.1,
                 n_hidden=5,
                 positional_encoding="laplacian",
                 pos_enc_dim=None,
                 pos_enc_sign_flip=False,
                 ):
        """
        Args:
            dag: directed acyclic graph
            num_encoders: number of transformer encoder layers
            d_input: model input vector dimension
            d_model: transformer layer input dimensions
            n_head: number of heads in multi-head attention
            dim_feedforward: encoder layer feedforward sublayer dimension
            dropout: dropout rate in the encoder layer
            positional_encoding: 'laplacian' or 'none'
            pos_enc_dim: dimension of the positional encoding.
                Default is the number of of nodes in the DAG
            pos_enc_sign_flip: In case of True signs of positional
                encoding are randomly flipped
        """
        super(GraphTransformer, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.dag = dag
        self.input_linear = nn.Linear(self.d_input, self.d_model)

        if positional_encoding == 'laplacian':
            if pos_enc_dim is None:
                self.pos_enc_dim = len(dag.nodes)
            else:
                self.pos_enc_dim = pos_enc_dim

            self.pe_linear = nn.Linear(self.pos_enc_dim, self.d_model)

            self.pe = self.get_laplacian_pe(
                sign_flip=pos_enc_sign_flip
            )
        elif positional_encoding == 'none':
            self.pe = None
            self.pe_linear = None
            self.pos_enc_dim = None
        else:
            raise ValueError("Invalid Positional Encoding")

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoders
        )

        adjacency_matrix = nx.adjacency_matrix(dag).todense()
        self.n_nodes, self.mask = self.generate_mask(adjacency_matrix)
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(n_hidden, 1)

        self.masks = {}
        self.pes = {}

        self.init_weights()

    def _get_mask(self, device):
        if device not in self.masks:
            self.masks[device] = self.mask.to(device)
        return self.masks[device]

    def _get_pe(self, device):
        if self.pe is None:
            return None
        if device not in self.pes:
            self.pes[device] = self.pe.to(device)
        return self.pes[device]

    def init_weights(self):
        init_range = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-init_range, init_range)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-init_range, init_range)

    def get_laplacian_pe(self, sign_flip: bool = False) -> torch.Tensor:
        # get the eigenvectors of the laplacian matrix
        laplacian_matrix = nx.laplacian_matrix(nx.Graph(self.dag)).toarray()
        eigenvalues, eigenvectors = linalg.eig(laplacian_matrix)

        # set the eigenvectors with the smallest eigenvalues as PE
        idx = eigenvalues.argsort()[::-1][:self.pos_enc_dim]
        eigenvectors = eigenvectors[:, idx]
        if sign_flip:
            # randomly flip signs of eigenvectors
            signs = np.random.uniform(size=self.pos_enc_dim) > 0.5
            signs = signs.astype(int)
            eigenvectors *= signs
        pos_enc = torch.from_numpy(eigenvectors).float()
        return pos_enc

    @staticmethod
    def generate_mask(adjacency_matrix):
        n_nodes = adjacency_matrix.shape[0]
        mask = np.diag(np.ones(n_nodes)) + adjacency_matrix.T
        mask = mask.astype(np.float32)

        def f(x):
            return -1e9 if x == 0 else 0.0

        # must add -inf to all attentions before applying softmax
        # that do not correspond to any edge from the DAG
        f = np.vectorize(f)
        mask = torch.from_numpy(f(mask))
        return n_nodes, mask

    def forward(self, src: torch.Tensor):
        """
        Args:
            src: normal random tensor with size (n_nodes, batch_size, d_model)

        Returns: generated data with size (n_node, batch_size, 1)

        """
        # TODO: Update the dims in docstrings and error messages
        assert src.size()[1] == self.n_nodes, \
            "First dim of src must be equal to model.n_nodes"
        assert src.size()[2] == self.d_input, \
            "Third dim of src must be equal to model.d_model(by default 4)"
        src = self.input_linear(src)

        mask, pe = self.mask, self.pe
        if mask.get_device() != src.get_device():
            mask = mask.to(src.get_device())
        if pe is not None and pe.get_device() != src.get_device():
            pe = pe.to(src.get_device())

        if pe is not None:
            batch_size = src.size()[0]
            pos_enc = torch.unsqueeze(pe, 0).repeat(batch_size, 1, 1)
            pos_enc = self.pe_linear(pos_enc)
            src += pos_enc

        #  setting n_nodes as the first dimension for transformer encoder layer
        src = torch.transpose(src, 0, 1)

        src = self.encoder(src, mask)

        output = self.activation(self.linear1(src))

        return torch.transpose(self.linear2(output), 0, 1)


def graph_evaluation(data, adj_matrix, device='cpu', batch_size=-1, **kwargs):
    """Evaluate a graph taking account of the hardware."""
    if isinstance(data, th.utils.data.Dataset):
        obs = data.to(device)
    else:
        obs = th.Tensor(scale(data.values)).to(device)
    if batch_size == -1:
        batch_size = obs.__len__()
        cgmt = CGMT_model(adj_matrix, batch_size, **kwargs)
        cgmt.to(device)
    cgmt.reset_parameters()
    return cgmt.run(obs, **kwargs)


def train_given_graph(data, adj_matrix, device='cpu', batch_size=-1, **kwargs):
    """Evaluate a graph taking account of the hardware."""
    if isinstance(data, th.utils.data.Dataset):
        obs = data.to(device)
    else:
        obs = th.Tensor(scale(data.values)).to(device)
    if batch_size == -1:
        batch_size = obs.__len__()
    cgnn = CGMT_model(adj_matrix, batch_size, **kwargs).to(device)
    cgnn.reset_parameters()
    cgnn.run(obs, **kwargs)
    return cgnn


def parallel_graph_evaluation(data, adj_matrix, nruns=16,
                              njobs=None, gpus=None, **kwargs):
    """Parallelize the various runs of CGNN to evaluate a graph."""
    njobs, gpus = SETTINGS.get_default(('njobs', njobs), ('gpu', gpus))

    if gpus == 0:
        output = [graph_evaluation(data, adj_matrix,
                                   device=SETTINGS.default_device, **kwargs)
                  for run in range(nruns)]
    else:
        output = parallel_run(graph_evaluation, data,
                              adj_matrix, njobs=njobs,
                              gpus=gpus, nruns=nruns, **kwargs)
    return np.mean(output)


def hill_climbing(data, graph, **kwargs):
    """Hill Climbing optimization: a greedy exploration algorithm."""
    if isinstance(data, th.utils.data.Dataset):
        nodelist = data.get_names()
    elif isinstance(data, pd.DataFrame):
        nodelist = list(data.columns)
    else:
        raise TypeError('Data type not understood')
    tested_candidates = [nx.adj_matrix(graph, nodelist=nodelist, weight=None)]
    best_score = parallel_graph_evaluation(data,
                                           tested_candidates[0].todense(),
                                           ** kwargs)
    best_candidate = graph
    can_improve = True
    while can_improve:
        can_improve = False
        for (i, j) in best_candidate.edges():
            test_graph = deepcopy(best_candidate)
            test_graph.add_edge(j, i, weight=test_graph[i][j]['weight'])
            test_graph.remove_edge(i, j)
            tadjmat = nx.adj_matrix(test_graph, nodelist=nodelist, weight=None)
            if (nx.is_directed_acyclic_graph(test_graph) and not any([(tadjmat != cand).nnz ==
                                                                      0 for cand in tested_candidates])):
                tested_candidates.append(tadjmat)
                score = parallel_graph_evaluation(data, tadjmat.todense(),
                                                  **kwargs)
                if score < best_score:
                    can_improve = True
                    best_candidate = test_graph
                    best_score = score
                    break
    return best_candidate


class CGMT(GraphModel):

    def __init__(self,
                 nh=5,
                 num_encoders=1,
                 d_input=2,
                 d_model=4,
                 dim_feedforward=8,
                 n_hidden=8,
                 n_head=1,
                 dropout=0,
                 nruns=16, njobs=None, gpus=None, batch_size=-1,
                 lr=0.01, train_epochs=1000, test_epochs=1000, verbose=None,
                 dataloader_workers=0):
        """ Initialize the CGNN Model."""
        super(CGMT, self).__init__()
        self.dropout = dropout
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.dim_feedforward = dim_feedforward
        self.d_model = d_model
        self.d_input = d_input
        self.num_encoders = num_encoders
        self.nh = nh
        self.nruns = nruns
        self.njobs = SETTINGS.get_default(njobs=njobs)
        self.gpus = SETTINGS.get_default(gpu=gpus)
        self.lr = lr
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

    def create_graph_from_data(self, data):
        """
        Args:
            data (pandas.DataFrame or torch.utils.data.Dataset):
        Returns:
            networkx.DiGraph:
        """
        warnings.warn("An exhaustive search of the causal structure of CGMT without"
                      " skeleton is super-exponential in the number of variables.")

        # Building all possible candidates:
        if not isinstance(data, th.utils.data.Dataset):
            nb_vars = len(list(data.columns))
            names = list(data.columns)
        else:
            nb_vars = data.__featurelen__()
            names = data.get_names()
        candidates = [np.reshape(np.array(i), (nb_vars, nb_vars)) for i in itertools.product([0, 1], repeat=nb_vars*nb_vars)
                      if (np.trace(np.reshape(np.array(i), (nb_vars, nb_vars))) == 0
                          and nx.is_directed_acyclic_graph(nx.DiGraph(np.reshape(np.array(i), (nb_vars, nb_vars)))))]
        warnings.warn("A total of {} graphs will be evaluated.".format(len(candidates)))
        scores = [parallel_graph_evaluation(data, i, njobs=self.njobs, nh=self.nh,
                                            nruns=self.nruns, gpus=self.gpus,
                                            lr=self.lr, train_epochs=self.train_epochs,
                                            test_epochs=self.test_epochs,
                                            verbose=self.verbose,
                                            batch_size=self.batch_size,
                                            dataloader_workers=self.dataloader_workers)
                  for i in candidates]
        final_candidate = candidates[scores.index(min(scores))]
        output = np.zeros(final_candidate.shape)

        # Retrieve the confidence score on each edge.
        for (i, j), x in np.ndenumerate(final_candidate):
            if x > 0:
                cand = np.copy(final_candidate)
                cand[i, j] = 0
                output[i, j] = min(scores) - scores[[np.array_equal(cand, tgraph)
                                                     for tgraph in candidates].index(True)]
        prediction = nx.DiGraph(final_candidate * output)
        return nx.relabel_nodes(prediction, {idx: i for idx, i in enumerate(names)})

    def orient_directed_graph(self, data, dag, alg='HC'):
        """

        Args:
            data (pandas.DataFrame or torch.utils.data.Dataset):
            dag (nx.DiGraph):
            alg (str):
        Returns:
            networkx.DiGraph:

        """
        alg_dic = {'HC': hill_climbing}  # , 'HCr': hill_climbing_with_removal,
        # 'tabu': tabu_search, 'EHC': exploratory_hill_climbing}
        # if not isinstance(data, th.utils.data.Dataset):
        #     data = MetaDataset(data)

        return alg_dic[alg](data, dag,
                            num_encoders=self.num_encoders,
                            d_input=self.d_input,
                            d_model=self.d_model,
                            dim_feedforward=self.dim_feedforward,
                            n_hidden=self.n_hidden,
                            n_head=self.n_head,
                            dropout=self.dropout,
                            njobs=self.njobs,
                            nruns=self.nruns, gpus=self.gpus,
                            lr=self.lr, train_epochs=self.train_epochs,
                            test_epochs=self.test_epochs, verbose=self.verbose,
                            batch_size=self.batch_size,
                            dataloader_workers=self.dataloader_workers)

    def orient_undirected_graph(self, data, umg, alg='HC'):
        """

        Args:
            data (pandas.DataFrame):
            umg (nx.Graph):
            alg (str):
        Returns:
            networkx.DiGraph:
        """
        warnings.warn("The pairwise GNN model is computed on each edge of the UMG "
                      "to initialize the model and start CGMT with a DAG")
        gnn = GNN(nh=self.nh, lr=self.lr, nruns=self.nruns,
                  njobs=self.njobs,
                  train_epochs=self.train_epochs, test_epochs=self.test_epochs,
                  verbose=self.verbose, gpus=self.gpus, batch_size=self.batch_size,
                  dataloader_workers=self.dataloader_workers)

        og = gnn.orient_graph(data, umg)  # Pairwise method
        dag = dagify_min_edge(og)

        return self.orient_directed_graph(data, dag, alg=alg)
