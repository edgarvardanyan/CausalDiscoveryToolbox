import sys
from os.path import dirname, join
import time
from datetime import timedelta
import pickle
import networkx as nx
import torch
import numpy as np
import pandas as pd
import os

sys.path.append(dirname(
    dirname((dirname(os.path.abspath(__file__))))))
print(sys.path)
from cdt.causality.graph import CGMT

torch.manual_seed(0)
np.random.seed(0)

basedir = dirname(__file__)

with open(join(basedir, "data.pickle"), 'rb') as file:
    data = pickle.load(file)

results = []

true, edges_processed = 0, 0
start = time.time()
for i, sample in enumerate(data):
    values = sample['values']
    graph = sample['graph']
    values = pd.DataFrame(values)
    model = CGMT(
        num_encoders=1,
        d_input=2,
        d_model=4,
        dim_feedforward=8,
        n_hidden=8,
        n_head=1,
        dropout=0,
        nruns=4,
        train_epochs=1000,
        test_epochs=1000,
        verbose=True
    )
    inferred = model.orient_undirected_graph(
        values,
        nx.Graph(graph))
    results.append((graph, inferred))
    with open(join(basedir, "results_cgnn.pickle"), "wb+") as file:
        pickle.dump(results, file)
    for edge in graph.edges:
        if edge in inferred.edges:
            true += 1
    edges_processed += len(graph.edges)
    print(f"Current result is {true}/{edges_processed}")
    current_time = time.time()
    difference = timedelta(seconds=(current_time - start))
    print(f"{i+1} graphs processed in {difference}")
    sys.stdout.flush()
