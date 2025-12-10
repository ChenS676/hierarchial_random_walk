import time
import numpy as np
import networkx as nx
import tqdm 
import graph_walker  # pylint: disable=import-error
import json
import fire
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_sparse import SparseTensor
import scipy.sparse as sp

# for recurrent random walk
from recurrent_random_walk import (nx_to_sparse_tensor, 
                                   get_recurrent_walks_for_cover,
                                   generate_walks_for_cover_time)
from hierarchial import (build_tree_graph, 
                            build_bottleneck_sbm, 
                            build_lollipop_graph,
                            quick_plot)
class RandomWalkConfig:
    def __init__(
            self,
            n_walks=20,
            walk_len=10000,
            min_degree=False,
            sub_sampling=0.,
            p=1, q=1, alpha=0, k=None,
            no_backtrack=False,
            start_nodes=None,
            seed=None,
            verbose=True
        ):
        self.n_walks = n_walks
        self.walk_len = walk_len
        self.min_degree = min_degree
        self.sub_sampling = sub_sampling
        self.p = p
        self.q = q
        self.alpha = alpha
        self.k = k
        self.no_backtrack = no_backtrack
        self.start_nodes = start_nodes
        self.seed = seed
        self.verbose = verbose


def compute_random_walks(G, config):
    walks, restarts = graph_walker.random_walks(
        G,
        n_walks=config.n_walks,
        walk_len=config.walk_len,
        min_degree=config.min_degree,
        sub_sampling=config.sub_sampling,
        p=config.p,
        q=config.q,
        alpha=config.alpha,
        k=config.k,
        no_backtrack=config.no_backtrack,
        start_nodes=config.start_nodes,
        seed=config.seed,
        verbose=config.verbose
    )
    return walks, restarts


def compute_stationary_distribution(G, config):
    """Compute the stationary distribution of the random walk."""
    probs = graph_walker.stationary_distribution(
        G,
        min_degree=config.min_degree,
        sub_sampling=config.sub_sampling
    )
    return probs


def compute_cover_times(G, walks, restarts):
    """Compute the cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        visited = np.zeros(len(G.nodes), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert i in G.nodes and j in G.nodes, "Node must exist in graph"
            visited[i] = True
            visited[j] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    assert np.all(cover_times != -1), "All walks must cover all edges"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(G.nodes), dtype=np.float32) - 1
    start_nodes = walks[:, 0]
    for start_node in np.unique(start_nodes):
        idxs = np.where(start_nodes == start_node)[0]
        cover_times_per_start_node[start_node] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times_per_start_node


def compute_undirected_edge_cover_times(G, walks, restarts):
    """Compute the undirected edge cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    edges = list(G.edges)
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        visited = np.zeros(len(edges), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert (i, j) in edges or (j, i) in edges, "Edge must exist in graph"
            edge_idx = edges.index((i, j)) if (i, j) in edges else edges.index((j, i))
            visited[edge_idx] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    assert np.all(cover_times != -1), "All walks must cover all edges"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(G.nodes), dtype=np.float32) - 1
    start_nodes = walks[:, 0]
    for start_node in np.unique(start_nodes):
        idxs = np.where(start_nodes == start_node)[0]
        cover_times_per_start_node[start_node] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times


def compute_empirical_stationary_distribution(G, walks):
    """Compute the empirical long-term distribution of the random walk."""
    unique, counts = np.unique(walks, return_counts=True)
    assert np.all(unique == np.arange(len(G.nodes)))
    return counts / np.sum(counts)


def run_tests(G, config):
    # generate random walk
    time_start = time.time()
    walks, restarts = compute_random_walks(G, config)
    time_end = time.time()
    seconds = time_end - time_start

    # compute cover times
    cover_time = np.max(compute_cover_times(G, walks, restarts))

    # compute undirected edge cover times
    undirected_edge_cover_time = np.max(compute_undirected_edge_cover_times(G, walks, restarts))

    # return results
    return (
        seconds,
        cover_time,
        undirected_edge_cover_time,
    )


def get_labels(G, start_nodes, cover_times):
    labels = {node: 0. for node in G.nodes}
    count = {node: 0. for node in G.nodes}
    for start_node, cover_time in zip(start_nodes, cover_times):
        if cover_time == -1:
            continue
        labels[start_node] += cover_time
        count[start_node] += 1
    for node in G.nodes:
        labels[node] = int(labels[node] / count[node]) if count[node] > 0 else -1
    return labels


def run_tests_torch(
    G,
    config,
    use_recurrent: bool = False,
    recurrent_steps: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = nx_to_sparse_tensor(G, device=device)
    n_nodes = G.number_of_nodes()

    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Decide recurrence depth
    if recurrent_steps is None:
        recurrent_steps = getattr(config, "recurrent_steps", 1)

    # -------------------------------------------------------
    # Generate walks
    # -------------------------------------------------------
    time_start = time.time()

    if use_recurrent and recurrent_steps > 1:
        # Recurrent (-style) walks
        start_nodes = torch.arange(n_nodes, device=device)
        walks, restarts = get_recurrent_walks_for_cover(
            adj=adj,
            start_nodes=start_nodes,
            walk_length=config.walk_len,
            num_walks=config.n_walks,
            steps=recurrent_steps,
            flip=True,  
        )
    else:
        # Simple unbiased walks: each node as start, n_walks per node
        walks, restarts = generate_walks_for_cover_time(
            adj=adj,
            num_nodes=n_nodes,
            walk_length=config.walk_len,
            num_walks_per_node=config.n_walks,
        )

    time_end = time.time()
    seconds = time_end - time_start

    node_cover_times = compute_cover_times(G, walks, restarts)
    cover_time = float(node_cover_times.max())

    undirected_edge_cover_times = compute_undirected_edge_cover_times(G, walks, restarts)
    undirected_edge_cover_time = int(undirected_edge_cover_times.max())

    return (
        seconds,
        cover_time,
        undirected_edge_cover_time,
    )


def run_all_tests(G, name, n_seeds=5):
    print(f"Running tests on {name}")
    results = {}
    results["graph"] = name
    
    # recurrent random walks
    for seed in range(n_seeds):
        config = RandomWalkConfig(
            n_walks=2,            # number of walks per source
            walk_len=10000,        # walk length
            seed=seed * 100,
        )
        config.use_torch = True          # activate torch backend
        config.recurrent_steps = 3       # depth of recurrent expansion
        seconds, cover_time, undirected_edge_cover_time = run_tests_torch(G, config)
        print(
            f"recurrent random walk (seed={seed}): {seconds:.2f} sec, "
            f"cover time= {cover_time}, "
            f"undirected edge cover time= {undirected_edge_cover_time}"
        )
        results["recurrent_random_walk"] = (seconds, cover_time, undirected_edge_cover_time)
    print("recurrent random walks done\n")

    
    # unbiased random walks
    for seed in range(n_seeds):
        config = RandomWalkConfig(seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"Unbiased (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
        results['unbiased_'] = (seconds, cover_time, undirected_edge_cover_time)
    print("Unbiased done\n")

    # unbiased random walks with no backtracking
    # for seed in range(n_seeds):
    #     config = RandomWalkConfig(no_backtrack=True, seed=seed * 100)
    #     seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
    #     print(f"Unbiased + no backtracking (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
    #     results['unbiased_no_backtracking_'] = (seconds, cover_time, undirected_edge_cover_time)
    # print("Unbiased + no backtracking done\n")

    # MDLR random walks
    for seed in range(n_seeds):
        config = RandomWalkConfig(min_degree=True, seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"MDLR (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
        results['MDLR_'] = (seconds, cover_time, undirected_edge_cover_time)
    print("MDLR done\n")

    # # MDLR random walks with no backtracking
    # for seed in range(n_seeds):
    #     config = RandomWalkConfig(min_degree=True, no_backtrack=True, seed=seed * 100)
    #     seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
    #     print(f"MDLR + no backtracking (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
    #     results['MDLR_no_backtrack'] = (seconds, cover_time, undirected_edge_cover_time)
    # print("MDLR + no backtracking done\n")

    # node2vec random walks
    for seed in range(n_seeds):
        config = RandomWalkConfig(p=0.25, q=0.25, seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"node2vec (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
        results['N2V'] = (seconds, cover_time, undirected_edge_cover_time)
    print("node2vec done\n")

    df = pd.DataFrame(results, index=["seconds", "vertex_cover_time", "edge_cover_time"]).T
    df.index.name = name

    df.to_csv(
        f"cover_time_results_{name}.csv",
        mode="a",             # append # only write header once
        index=True
    )
    print(f"saved → cover_time_results_{name}.csv")
    print(df)
    return results

    

def main(N=5):
    G = build_tree_graph(branching_factor=2, levels=3)
    #  G = build_bottleneck_sbm(n1=3, n2=5, p_intra1=0.8, p_intra2=0.8, p_inter=0.06)
    #  G = build_lollipop_graph(n_lollipops=N, lollipop_size=5)
    print("Clique chain:", G.number_of_nodes(), G.number_of_edges())
    quick_plot(G, "Hierarchical clique chain", "clique_chain.pdf")

    run_all_tests(G, f"sbm_(n={G.number_of_nodes()},m={G.number_of_edges()})", n_seeds=1)


if __name__ == '__main__':
    fire.Fire(main)