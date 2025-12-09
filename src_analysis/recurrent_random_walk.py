import os
import sys 
# add path to import graph_walker
sys.path.append(os.path.abspath('/home/aifb/cc7738/random-walk/graph-walker/graph_walker'))
import time
import numpy as np
import networkx as nx
import tqdm
import random_walks as random_walks  # pylint: disable=import-error
import torch
import fire


import networkx as nx
import torch
from torch_sparse import SparseTensor

def nx_to_sparse_tensor(G, device=torch.device("cpu")) -> SparseTensor:
    """
    Convert NetworkX graph with nodes labeled 0..n-1
    into a symmetric torch_sparse SparseTensor.
    """
    n = G.number_of_nodes()
    assert set(G.nodes()) == set(range(n)), \
        "Graph nodes must be exactly 0..n-1"

    row = []
    col = []
    for u, v in G.edges():
        # undirected → store both directions
        row.append(u); col.append(v)
        row.append(v); col.append(u)

    row = torch.tensor(row, dtype=torch.long, device=device)
    col = torch.tensor(col, dtype=torch.long, device=device)

    return SparseTensor(row=row, col=col, sparse_sizes=(n, n))



def get_recurrent_walks_for_cover(
    adj: SparseTensor,
    start_nodes: torch.Tensor,
    walk_length: int,
    num_walks: int,
    steps: int,
    flip: bool = True,
):
    device = adj.storage.row().device
    start_nodes = start_nodes.to(device)

    current_sources = start_nodes
    all_walks = []

    print(f"--- Recurrent random walks for cover (steps={steps}) ---")

    for step in range(steps):
        num_sources = current_sources.size(0)

        # 1) Repeat sources for num_walks
        #    shape: (num_sources * num_walks,)
        sources_repeated = current_sources.repeat_interleave(num_walks)
        num_total = sources_repeated.size(0)

        # 2) Random walk:
        #    torch_sparse.random_walk(seeds, walk_length-1)
        #    returns a tensor of shape (num_seeds, walk_length)
        walks = adj.random_walk(sources_repeated, walk_length - 1)
        walks = walks.view(num_total, walk_length)  # ensure correct shape

        # 3) Reshape to (num_sources, num_walks, walk_length)
        walks = walks.view(num_sources, num_walks, walk_length)

        # 4) Flip if needed (HeART uses reversed walks)
        if flip:
            walks = torch.flip(walks, dims=[-1])

        # 5) Treat each (source, k) walk as an independent walk
        walks_flat = walks.view(num_sources * num_walks, walk_length)
        all_walks.append(walks_flat)

        # 6) Recurrence: all visited nodes become sources for the next step
        if steps > 1 and step < steps - 1:
            current_sources = walks_flat.reshape(-1)

        print(
            f"Step {step+1}/{steps}: "
            f"generated {walks_flat.size(0)} walks, "
            f"next sources: {current_sources.size(0)}"
        )

    # Concatenate all recurrent steps
    walks_all = torch.cat(all_walks, dim=0)  # (total_walks_over_all_steps, walk_length)

    # No restarts in this scheme → all False
    restarts = torch.zeros_like(walks_all, dtype=torch.bool)

    # Convert to numpy for your existing compute_cover_times
    walks_np = walks_all.cpu().numpy()
    restarts_np = restarts.cpu().numpy()

    return walks_np, restarts_np



def _sample_neighbor(indices, weights, rng):
    """
    Sample an index into `indices` with unnormalized weights `weights`.
    Returns indices[pos].
    """
    weight_sum = weights.sum()
    if weight_sum <= 0:
        # all weights zero, fall back to uniform
        return rng.choice(indices)

    draw = rng.random() * weight_sum
    cumsum = 0.0
    for idx, w in zip(indices, weights):
        cumsum += w
        if draw <= cumsum:
            return idx
    # numerical fall-back
    return indices[-1]



def random_walks_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len):
    """
    Python/Numpy version of C++ randomWalksNoBacktrack.
    Same as random_walks, but avoids immediately going back to the previous node.
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)

        step = int(start_nodes[i % n_nodes])
        walks[i, 0] = step

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors → stay
            if start == end:
                walks[i, k] = step
                continue

            neigh_idx = indices[start:end]
            weights = data[start:end].copy()

            if k >= 2:
                prev = int(walks[i, k - 2])

                # set weight to 0 for prev node
                for z in range(start, end):
                    if indices[z] == prev:
                        weights[z - start] = 0.0

            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step

    return walks


import numpy as np
import torch
from torch_sparse import SparseTensor
import scipy.sparse as sp

# assumes _sample_neighbor + random_walks_no_backtrack are already defined
# exactly as in your snippet



def generate_walks_for_cover_time(
    adj: SparseTensor,
    num_nodes: int,
    walk_length: int,
    num_walks_per_node: int,
):
    """
    Generate unbiased random walks suitable for cover-time computation.

    Each node 0..num_nodes-1 is used as a start node, and from each start node
    we generate `num_walks_per_node` walks of length `walk_length`.

    Args:
        adj:               torch_sparse.SparseTensor adjacency (n x n).
        num_nodes:         number of nodes in the graph (assumed 0..num_nodes-1).
        walk_length:       length of each walk (number of nodes in the sequence).
        num_walks_per_node:number of walks starting from each node.´

    Returns:
        walks_np:    (num_nodes * num_walks_per_node, walk_length) np.int64
        restarts_np: same shape, np.bool_ (all False → no restarts)
    """
    device = adj.storage.row().device

    # Start from every node
    start_nodes = torch.arange(num_nodes, device=device)  # [0, 1, ..., num_nodes-1]

    # Repeat each start node num_walks_per_node times
    # shape: (num_nodes * num_walks_per_node,)
    sources_repeated = start_nodes.repeat_interleave(num_walks_per_node)
    num_total_walks = sources_repeated.size(0)

    # torch_sparse.random_walk(seeds, walk_length-1)
    # returns a tensor of shape (num_seeds, walk_length)
    walks = adj.random_walk(sources_repeated, walk_length - 1)  # (num_total_walks, walk_length)

    # Ensure shape is [num_total_walks, walk_length]
    walks = walks.view(num_total_walks, walk_length)

    # No restarts in this scheme → all False
    restarts = torch.zeros_like(walks, dtype=torch.bool)

    # Convert to numpy for your existing cover-time functions
    walks_np = walks.cpu().numpy()
    restarts_np = restarts.cpu().numpy()

    return walks_np, restarts_np


def compute_random_walks_torch(G, config):
    """
    Torch_sparse-based replacement of graph_walker.random_walks
    Returns:
        walks:    np.ndarray [num_walks, walk_length]
        restarts: np.ndarray [num_walks, walk_length] all False (no restart)
    Behavior matches compute_random_walks but:
        - p, q, alpha, k, min_degree, no_backtrack are IGNORED for now
        - OPTIONAL recurrent expansion if config.recurrent_steps > 1
    """
    # Build torch adjacency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = nx_to_sparse_tensor(G, device=device)

    # Seed
    if config.seed is not None:
        torch.manual_seed(config.seed)

    n_nodes = G.number_of_nodes()
    walk_len = config.walk_len
    num_walks_per_node = config.n_walks

    # IF using recurrence (HeART expansion)
    if hasattr(config, "recurrent_steps") and config.recurrent_steps > 1:
        start_nodes = torch.arange(n_nodes, device=device)
        walks, restarts = get_recurrent_walks_for_cover(
            adj=adj,
            start_nodes=start_nodes,
            walk_length=walk_len,
            num_walks=num_walks_per_node,
            steps=config.recurrent_steps,
        )
        return walks, restarts




def compute_stationary_distribution(G, config):
    """Compute the stationary distribution of the random walk."""
    probs = random_walks.stationary_distribution(
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


def compute_node_cover_times(
    G: nx.Graph,
    walks: np.ndarray,
    restarts: np.ndarray,
) -> np.ndarray:
    """
    Compute node cover time per starting node.

    Args:
        G:        NetworkX graph with nodes labeled 0..n-1.
        walks:    (num_walks, walk_length) int array of node IDs.
        restarts: (num_walks, walk_length) bool array; True indicates a restart
                  at that position (we assume restart means jump to start node).

    Returns:
        cover_times_per_start_node: (n_nodes,) float64
            For each node v, the average (over all walks starting at v) of
            the first time step t where all nodes have been visited.
            If some start node has no valid covering walks, its entry is -1.
    """
    # basic checks
    n_nodes = G.number_of_nodes()
    assert set(G.nodes()) == set(range(n_nodes)), \
        "G's nodes must be exactly 0..n-1 for array indexing."
    num_walks, walk_length = walks.shape
    assert restarts.shape == walks.shape

    # cover time for each individual walk
    cover_times = np.full(num_walks, -1, dtype=np.int64)
    start_nodes = walks[:, 0]

    for walk_idx in range(num_walks):
        walk = walks[walk_idx]
        restart = restarts[walk_idx]
        start_node = walk[0]

        visited = np.zeros(n_nodes, dtype=bool)

        for t in range(walk_length - 1):
            i = walk[t]
            j = walk[t + 1]
            rs = restart[t + 1]

            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                # restart to start_node: skip marking this step as transition
                assert j == start_node, "Restart node must be start node"
                continue

            # mark nodes visited
            visited[i] = True
            visited[j] = True

            if visited.all():
                cover_times[walk_idx] = t + 1
                break

    # average cover time per starting node
    cover_times_per_start = np.full(n_nodes, -1.0, dtype=np.float64)

    for node in range(n_nodes):
        idxs = np.where(start_nodes == node)[0]
        if len(idxs) == 0:
            continue
        valid = cover_times[idxs] != -1
        if valid.any():
            cover_times_per_start[node] = cover_times[idxs][valid].mean()

    return cover_times_per_start


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
        # Recurrent (HeART-style) walks
        start_nodes = torch.arange(n_nodes, device=device)
        walks, restarts = get_recurrent_walks_for_cover(
            adj=adj,
            start_nodes=start_nodes,
            walk_length=config.walk_len,
            num_walks=config.n_walks,
            steps=recurrent_steps,
            flip=True,  # keep HeART convention
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

    node_cover_times = compute_node_cover_times(G, walks, restarts)
    cover_time = float(node_cover_times.max())

    undirected_edge_cover_times = compute_undirected_edge_cover_times(G, walks, restarts)
    undirected_edge_cover_time = int(undirected_edge_cover_times.max())

    return (
        seconds,
        cover_time,
        undirected_edge_cover_time,
    )


