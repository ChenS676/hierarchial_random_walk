import time
import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix


# ============================================================
# 1. Graph construction: Lollipop
# ============================================================

def build_lollipop_graph(tail_len: int = 20, clique_factor: int = 4) -> nx.Graph:
    """
    Build a lollipop graph: large clique + path tail + single bridge edge.

    Args:
        tail_len: length of the path tail (N).
        clique_factor: clique size = clique_factor * tail_len.

    Returns:
        G: undirected NetworkX graph with nodes labeled 0..n-1.
    """
    N = tail_len
    clique_size = clique_factor * N

    clique = nx.complete_graph(clique_size)
    chain = nx.path_graph(N)

    G = nx.disjoint_union(clique, chain)
    # bridge from last clique node to first tail node
    G.add_edge(clique_size - 1, clique_size)

    G = nx.convert_node_labels_to_integers(G, first_label=0)
    G = nx.to_undirected(G)
    return G


# ============================================================
# 2. CSR + transition probabilities
# ============================================================

def graph_to_csr_transition(G: nx.Graph):
    """
    Build CSR representation (indptr, indices, data) of the transition matrix
    for an unbiased random walk on G (row-normalized adjacency).
    """
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=float)
    # row-normalize
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    D_inv = 1.0 / row_sums
    A = A.multiply(D_inv[:, None])

    A = csr_matrix(A)
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)
    return indptr, indices, data


# ============================================================
# 3. Random walk kernels (NumPy versions)
# ============================================================

def _sample_neighbor(indices, weights, rng):
    """Sample index from 'indices' with unnormalized 'weights'."""
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return rng.choice(indices)

    draw = rng.random() * weight_sum
    cumsum = 0.0
    for idx, w in zip(indices, weights):
        cumsum += w
        if draw <= cumsum:
            return idx
    return indices[-1]


def random_walks(indptr, indices, data, start_nodes, seed, n_walks, walk_len):
    """
    Unbiased weighted random walks (row-normalized weights in 'data').
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

            if start == end:
                walks[i, k] = step
                continue

            neigh_idx = indices[start:end]
            weights = data[start:end].copy()
            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step

    return walks


def random_walks_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len):
    """
    Non-backtracking variant: zeroes the weight of the previous node.
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

            if start == end:
                walks[i, k] = step
                continue

            neigh_idx = indices[start:end]
            weights = data[start:end].copy()

            if k >= 2:
                prev = int(walks[i, k - 2])
                # zero weight for prev
                for z in range(start, end):
                    if indices[z] == prev:
                        weights[z - start] = 0.0

            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step

    return walks


def n2v_random_walks(indptr, indices, data, start_nodes, seed, n_walks, walk_len, p, q):
    """
    node2vec-style biased random walks (p, q).
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

            if start == end:
                walks[i, k] = step
                continue

            neigh_idx = indices[start:end]
            weights = data[start:end].copy()

            if k >= 2:
                prev = int(walks[i, k - 2])
                prev_start = indptr[prev]
                prev_end = indptr[prev + 1]
                prev_neighbors = indices[prev_start:prev_end]

                for local_idx, z in enumerate(range(start, end)):
                    neighbor = indices[z]
                    w = weights[local_idx]
                    if neighbor == prev:
                        w = w / p
                    else:
                        if neighbor not in prev_neighbors:
                            w = w / q
                    weights[local_idx] = w

            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step

    return walks


# ============================================================
# 4. Recurrent / hierarchical walks
# ============================================================

def generate_recurrent_walks_for_cover_time(
    indptr, indices, data, num_nodes, walk_length, num_walks_per_node, seed
):
    """
    2-round recurrent random walks:

    Round 1:
      - From each node 0..num_nodes-1, run num_walks_per_node walks of length walk_length.

    Round 2:
      - For each Round-1 walk, take its middle node as a new start node,
        and run another walk of length walk_length.

    Returns:
        walks_all: (2 * num_nodes * num_walks_per_node, walk_length)
        restarts:  same shape, all False
    """
    start_nodes = np.arange(num_nodes, dtype=np.uint32)

    # Round 1
    walks1 = random_walks(indptr, indices, data, start_nodes,
                          seed=seed, n_walks=num_walks_per_node, walk_len=walk_length)
    num_total_walks = walks1.shape[0]

    # Round 2: from midpoints
    mid_idx = walk_length // 2
    mid_nodes = walks1[:, mid_idx]
    walks2 = random_walks(indptr, indices, data, mid_nodes,
                          seed=seed + 10_000, n_walks=1, walk_len=walk_length)

    walks_all = np.vstack([walks1, walks2])
    restarts = np.zeros_like(walks_all, dtype=bool)
    return walks_all, restarts


# ============================================================
# 5. Cover time computations
# ============================================================

def compute_node_cover_times(G: nx.Graph, walks: np.ndarray, restarts: np.ndarray) -> np.ndarray:
    """
    Per-start-node node cover time.

    For each walk, find the earliest step where all nodes have been visited.
    Then average over walks for each starting node.
    """
    n_nodes = G.number_of_nodes()
    assert set(G.nodes()) == set(range(n_nodes)), "Nodes must be 0..n-1"

    num_walks, walk_len = walks.shape
    assert restarts.shape == walks.shape

    cover_times = np.full(num_walks, -1, dtype=np.int64)
    start_nodes = walks[:, 0]

    for w_idx in range(num_walks):
        walk = walks[w_idx]
        restart = restarts[w_idx]
        start_node = walk[0]
        visited = np.zeros(n_nodes, dtype=bool)

        for t in range(walk_len - 1):
            i = walk[t]
            j = walk[t + 1]
            rs = restart[t + 1]

            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue

            visited[i] = True
            visited[j] = True

            if visited.all():
                cover_times[w_idx] = t + 1
                break

    cover_times_per_start = np.full(n_nodes, -1.0, dtype=np.float64)
    for node in range(n_nodes):
        idxs = np.where(start_nodes == node)[0]
        if len(idxs) == 0:
            continue
        valid = cover_times[idxs] != -1
        if valid.any():
            cover_times_per_start[node] = cover_times[idxs][valid].mean()

    return cover_times_per_start


def compute_undirected_edge_cover_times(G: nx.Graph, walks: np.ndarray, restarts: np.ndarray) -> np.ndarray:
    """
    Per-walk undirected edge cover time.

    For each walk, find the earliest step where all undirected edges have been traversed.
    """
    num_walks, walk_len = walks.shape

    # map undirected edge → index
    edges = {tuple(sorted(e)) for e in G.edges()}
    edges = sorted(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    n_edges = len(edges)

    cover_times = np.full(num_walks, -1, dtype=np.int64)

    for w_idx in range(num_walks):
        walk = walks[w_idx]
        restart = restarts[w_idx]

        visited = np.zeros(n_edges, dtype=bool)

        for t in range(walk_len - 1):
            i = int(walk[t])
            j = int(walk[t + 1])
            rs = restart[t + 1]

            if rs:
                # restart steps ignored for edges
                continue

            e = tuple(sorted((i, j)))
            if e in edge_to_idx:
                edge_idx = edge_to_idx[e]
                visited[edge_idx] = True

            if visited.all():
                cover_times[w_idx] = t + 1
                break

    return cover_times


# ============================================================
# 6. Running experiments on lollipop
# ============================================================

def run_method(G, indptr, indices, data,
               method: str,
               walk_length: int,
               num_walks_per_node: int,
               seed: int = 0):
    """
    Run a single random walk method and compute timings + cover times.

    Returns:
        (seconds, max_vertex_cover_time, max_edge_cover_time)
    """
    n_nodes = G.number_of_nodes()
    start_nodes = np.arange(n_nodes, dtype=np.uint32)

    t0 = time.time()

    if method == "unbiased":
        walks = random_walks(indptr, indices, data, start_nodes,
                             seed=seed, n_walks=num_walks_per_node, walk_len=walk_length)
        restarts = np.zeros_like(walks, dtype=bool)

    elif method == "unbiased_no_backtrack":
        walks = random_walks_no_backtrack(indptr, indices, data, start_nodes,
                                          seed=seed, n_walks=num_walks_per_node, walk_len=walk_length)
        restarts = np.zeros_like(walks, dtype=bool)

    elif method == "node2vec":
        walks = n2v_random_walks(indptr, indices, data, start_nodes,
                                 seed=seed, n_walks=num_walks_per_node, walk_len=walk_length,
                                 p=0.25, q=0.25)
        restarts = np.zeros_like(walks, dtype=bool)

    elif method == "recurrent":
        walks, restarts = generate_recurrent_walks_for_cover_time(
            indptr, indices, data,
            num_nodes=n_nodes,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    t1 = time.time()
    seconds = t1 - t0

    # vertex cover
    node_cover = compute_node_cover_times(G, walks, restarts)
    max_vertex_cover_time = float(node_cover.max())

    # edge cover
    edge_cover = compute_undirected_edge_cover_times(G, walks, restarts)
    max_edge_cover_time = int(edge_cover.max())

    return seconds, max_vertex_cover_time, max_edge_cover_time


def run_lollipop_experiments(
    tail_len: int = 20,
    clique_factor: int = 4,
    walk_length: int = 512,
    num_walks_per_node: int = 10,
    seed: int = 0,
):
    """
    Build a lollipop graph and compare multiple random-walk strategies.
    Returns a pandas DataFrame and also saves a CSV.
    """
    G = build_lollipop_graph(tail_len=tail_len, clique_factor=clique_factor)
    indptr, indices, data = graph_to_csr_transition(G)

    methods = [
        "recurrent",
        "unbiased",
        "unbiased_no_backtrack",
        "node2vec",
    ]

    results = {}

    for m in methods:
        print(f"Running method: {m}")
        secs, v_cover, e_cover = run_method(
            G, indptr, indices, data,
            method=m,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            seed=seed,
        )
        print(f"  {m}: time={secs:.3f}s, "
              f"max vertex cover={v_cover:.1f}, "
              f"max edge cover={e_cover}")

        results[m] = (secs, v_cover, e_cover)

    df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["seconds", "max_vertex_cover_time", "max_edge_cover_time"],
    )
    df.index.name = "method"

    csv_name = f"lollipop_cover_times_tail{tail_len}_clq{clique_factor}.csv"
    df.to_csv(csv_name)
    print(f"\nSaved results to {csv_name}\n")
    print(df)

    return df


import matplotlib.pyplot as plt

def plot_lollipop_results(df, title="Lollipop Coverage Comparison"):
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    x = range(len(df))
    ax.plot(x, df["max_vertex_cover_time"], marker="o", label="Vertex Cover Time")
    ax.plot(x, df["max_edge_cover_time"], marker="s", label="Edge Cover Time")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45)
    ax.set_ylabel("Steps")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def sweep_lollipop_sizes(
    tail_lens = (20, 40, 80),
    clique_factor: int = 4,
    base_walk_len: int = 512,
    num_walks_per_node: int = 10,
    seed: int = 42,
):
    all_dfs = []
    for N in tail_lens:
        # scale walk length ~ N^2 (relative to N=20 baseline)
        scale = (N / 20.0) ** 2
        walk_length = int(base_walk_len * scale)

        print(f"\n=== Running lollipop with tail_len={N}, clique_factor={clique_factor}, "
              f"walk_length={walk_length}, num_walks_per_node={num_walks_per_node} ===")

        df = run_lollipop_experiments(
            tail_len=N,
            clique_factor=clique_factor,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            seed=seed,
        )
        df["tail_len"] = N
        df["clique_factor"] = clique_factor
        df["walk_length"] = walk_length
        df["num_walks_per_node"] = num_walks_per_node
        all_dfs.append(df.reset_index())  # bring 'method' back as a column

    big_df = pd.concat(all_dfs, ignore_index=True)
    big_df.to_csv("lollipop_size_sweep_results.csv", index=False)
    print("\nSaved sweep results to lollipop_size_sweep_results.csv")
    return big_df

if __name__ == "__main__":
    # Example run where recurrent/hierarchical should have an advantage
    big_df = sweep_lollipop_sizes(
        tail_lens=[9, 12, 15, 18],
        clique_factor=4,
        base_walk_len=512,
        num_walks_per_node=10,
        seed=42,
    )
    print(big_df)

    # compare methods vs tail_len on max vertex cover time
    pivot = big_df.pivot_table(
        index="tail_len",
        columns="method",
        values="max_vertex_cover_time",
    )
    print(pivot)

