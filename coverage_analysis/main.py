#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import networkx as nx
import pandas as pd
import tqdm
import torch
from torch_sparse import SparseTensor

# If needed, add your local graph_walker path here:
# sys.path.append(os.path.abspath('/home/USER/random-walk/graph-walker/graph_walker'))
import graph_walker  # assumes graph_walker is importable


# ============================================================
# 1. Config
# ============================================================

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
        verbose=True,
        recurrent_steps=1,      # for torch recurrent RW
        use_torch=False,        # flag to switch backend if you want
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
        self.recurrent_steps = recurrent_steps
        self.use_torch = use_torch


# ============================================================
# 2. Torch adjacency & walk generators
# ============================================================

def nx_to_sparse_tensor(G: nx.Graph, device=None) -> SparseTensor:
    """
    Convert NetworkX graph with nodes labeled 0..n-1
    into a symmetric torch_sparse SparseTensor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = G.number_of_nodes()
    assert set(G.nodes()) == set(range(n)), "Graph nodes must be exactly 0..n-1"

    row = []
    col = []
    for u, v in G.edges():
        row.append(u); col.append(v)
        row.append(v); col.append(u)

    row = torch.tensor(row, dtype=torch.long, device=device)
    col = torch.tensor(col, dtype=torch.long, device=device)

    adj = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
    return adj


def get_recurrent_walks_for_cover(
    adj: SparseTensor,
    start_nodes: torch.Tensor,
    walk_length: int,
    num_walks: int,
    steps: int,
    flip: bool = True,
):
    """
    Recurrent random walks for cover-time computation.

    At each recurrent step:
      - From each current source node, run `num_walks` random walks of length walk_length.
      - Optionally reverse (flip) each walk.
      - All visited nodes become sources for the next step.

    Returns:
        walks_np:    (total_walks, walk_length) np.int64
        restarts_np: same shape, bool (all False)
    """
    device = adj.storage.row().device
    start_nodes = start_nodes.to(device)
    current_sources = start_nodes
    all_walks = []

    print(f"--- Recurrent random walks for cover (steps={steps}) ---")

    for step in range(steps):
        num_sources = current_sources.size(0)

        # repeat sources
        sources_repeated = current_sources.repeat_interleave(num_walks)
        num_total = sources_repeated.size(0)

        # random_walk from torch_sparse: seeds x walk_length
        walks = adj.random_walk(sources_repeated, walk_length - 1)
        walks = walks.view(num_total, walk_length)

        # reshape to (num_sources, num_walks, walk_length)
        walks = walks.view(num_sources, num_walks, walk_length)

        # flip if needed
        if flip:
            walks = torch.flip(walks, dims=[-1])

        walks_flat = walks.view(num_sources * num_walks, walk_length)
        all_walks.append(walks_flat)

        # recurrence: all visited nodes become new sources (except last step)
        if steps > 1 and step < steps - 1:
            current_sources = walks_flat.reshape(-1)

        print(
            f"Step {step+1}/{steps}: generated {walks_flat.size(0)} walks, "
            f"next sources: {current_sources.size(0)}"
        )

    walks_all = torch.cat(all_walks, dim=0)
    restarts = torch.zeros_like(walks_all, dtype=torch.bool)

    walks_np = walks_all.cpu().numpy()
    restarts_np = restarts.cpu().numpy()
    return walks_np, restarts_np


def generate_walks_for_cover_time(
    adj: SparseTensor,
    num_nodes: int,
    walk_length: int,
    num_walks_per_node: int,
):
    """
    Unbiased random walks, each node used as start node, num_walks_per_node walks.

    Returns:
        walks_np:    (num_nodes * num_walks_per_node, walk_length) np.int64
        restarts_np: same shape, bool (all False)
    """
    device = adj.storage.row().device
    start_nodes = torch.arange(num_nodes, device=device)

    sources_repeated = start_nodes.repeat_interleave(num_walks_per_node)
    num_total_walks = sources_repeated.size(0)

    walks = adj.random_walk(sources_repeated, walk_length - 1)
    walks = walks.view(num_total_walks, walk_length)

    restarts = torch.zeros_like(walks, dtype=torch.bool)

    walks_np = walks.cpu().numpy()
    restarts_np = restarts.cpu().numpy()
    return walks_np, restarts_np


# ============================================================
# 3. graph_walker backend
# ============================================================

def compute_random_walks(G, config: RandomWalkConfig):
    """
    Wrap graph_walker.random_walks with your config.
    """
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
        verbose=config.verbose,
    )
    return walks, restarts


def compute_stationary_distribution(G, config: RandomWalkConfig):
    return graph_walker.stationary_distribution(
        G,
        min_degree=config.min_degree,
        sub_sampling=config.sub_sampling,
    )


def compute_empirical_stationary_distribution(G, walks: np.ndarray):
    """
    Long-term empirical node distribution from walks.
    """
    unique, counts = np.unique(walks, return_counts=True)
    assert np.all(unique == np.arange(len(G.nodes)))
    return counts / np.sum(counts)


# ============================================================
# 4. Cover time computations
# ============================================================

def compute_cover_times(G: nx.Graph, walks: np.ndarray, restarts: np.ndarray):
    """
    Node cover time per starting node (graph_walker style).

    Returns:
        cover_times_per_start_node: shape (n_nodes,)
    """
    n_nodes = G.number_of_nodes()
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1

    for walk_idx, (walk, restart) in tqdm.tqdm(
        enumerate(zip(walks, restarts)), total=walks.shape[0]
    ):
        start_node = walk[0]
        visited = np.zeros(n_nodes, dtype=bool)

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

    assert np.all(cover_times != -1), "All walks must cover all nodes"

    cover_times_per_start_node = np.zeros(n_nodes, dtype=np.float32) - 1
    start_nodes = walks[:, 0]
    for start_node in np.unique(start_nodes):
        idxs = np.where(start_nodes == start_node)[0]
        cover_times_per_start_node[start_node] = np.mean(cover_times[idxs])

    assert np.all(
        cover_times_per_start_node != -1
    ), "All nodes must be used as start nodes"
    return cover_times_per_start_node


def compute_node_cover_times(G: nx.Graph, walks: np.ndarray, restarts: np.ndarray):
    """
    Variant: node cover time per starting node, used in torch backend.
    Very similar to compute_cover_times, but a bit more explicit.
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


def compute_undirected_edge_cover_times(G: nx.Graph, walks: np.ndarray, restarts: np.ndarray):
    """
    Per-walk undirected edge cover time: earliest step t where that walk
    has traversed all edges (as undirected).
    """
    num_walks, walk_len = walks.shape

    # build edge index: undirected edges
    edges = {tuple(sorted(e)) for e in G.edges()}
    edges = sorted(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    n_edges = len(edges)

    cover_times = np.full(num_walks, -1, dtype=np.int64)

    for w_idx in range(num_walks):
        walk = walks[w_idx]
        restart = restarts[w_idx]
        visited = np.zeros(n_edges, dtype=bool)

        for t in range(1, walk_len):
            if restart[t]:
                continue
            u = int(walk[t - 1])
            v = int(walk[t])
            e = tuple(sorted((u, v)))
            if e in edge_to_idx:
                idx = edge_to_idx[e]
                visited[idx] = True
            if visited.all():
                cover_times[w_idx] = t
                break

    return cover_times


# ============================================================
# 5. Backends: graph_walker & torch recurrent
# ============================================================

def run_tests(G: nx.Graph, config: RandomWalkConfig):
    """
    Standard graph_walker backend.
    """
    t0 = time.time()
    walks, restarts = compute_random_walks(G, config)
    seconds = time.time() - t0

    # stationary check only for unbiased SRW
    if (
        config.p == 1
        and config.q == 1
        and config.alpha == 0
        and config.k is None
        and not config.no_backtrack
    ):
        sample_probs = compute_empirical_stationary_distribution(G, walks)
        true_probs = compute_stationary_distribution(G, config)
        err = np.linalg.norm(sample_probs - true_probs)
        print(f"[run_tests] Stationary distribution error: {err:.2e}")

    node_cover = compute_cover_times(G, walks, restarts)
    cover_time = float(node_cover.max())

    edge_cover = compute_undirected_edge_cover_times(G, walks, restarts)
    undirected_edge_cover_time = int(edge_cover.max())

    return seconds, cover_time, undirected_edge_cover_time


def run_tests_torch(
    G: nx.Graph,
    config: RandomWalkConfig,
    use_recurrent: bool = True,
    recurrent_steps: int = None,
):
    """
    Torch-based analogue of run_tests(G, config).
    Uses:
      - generate_walks_for_cover_time   (unbiased)
      - get_recurrent_walks_for_cover   (HeART-style recurrent)
      - compute_node_cover_times
      - compute_undirected_edge_cover_times
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    adj = nx_to_sparse_tensor(G, device=device)
    n_nodes = G.number_of_nodes()

    if config.seed is not None:
        torch.manual_seed(config.seed)

    if recurrent_steps is None:
        recurrent_steps = getattr(config, "recurrent_steps", 1)

    t0 = time.time()

    if use_recurrent and recurrent_steps > 1:
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
        walks, restarts = generate_walks_for_cover_time(
            adj=adj,
            num_nodes=n_nodes,
            walk_length=config.walk_len,
            num_walks_per_node=config.n_walks,
        )

    seconds = time.time() - t0

    node_cover_times = compute_node_cover_times(G, walks, restarts)
    cover_time = float(node_cover_times.max())

    edge_cover_times = compute_undirected_edge_cover_times(G, walks, restarts)
    undirected_edge_cover_time = int(edge_cover_times.max())

    return seconds, cover_time, undirected_edge_cover_time


# ============================================================
# 6. Graph generators for families
# ============================================================

def make_lollipop(n_clique=20, tail_len=40):
    clique = nx.complete_graph(n_clique)
    tail = nx.path_graph(tail_len)
    G = nx.disjoint_union(clique, tail)
    G.add_edge(n_clique - 1, n_clique)
    return nx.to_undirected(G)


def make_barbell(n_clique=20, bridge_len=40):
    left = nx.complete_graph(n_clique)
    bridge = nx.path_graph(bridge_len)
    right = nx.complete_graph(n_clique)

    G = nx.disjoint_union(left, bridge)
    G = nx.disjoint_union(G, right)

    offset_bridge = n_clique
    offset_right = n_clique + bridge_len

    G.add_edge(n_clique - 1, offset_bridge)
    G.add_edge(offset_bridge + bridge_len - 1, offset_right)

    return nx.to_undirected(G)


def make_dumbbell(cycle_len=40, bridge_len=40):
    left = nx.cycle_graph(cycle_len)
    bridge = nx.path_graph(bridge_len)
    right = nx.cycle_graph(cycle_len)

    G = nx.disjoint_union(left, bridge)
    G = nx.disjoint_union(G, right)

    offset_bridge = cycle_len
    offset_right = cycle_len + bridge_len

    G.add_edge(cycle_len - 1, offset_bridge)
    G.add_edge(offset_bridge + bridge_len - 1, offset_right)

    return nx.to_undirected(G)


def make_modular_community_graph(n_per_block=50, p_in=0.2, p_out=0.005, seed=0):
    sizes = [n_per_block, n_per_block]
    probs = [[p_in, p_out],
             [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return nx.to_undirected(G)


def make_star_with_tail(n_leaves=50, tail_len=50):
    star = nx.star_graph(n_leaves)   # center 0, leaves 1..n_leaves
    tail = nx.path_graph(tail_len)   # 0..tail_len-1
    G = nx.disjoint_union(star, tail)
    offset_tail = n_leaves + 1
    G.add_edge(0, offset_tail)
    return nx.to_undirected(G)


# ============================================================
# 7. RW vs HRW comparison on one graph
# ============================================================

def compare_rw_vs_hrw_on_graph(
    G: nx.Graph,
    name: str,
    walk_len: int = 2000,
    n_walks_per_node: int = 10,
    recurrent_steps: int = 2,
    seeds = (0, 1, 2, 3, 4),
):
    """
    Runs:
      - HRW (recurrent) via run_tests_torch
      - Unbiased RW via run_tests
    Aggregates mean cover times over seeds.
    """
    hrw_vertex = []
    hrw_edge   = []
    hrw_time   = []

    rw_vertex  = []
    rw_edge    = []
    rw_time    = []

    for s in seeds:
        # HRW
        config_hrw = RandomWalkConfig(
            n_walks=n_walks_per_node,
            walk_len=walk_len,
            seed=s * 100,
            recurrent_steps=recurrent_steps,
        )
        secs, cv, ce = run_tests_torch(
            G,
            config_hrw,
            use_recurrent=True,
            recurrent_steps=recurrent_steps,
        )
        hrw_time.append(secs)
        hrw_vertex.append(cv)
        hrw_edge.append(ce)

        # Unbiased RW baseline
        config_rw = RandomWalkConfig(
            n_walks=n_walks_per_node,
            walk_len=walk_len,
            seed=s * 100,
        )
        secs_rw, cv_rw, ce_rw = run_tests(G, config_rw)
        rw_time.append(secs_rw)
        rw_vertex.append(cv_rw)
        rw_edge.append(ce_rw)

    res = {
        "graph": name,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "walk_len": walk_len,
        "n_walks_per_node": n_walks_per_node,
        "recurrent_steps": recurrent_steps,

        "HRW_vertex_cover_mean": float(np.mean(hrw_vertex)),
        "HRW_edge_cover_mean": float(np.mean(hrw_edge)),
        "HRW_time_mean": float(np.mean(hrw_time)),

        "RW_vertex_cover_mean": float(np.mean(rw_vertex)),
        "RW_edge_cover_mean": float(np.mean(rw_edge)),
        "RW_time_mean": float(np.mean(rw_time)),

        "vertex_cover_ratio_HRW_over_RW": float(np.mean(hrw_vertex) / np.mean(rw_vertex)
                                                if np.mean(rw_vertex) > 0 else np.nan),
        "edge_cover_ratio_HRW_over_RW": float(np.mean(hrw_edge) / np.mean(rw_edge)
                                              if np.mean(rw_edge) > 0 else np.nan),
    }

    print(f"\n=== {name} ===")
    print(f"Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
    print(f"HRW vertex cover (mean): {res['HRW_vertex_cover_mean']:.2f}")
    print(f"RW  vertex cover (mean): {res['RW_vertex_cover_mean']:.2f}")
    print(f"HRW / RW vertex ratio:   {res['vertex_cover_ratio_HRW_over_RW']:.3f}")
    print(f"HRW edge cover (mean):   {res['HRW_edge_cover_mean']:.2f}")
    print(f"RW  edge cover (mean):   {res['RW_edge_cover_mean']:.2f}")
    print(f"HRW / RW edge ratio:     {res['edge_cover_ratio_HRW_over_RW']:.3f}")
    return res


# ============================================================
# 8. Run all graph families & summarize
# ============================================================

def run_all_graph_families():
    graphs = [
        ("Lollipop",              make_lollipop(n_clique=20, tail_len=40)),
        ("Barbell",               make_barbell(n_clique=20, bridge_len=40)),
        ("Dumbbell",              make_dumbbell(cycle_len=40, bridge_len=40)),
        ("ModularCommunity",      make_modular_community_graph(n_per_block=50, p_in=0.2, p_out=0.005, seed=0)),
        ("StarWithTail",          make_star_with_tail(n_leaves=50, tail_len=50)),
    ]

    all_results = []
    for name, G in graphs:
        res = compare_rw_vs_hrw_on_graph(
            G,
            name=name,
            walk_len=2000,
            n_walks_per_node=10,
            recurrent_steps=2,
            seeds=[0, 1, 2, 3, 4],
        )
        all_results.append(res)

    df = pd.DataFrame(all_results)
    df.to_csv("rw_vs_hrw_graph_families.csv", index=False)
    print("\nSaved results to rw_vs_hrw_graph_families.csv")
    print(df)
    return df


def summarize_family_table(df: pd.DataFrame):
    rows = []
    for name in df["graph"].unique():
        sub = df[df["graph"] == name].iloc[0]
        v_ratio = sub["vertex_cover_ratio_HRW_over_RW"]
        e_ratio = sub["edge_cover_ratio_HRW_over_RW"]

        if name == "Lollipop":
            theory_rw = r"$\Theta(n^3)$"
            theory_hrw = r"$\tilde O(n^2)$"
        elif name == "Barbell":
            theory_rw = r"$\Theta(n^3)$"
            theory_hrw = r"order reduction"
        elif name == "Dumbbell":
            theory_rw = r"$\Theta(n^3)$"
            theory_hrw = r"order reduction"
        elif name == "ModularCommunity":
            theory_rw = r"slow mixing"
            theory_hrw = r"faster crossing"
        elif name == "StarWithTail":
            theory_rw = r"hub trapping"
            theory_hrw = r"frontier helps"
        else:
            theory_rw = ""
            theory_hrw = ""

        rows.append(dict(
            GraphStructure=name,
            RW_theory=theory_rw,
            HRW_theory= theory_hrw,
            HRW_over_RW_vertex=f"{v_ratio:.3f}",
            HRW_over_RW_edge=f"{e_ratio:.3f}",
        ))

    summary_df = pd.DataFrame(rows)

    print("\nMarkdown-style summary table:\n")
    print("| Graph structure | RW cover time (theory) | HRW improvement (theory) | HRW/RW vertex cover (empirical) | HRW/RW edge cover (empirical) |")
    print("|-----------------|------------------------|---------------------------|----------------------------------|-------------------------------|")
    for _, r in summary_df.iterrows():
        print(f"| {r['GraphStructure']} | {r['RW_theory']} | {r['HRW_theory']} | {r['HRW_over_RW_vertex']} | {r['HRW_over_RW_edge']} |")

    return summary_df


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    df_families = run_all_graph_families()
    summary_df = summarize_family_table(df_families)
