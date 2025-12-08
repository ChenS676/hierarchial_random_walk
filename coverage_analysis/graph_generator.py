import networkx as nx
import numpy as np
import pandas as pd


# 1) Lollipop: clique + path
def make_lollipop(n_clique=20, tail_len=40):
    clique = nx.complete_graph(n_clique)
    tail = nx.path_graph(tail_len)
    G = nx.disjoint_union(clique, tail)
    # connect last clique node to first tail node
    G.add_edge(n_clique - 1, n_clique)
    return nx.to_undirected(G)


# 2) Barbell: clique – path – clique
def make_barbell(n_clique=20, bridge_len=40):
    # networkx has a standard barbell_graph, but uses a single edge between cliques.
    # We want clique–path–clique explicitly.
    left = nx.complete_graph(n_clique)
    bridge = nx.path_graph(bridge_len)
    right = nx.complete_graph(n_clique)

    # disjoint union
    G = nx.disjoint_union(left, bridge)        # nodes: [0..n_clique-1], [n_clique..n_clique+bridge_len-1]
    G = nx.disjoint_union(G, right)           # add right clique

    offset_bridge = n_clique
    offset_right = n_clique + bridge_len

    # connect left clique to bridge start
    G.add_edge(n_clique - 1, offset_bridge)   # left clique last → first bridge node
    # connect bridge end to right clique start
    G.add_edge(offset_bridge + bridge_len - 1, offset_right)

    return nx.to_undirected(G)


# 3) Dumbbell: cycle – bridge – cycle
def make_dumbbell(cycle_len=40, bridge_len=40):
    left = nx.cycle_graph(cycle_len)
    bridge = nx.path_graph(bridge_len)
    right = nx.cycle_graph(cycle_len)

    G = nx.disjoint_union(left, bridge)
    G = nx.disjoint_union(G, right)

    offset_bridge = cycle_len
    offset_right = cycle_len + bridge_len

    # connect left cycle last node to bridge start
    G.add_edge(cycle_len - 1, offset_bridge)
    # connect bridge end to right cycle start
    G.add_edge(offset_bridge + bridge_len - 1, offset_right)

    return nx.to_undirected(G)


# 4) Highly modular community graph (2-block SBM)
def make_modular_community_graph(n_per_block=50, p_in=0.2, p_out=0.005, seed=0):
    sizes = [n_per_block, n_per_block]
    probs = [[p_in, p_out],
             [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return nx.to_undirected(G)


# 5) Hub–spoke / star-with-tail: star + path attached to hub
def make_star_with_tail(n_leaves=50, tail_len=50):
    # star: center node 0, leaves 1..n_leaves
    star = nx.star_graph(n_leaves)  # nodes: 0..n_leaves
    tail = nx.path_graph(tail_len)  # nodes: 0..tail_len-1

    G = nx.disjoint_union(star, tail)
    offset_tail = n_leaves + 1

    # attach tail to the hub (node 0)
    G.add_edge(0, offset_tail)  # hub → first tail node
    return nx.to_undirected(G)
