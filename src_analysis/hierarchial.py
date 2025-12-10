
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def hierarchical_tree_connected(levels: int, cluster_size: int,
                                p_intra: float = 0.8,
                                p_inter_extra: float = 0.2,
                                seed: int = 0) -> nx.Graph:
    """
    Build a hierarchical binary tree of clusters.
    - `levels`: tree depth (0..levels-1)
    - each cluster has `cluster_size` nodes
    - within each cluster: edges with prob `p_intra`
    - between each child cluster and its parent cluster:
        - at least one edge is added deterministically
        - plus extra random edges with prob `p_inter_extra`
    This guarantees connectivity.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    node_id = 0
    clusters = []  # clusters[level][idx] = list of nodes

    for lvl in range(levels):
        new_clusters = []
        for _ in range(2**lvl):
            nodes = list(range(node_id, node_id + cluster_size))
            node_id += cluster_size
            G.add_nodes_from(nodes)
            new_clusters.append(nodes)

            # intra-cluster edges
            for u in nodes:
                for v in nodes:
                    if u < v and rng.random() < p_intra:
                        G.add_edge(u, v)

        clusters.append(new_clusters)

    # connect clusters along the tree
    for lvl in range(levels - 1):
        for i, child_cluster in enumerate(clusters[lvl]):
            parent_cluster = clusters[lvl + 1][i // 2]

            # guarantee at least one parent-child bridge
            G.add_edge(child_cluster[0], parent_cluster[0])

            # add extra random parent-child edges
            for u in child_cluster:
                for v in parent_cluster:
                    if rng.random() < p_inter_extra:
                        G.add_edge(u, v)

    assert nx.is_connected(G)
    return G


def build_tree_graph(branching_factor: int = 2, levels: int = 4) -> nx.Graph:
    """
    Build a balanced tree with given branching factor and depth.
    Connectivity guaranteed.

    Args:
        branching_factor : number of children per node
        levels           : depth of the tree 

    Returns:
        G : connected tree graph
    """
    G = nx.balanced_tree(r=branching_factor, h=levels)
    assert nx.is_tree(G), "Graph must be a tree"
    assert nx.is_connected(G), "Tree must be connected"
    return G


# ============================================================
# 2. BOTTLENECK SBM (two dense blocks, sparse bridge) - force connected
# ============================================================

def build_bottleneck_sbm(n1: int = 50, n2: int = 50,
                         p_intra1: float = 0.4,
                         p_intra2: float = 0.4,
                         p_inter: float = 0.01,
                         seed: int = 0) -> nx.Graph:
    """
    Two-community SBM with sparse inter-community edges.
    Connectivity is guaranteed by enforcing at least one bridge edge.
    """
    rng = np.random.default_rng(seed)
    sizes = [n1, n2]
    probs = [
        [p_intra1,    p_inter],
        [p_inter,     p_intra2],
    ]
    G = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(1_000_000)))

    # ensure at least one inter-block edge
    # block 0: nodes [0 .. n1-1], block 1: [n1 .. n1+n2-1]
    u = 0
    v = n1
    if not G.has_edge(u, v):
        G.add_edge(u, v)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(
    G.subgraph(max(nx.connected_components(G), key=len)).copy(),
    first_label=0)
    assert nx.is_connected(G)
    return G



def build_lollipop_graph(N: int, clique_factor: int = 2) -> nx.Graph:
    """
    Build a lollipop graph consisting of:
    - a clique of size clique_factor * N
    - a path (tail) of length N
    - a single bridge edge connecting clique to tail

    Args:
        N : length of the path
        clique_factor : multiplier for clique size (default = 2, so clique size = 2N)

    Returns:
        G : connected lollipop graph
    """
    clique_size = clique_factor * N

    clique = nx.complete_graph(clique_size)
    chain = nx.path_graph(N)

    # disjoint union relabels second block automatically
    G = nx.disjoint_union(clique, chain)

    # add bridge between end of clique and start of chain
    G.add_edge(clique_size - 1, clique_size)

    G = nx.to_undirected(G)
    assert nx.is_connected(G), "Lollipop graph must be connected"

    return G


import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# 1. BARBELL GRAPH (two cliques + thin path) - always connected
# ============================================================

def build_barbell_graph(clique_size: int = 20, bridge_len: int = 10) -> nx.Graph:
    """
    Classic 'barbell': K_k -- path -- K_k
    This is always connected by construction.
    Args:
        clique_size: size of each clique (k)
        bridge_len: length of the path between cliques (number of edges),
                    if 0 => single edge between cliques
    """
    if bridge_len <= 0:
        # standard barbell: two cliques joined by a single edge
        G = nx.barbell_graph(clique_size, 0)
        assert nx.is_connected(G)
        return G

    # left and right cliques
    left = nx.complete_graph(clique_size)
    right = nx.complete_graph(clique_size)
    right = nx.relabel_nodes(right, lambda i: i + clique_size)

    # path in the middle
    path = nx.path_graph(bridge_len)
    path = nx.relabel_nodes(path, lambda i: i + 2 * clique_size)

    # union all
    G = nx.union_all([left, right, path])

    # connect left clique to first path node
    G.add_edge(clique_size - 1, 2 * clique_size)

    # connect path end to right clique
    G.add_edge(2 * clique_size + bridge_len - 1, clique_size)

    assert nx.is_connected(G)
    return G



# ============================================================
# 3. HIERARCHICAL CLIQUE CHAIN (cliques on a line) - connected
# ============================================================

def build_hierarchical_clique_chain(num_levels: int = 4,
                                    base_clique_size: int = 8,
                                    shrink_factor: float = 0.5,
                                    p_bridge: float = 0.2,
                                    seed: int = 0) -> nx.Graph:
    """
    Chain of cliques with gradually shrinking size (hierarchical cores).
    Level 0: largest clique, Level L-1: smallest clique.

    For each adjacent pair of cliques (C_i, C_{i+1}):
      - ensure at least one deterministic bridge
      - add extra random bridges with prob p_bridge

    Always connected by construction.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    cliques = []
    node_offset = 0

    # construct cliques of descending size
    for lvl in range(num_levels):
        size = max(2, int(base_clique_size * (shrink_factor ** lvl)))
        nodes = list(range(node_offset, node_offset + size))
        node_offset += size
        G.add_nodes_from(nodes)

        # make clique
        for u in nodes:
            for v in nodes:
                if u < v:
                    G.add_edge(u, v)

        cliques.append(nodes)

    # connect adjacent cliques with bottlenecks
    for i in range(num_levels - 1):
        c1 = cliques[i]
        c2 = cliques[i + 1]

        # guarantee 1 bridge
        G.add_edge(c1[0], c2[0])

        # extra random bridges
        for u in c1:
            for v in c2:
                if rng.random() < p_bridge:
                    G.add_edge(u, v)

    assert nx.is_connected(G)
    return G


# ============================================================
# 4. CORE–PERIPHERY GRAPH - connected by design
# ============================================================

def build_core_periphery(core_size: int = 20,
                         periphery_size: int = 80,
                         p_core: float = 0.5,
                         p_periphery: float = 0.02,
                         p_core_periphery: float = 0.05,
                         seed: int = 0) -> nx.Graph:
    """
    Core–periphery graph:
      - dense core: Erdos-Renyi G(core_size, p_core)
      - sparse periphery: G(periphery_size, p_periphery)
      - core-periphery edges: prob p_core_periphery + at least 1 edge per periphery node

    Connected by construction (each periphery node gets at least one edge to core).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    # core: nodes 0..core_size-1
    core_nodes = list(range(core_size))
    G.add_nodes_from(core_nodes)
    for u in core_nodes:
        for v in core_nodes:
            if u < v and rng.random() < p_core:
                G.add_edge(u, v)

    # periphery: nodes core_size..core_size+periphery_size-1
    periph_nodes = list(range(core_size, core_size + periphery_size))
    G.add_nodes_from(periph_nodes)
    for u in periph_nodes:
        for v in periph_nodes:
            if u < v and rng.random() < p_periphery:
                G.add_edge(u, v)

    # connect periphery to core
    for u in periph_nodes:
        # ensure at least one edge to core
        v0 = rng.choice(core_nodes)
        G.add_edge(u, v0)
        # extra random edges to core
        for v in core_nodes:
            if rng.random() < p_core_periphery:
                G.add_edge(u, v)

    assert nx.is_connected(G)
    return G


# ============================================================
# 5. Quick visualization helper (optional)
# ============================================================

def quick_plot(G: nx.Graph, title: str, filename: str):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(4, 4))
    nx.draw(G, pos, node_size=40, linewidths=0, edge_color="gray")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")


# ============================================================
# 6. Demo: generate all and check connectivity
# ============================================================

if __name__ == "__main__":
    # Barbell
    G_barbell = build_barbell_graph(clique_size=15, bridge_len=10)
    print("Barbell:", G_barbell.number_of_nodes(), G_barbell.number_of_edges())
    quick_plot(G_barbell, "Barbell graph", "barbell_graph.pdf")

    # Bottleneck SBM
    G_sbm = build_bottleneck_sbm(n1=40, n2=40, p_intra1=0.4, p_intra2=0.4, p_inter=0.01)
    print("Bottleneck SBM:", G_sbm.number_of_nodes(), G_sbm.number_of_edges())
    quick_plot(G_sbm, "Bottleneck SBM", "bottleneck_sbm.pdf")

    # Hierarchical clique chain
    G_chain = build_hierarchical_clique_chain(num_levels=4,
                                              base_clique_size=16,
                                              shrink_factor=0.6,
                                              p_bridge=0.1)
    print("Clique chain:", G_chain.number_of_nodes(), G_chain.number_of_edges())
    quick_plot(G_chain, "Hierarchical clique chain", "clique_chain.pdf")

    # Core–periphery
    G_cp = build_core_periphery(core_size=20,
                                periphery_size=60,
                                p_core=0.5,
                                p_periphery=0.01,
                                p_core_periphery=0.02)
    print("Core-periphery:", G_cp.number_of_nodes(), G_cp.number_of_edges())
    quick_plot(G_cp, "Core–periphery", "core_periphery.pdf")
