def compare_rw_vs_hrw_on_graph(
    G,
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
    on the same graph, aggregates mean cover times over seeds.

    Returns a dict with averaged metrics.
    """
    hrw_vertex = []
    hrw_edge   = []
    hrw_time   = []

    rw_vertex  = []
    rw_edge    = []
    rw_time    = []

    for s in seeds:
        # HRW / recurrent
        config_hrw = RandomWalkConfig(
            n_walks=n_walks_per_node,
            walk_len=walk_len,
            seed=s * 100,
        )
        # we pass use_recurrent=True, recurrent_steps=recurrent_steps
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

from graph_generator import (
	make_lollipop,
	make_barbell,
	make_dumbbell,
	make_modular_community_graph,
	make_star_with_tail
)
import pandas as pd

def run_all_graph_families():
    # Build graphs
    graphs = [
        ("Lollipop",              make_lollipop(n_clique=20, tail_len=40)),
        ("Barbell",               make_barbell(n_clique=20, bridge_len=40)),
        ("Dumbbell",              make_dumbbell(cycle_len=40, bridge_len=40)),
        ("ModularCommunity",      make_modular_community_graph(n_per_block=50, p_in=0.2, p_out=0.005, seed=0)),
        ("StarWithTail",          make_star_with_tail(n_leaves=50, tail_len=50)),
    ]

    all_results = []
    for name, G in graphs:
        # You can tweak walk_len per graph if needed
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
    return df
