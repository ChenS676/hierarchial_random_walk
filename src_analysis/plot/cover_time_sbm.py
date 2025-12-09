import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Load + Clean SBM Summary
# =========================

def load_sbm_summary(path: str) -> pd.DataFrame:
    """
    Parse cover_time_results_sbm_summary.csv into tidy format.
    Expected block header format:
        sbm_(n=80,m=312)
    Followed by method rows.
    """
    raw = pd.read_csv(path)
    records = []
    expect_header = False
    current_n = None
    current_m = None

    for _, row in raw.iterrows():
        x = str(row.iloc[0])

        # block header
        m = re.match(r"sbm_\(n=(\d+),m=(\d+)\)", x)
        if m:
            current_n = int(m.group(1))
            current_m = int(m.group(2))
            expect_header = True
            continue

        if expect_header:
            expect_header = False
            continue

        if x == "graph":
            continue

        seconds = float(row["seconds"])
        v = float(row["vertex_cover_time"])
        e = float(row["edge_cover_time"])

        records.append(
            {
                "method_raw": x,
                "seconds": seconds,
                "vertex_cover_time": v,
                "edge_cover_time": e,
                "n": current_n,
                "m": current_m,
            }
        )

    df = pd.DataFrame(records)

    # rename nicer labels
    name_map = {
        "recurrent_random_walk": "Recurrent RW",
        "unbiased_": "Unbiased",
        "unbiased_no_backtracking_": "Unbiased (no-backtrack)",
        "MDLR_": "MDLR",
        "N2V": "Node2Vec"
    }
    df["method"] = df["method_raw"].map(name_map).fillna(df["method_raw"])
    df = df.sort_values(["n", "method"]).reset_index(drop=True)
    return df


# =========================
# 2. NeurIPS-Style Plotting
# =========================

def plot_edge_cover_vs_n_sbm(df: pd.DataFrame, out="sbm_edge_cover_vs_n.pdf"):
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.3, 2.5))  # NeurIPS single-col width

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for i, (method, sub) in enumerate(df.groupby("method")):
        sub = sub.sort_values("n")
        ax.plot(
            sub["n"],
            sub["edge_cover_time"],
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            linestyle=linestyles[i % len(linestyles)],
            label=method
        )

    ax.set_xlabel(r"$n$ (SBM size)")
    ax.set_ylabel("Edge cover time")
    ax.set_title("Edge Cover Time vs n on SBM graphs")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        borderpad=0.3,
        handlelength=2.0,
        handletextpad=0.4,
    )

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# =========================
# 3. Main Entry
# =========================

if __name__ == "__main__":
    path = "cover_time_results_sbm_summary.csv"  # adjust if needed
    df = load_sbm_summary(path)

    print("Cleaned SBM Data:\n")
    print(df.to_string(index=False))

    plot_edge_cover_vs_n_sbm(df)
