import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 读取并清洗数据
# =========================

def load_lollipop_summary(path: str) -> pd.DataFrame:
    """
    从 cover_time_results_lollipop_summary.csv 读取并清洗数据，返回 tidy DataFrame：
    列包括：
      - method_raw: 原始方法名
      - method:     清洗后的方法名（更好看）
      - seconds
      - vertex_cover_time
      - edge_cover_time
      - n, m        lollipop_(n, m)
    """
    raw_df = pd.read_csv(path)

    # 从第一列列名解析第一个图的 n, m，比如 "lollipop_(n=6,m=8)"
    first_col_name = raw_df.columns[0]
    m0 = re.match(r"lollipop_\(n=(\d+),m=(\d+)\)", first_col_name)
    if m0:
        current_n = int(m0.group(1))
        current_m = int(m0.group(2))
    else:
        current_n = None
        current_m = None

    records = []
    expect_header_next = False

    for _, row in raw_df.iterrows():
        first = str(row.iloc[0])

        # 检测块头，例如 'lollipop_(n=9,m=18)'
        m = re.match(r"lollipop_\(n=(\d+),m=(\d+)\)", first)
        if m:
            current_n = int(m.group(1))
            current_m = int(m.group(2))
            # 下一行是 graph,lollipop_xxx,lollipop_xxx,edge_cover_time 的 header，跳过
            expect_header_next = True
            continue

        if expect_header_next:
            # 跳过 header 行
            expect_header_next = False
            continue

        method = first
        # 跳过 "graph" 那一行
        if method == "graph":
            continue

        seconds = float(row["seconds"])
        v = float(row["vertex_cover_time"])
        e = float(row["edge_cover_time"])

        records.append(
            {
                "method_raw": method,
                "seconds": seconds,
                "vertex_cover_time": v,
                "edge_cover_time": e,
                "n": current_n,
                "m": current_m,
            }
        )

    df = pd.DataFrame(records)

    # 方法名清洗
    name_map = {
        "recurrent_random_walk": "Recurrent RW",
        "unbiased_": "Unbiased",
        "unbiased_no_backtracking_": "Unbiased (no-backtrack)",
        "MDLR_": "MDLR",
        "N2V": "Node2Vec",
    }
    df["method"] = df["method_raw"].map(name_map).fillna(df["method_raw"])

    # 排序
    df = df.sort_values(["n", "method"]).reset_index(drop=True)
    return df


# =========================
# 2. 画图函数
# =========================

def plot_vertex_cover_vs_n(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    for method, sub in df.groupby("method"):
        sub_sorted = sub.sort_values("n")
        plt.plot(
            sub_sorted["n"],
            sub_sorted["vertex_cover_time"],
            marker="o",
            label=method,
        )
    plt.xlabel("n (lollipop size)")
    plt.ylabel("Vertex cover time")
    plt.title("Vertex cover time vs n on lollipop graphs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lollipop_vertex_cover_vs_n.pdf")


import matplotlib.pyplot as plt
import numpy as np

def plot_edge_cover_vs_n(df, fname: str):
    # --- reasonable defaults for paper-quality plots ---
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,   # TrueType, avoids Type 3 fonts in PDF
        "ps.fonttype": 42,
    })

    # NeurIPS single-column-ish size
    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    methods = sorted(df["method"].unique())
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].sort_values("n")
        ax.plot(
            sub["n"],
            sub["edge_cover_time"],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            linestyle=linestyles[i % len(linestyles)],
            label=method,
        )

    ax.set_xlabel(r"$n$ (lollipop graph size)")
    ax.set_ylabel("Edge cover time")

    # optional: log-scale if ranges are very different
    # ax.set_yscale("log")

    # light grid for readability
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

    # legend slightly outside to avoid occluding curves
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        borderpad=0.3,
        handlelength=1.8,
        handletextpad=0.4,
    )

    fig.tight_layout()
    fig.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {fname}")


def plot_runtime_vs_n(df: pd.DataFrame, logy: bool = True):
    plt.figure(figsize=(6, 4))
    for method, sub in df.groupby("method"):
        sub_sorted = sub.sort_values("n")
        plt.plot(
            sub_sorted["n"],
            sub_sorted["seconds"],
            marker="o",
            label=method,
        )
    plt.xlabel("n (lollipop size)")
    plt.ylabel("Seconds")
    if logy:
        plt.yscale("log")
        plt.title("Runtime vs n on lollipop graphs (log scale)")
    else:
        plt.title("Runtime vs n on lollipop graphs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tree_runtime_vs_n.pdf")


# =========================
# 3. 主入口：加载 + 打印 + 画图
# =========================

if __name__ == "__main__":
    path = "/home/aifb/cc7738/hierarchial_random_walk/src_analysis/plot/cover_time_results_lollipop_summary.csv"
    #  path = "/home/aifb/cc7738/hierarchial_random_walk/src_analysis/plot/cover_time_results_tree_summary.csv"
    df = load_lollipop_summary(path)

    # print
    print("Cleaned DataFrame:\n")
    print(df.to_string(index=False))

    # plot
    # plot_vertex_cover_vs_n(df)
    plot_edge_cover_vs_n(df, "lollipop_edge_cover_vs_n.pdf")
    # plot_runtime_vs_n(df, logy=True)
