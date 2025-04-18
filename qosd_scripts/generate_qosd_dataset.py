#!/usr/bin/env python3
"""Generate synthetic QoSD instances with connected (s,t) pairs."""
import os, json, random, argparse, pathlib, networkx as nx

# ───────────────────────────────────────────────── generate ONE dataset
def generate_dataset(out_dir: str,
                     n_train: int, n_test: int,
                     n_nodes: int = 100,
                     generator: str = "erdos-renyi",
                     p_edge: float = 0.05,
                     n_pairs: int = 5,
                     threshold: int = 5,
                     max_budget: int = 5,
                     **kwargs):
    """
    Create *n_train+n_test* raw JSON instances and save them to *out_dir*.
    """
    n_total = n_train + n_test
    out_dir = pathlib.Path(out_dir)
    raw_dir = out_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def new_graph():
        if generator == "erdos-renyi":
            return nx.gnp_random_graph(n_nodes, p_edge, directed=True)
        m = kwargs.get("ba_m", 3)
        return nx.barabasi_albert_graph(n_nodes, m).to_directed()

    gid = 0
    while gid < n_total:
        G = new_graph()
        # annotate edges
        edges = []
        for u, v in G.edges():
            edges.append({
                "src": u, "dest": v,
                "initial_weight": 1,
                "budget": max_budget
            })
        # pick n_pairs connected pairs
        pairs = []
        while len(pairs) < n_pairs:
            s, t = random.sample(range(n_nodes), 2)
            try:
                nx.shortest_path(G, s, t)
                pairs.append((s, t))
            except nx.NetworkXNoPath:
                continue

        inst = {
            "graph_id": gid,
            "n_nodes": n_nodes,
            "edges": edges,
            "pairs": [{"src": s, "dest": t} for s, t in pairs],
            "threshold": threshold,
        }
        with open(raw_dir / f"instance_{gid}.json", "w") as f:
            json.dump(inst, f)
        gid += 1

    print(f"[generate] {n_total} raw JSON saved to {raw_dir}")
    return raw_dir


# ──────────────────────────────────────────────────── CLI entry point
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", required=True, type=int)
    p.add_argument("--n_test",  required=True, type=int)
    p.add_argument("--n_nodes", type=int, default=100)
    p.add_argument("--generator", default="erdos-renyi",
                   choices=["erdos-renyi", "barabasi-albert"])
    p.add_argument("--p_edge", type=float, default=0.05)
    p.add_argument("--n_pairs", type=int, default=5)
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--max_budget", type=int, default=5)
    args = p.parse_args()
    generate_dataset(args.out_dir, args.n_train, args.n_test,
                     n_nodes=args.n_nodes, generator=args.generator, p_edge=args.p_edge, n_pairs=args.n_pairs, threshold=args.threshold, max_budget=args.max_budget)

if __name__ == "__main__":
    _cli()
