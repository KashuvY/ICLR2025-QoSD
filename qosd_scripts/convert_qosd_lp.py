#!/usr/bin/env python3
"""Convert raw JSON QoSD instances → LP + tensors."""
import os, json, pathlib, torch, networkx as nx

def convert_dataset(raw_dir: str, out_dir: str):
    raw_dir = pathlib.Path(raw_dir)
    out_dir = pathlib.Path(out_dir)
    lp_dir   = out_dir / "lp"
    tens_dir = out_dir / "tensors"
    lp_dir.mkdir(parents=True, exist_ok=True)
    tens_dir.mkdir(parents=True, exist_ok=True)

    pairs_map, thresh_map = {}, {}

    for fjson in raw_dir.glob("instance_*.json"):
        inst   = json.load(open(fjson))
        gid    = inst["graph_id"]
        T      = inst["threshold"]
        edges  = inst["edges"]
        pairs  = [(p["src"], p["dest"]) for p in inst["pairs"]]

        # build graph
        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e["src"], e["dest"], weight=e["initial_weight"])

        elist  = [(e["src"], e["dest"]) for e in edges]
        eidx   = {uv: i for i, uv in enumerate(elist)}
        m      = len(elist)

        lp_name = f"instance_{gid}.lp"
        pairs_map [lp_name] = pairs
        thresh_map[lp_name] = T

        # ---- LP file --------------------------------------------------------
        lines = ["Minimize", " obj: " + " + ".join(f"x_{u}_{v}" for u, v in elist),
                 "\nSubject To"]
        for k, (s, t) in enumerate(pairs):
            _, path = nx.single_source_dijkstra(G, s, t)
            rhs = T - len(path) + 1
            lines.append(f" c_{gid}_{k}: " +
                         " + ".join(f"x_{path[i]}_{path[i+1]}" for i in range(len(path)-1)) +
                         f" >= {rhs}")
        lines.append("\nBounds")
        for u, v in elist:
            lines.append(f" 0 <= x_{u}_{v} <= 5")
        lines.append("\nGeneral")
        lines.extend([f" x_{u}_{v}" for u, v in elist])
        lines.append("\nEnd")
        with open(lp_dir / lp_name, "w") as f:
            f.write("\n".join(lines))

        # ---- tensors --------------------------------------------------------
        k = len(pairs)
        A = torch.zeros((k, m))
        b = torch.zeros((k,))
        c = torch.ones((m, 1))
        for row, (s, t) in enumerate(pairs):
            _, path = nx.single_source_dijkstra(G, s, t)
            b[row] = T - len(path) + 1
            for u, v in zip(path, path[1:]):
                A[row, eidx[(u, v)]] = 1
        torch.save((A, b, c), tens_dir / (lp_name.replace(".lp", ".pt")))

    json.dump(pairs_map,  open(out_dir / "pairs.json",      "w"), indent=2)
    json.dump(thresh_map, open(out_dir / "thresholds.json", "w"), indent=2)
    print(f"[convert] {len(pairs_map)} instances → LPs in {lp_dir}")
    return lp_dir, tens_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    convert_dataset(args.in_dir, args.out_dir)
