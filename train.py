#!/usr/bin/env python3
"""
Train the DiffILO GNN on a QoSD run.
* Generates dataset if missing           (generate_qosd_dataset.py)
* Converts → LP + tensors                (convert_qosd_lp.py)
* Preprocesses to graph objects          (preprocess.py --run …)
"""

import argparse, os, subprocess, random, json, logging, torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.model import GraphDataset, GNNPredictor
from src.utils  import set_seed, gumbel_sample
from torch_geometric.utils import unbatch
from tqdm import tqdm
import torch.optim as optim
import yaml
from types import SimpleNamespace

# … imports …
from qosd_scripts import generate_dataset, convert_dataset    # NEW

# ────────────────────────────────────────────────── helpers
def ensure_raw_dataset(run, n_train, n_test, n_nodes, generator, p_edge, n_pairs, threshold, max_budget):
    raw_dir = f"runs/{run}/raw/data/raw"
    if not (os.path.isdir(raw_dir) and os.listdir(raw_dir)):
        generate_dataset(f"runs/{run}/raw", n_train, n_test,
                         n_nodes=n_nodes, generator=generator, p_edge=p_edge, n_pairs=n_pairs, threshold=threshold, max_budget=max_budget)

def ensure_converted(run):
    lp_dir = f"runs/{run}/lp"
    if not (os.path.isdir(lp_dir) and os.listdir(lp_dir)):
        convert_dataset(f"runs/{run}/raw/data/raw", f"runs/{run}")

def ensure_preprocessed(run, workers):
    if not os.path.isdir(f"runs/{run}/preprocess/samples"):
        subprocess.run(["python", "preprocess.py",
                        "--run", run, f"--workers={workers}"], check=True)


# ────────────────────────────────────────────────────────────────────────────────
def make_split(run, n_train):
    split_file = f"runs/{run}/split.json"
    if os.path.isfile(split_file):
        return json.load(open(split_file))
    files = sorted(os.listdir(f"runs/{run}/preprocess/samples"))
    random.shuffle(files)
    split = {
        "train": files[:n_train],
        "test" : files[n_train:]
    }
    json.dump(split, open(split_file,"w"), indent=2)
    return split

# ────────────────────────────────────────────────────────────────────────────────
def train_loop(run, split, device, epochs, batch_size):
    sampler = lambda part: [
        os.path.join("runs", run, "preprocess", "samples", f) for f in split[part]
    ]
    train_ds, valid_ds = GraphDataset(sampler("train")), GraphDataset(sampler("test"))
    with open("config/model/QoSD.yaml") as f:
        yaml_cfg = yaml.safe_load(f)
    model_cfg = SimpleNamespace(**yaml_cfg)

    model = GNNPredictor(model_cfg).to(device)

    loader = lambda ds, sh: DataLoader(
        ds, batch_size=batch_size, shuffle=sh, follow_batch=["constraint_features","variable_features"]
    )
    opt = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(loader(train_ds, True), desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            y,_ = model(batch)
            y = y.view(-1,1)
            logits = unbatch(y, batch=batch.variable_features_batch)
            loss = 0
            for g,l in zip(batch.to_data_list(), logits):
                x = gumbel_sample(l, 8, 1.0).float().view(8,-1)
                A,b,c = [t.to(device) for t in (g.A,g.b,g.c)]
                obj = (torch.sigmoid(l)*c).sum()
                cons = torch.relu(A @ x.T - b).sum()
                loss += obj + 10*cons
            loss /= len(logits)
            loss.backward(); opt.step(); opt.zero_grad()

    out = f"runs/{run}/model.pth"
    torch.save(model.state_dict(), out)
    logging.info("Model saved to %s", out)

# ────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--n_test",  type=int, default=20)
    p.add_argument("--n_nodes", type=int, default=100)
    p.add_argument("--generator", default="erdos-renyi")
    p.add_argument("--p_edge", type=float, default=0.05)
    p.add_argument("--n_pairs", type=int, default=5)
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--max_budget", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cuda",    type=int, default=0)
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="If set, load graph.txt + pairs.txt from this folder (skips generate_dataset)"
    )
    args = p.parse_args()

    set_seed(0)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"runs/{args.run}", exist_ok=True)

    if args.data_dir:
        # ————————————————
        # write your graph.txt + pairs.txt → raw JSON
        raw_dir = f"runs/{args.run}/raw/data/raw"
        os.makedirs(raw_dir, exist_ok=True)

        # load edges
        df = pd.read_csv(os.path.join(args.data_dir, "graph.txt"))
        df = df[["u", "v", "weight", "max_budget"]]
        edges = []
        for _, row in df.iterrows():
            edges.append({
                "src": int(row.u),
                "dest": int(row.v),
                "initial_weight": float(row.weight),
                "budget": int(row.max_budget)
            })

        # load pairs
        pairs = []
        with open(os.path.join(args.data_dir, "pairs.txt")) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                s, t = map(int, line.split())
                pairs.append({"src": s, "dest": t})

        inst = {
            "graph_id": 0,
            "n_nodes": int(df[["u","v"]].max().max()) + 1,
            "edges": edges,
            "pairs": pairs,
            "threshold": args.threshold
        }
        with open(os.path.join(raw_dir, "instance_0.json"), "w") as f:
            json.dump(inst, f)
    else:
        ensure_raw_dataset(
            args.run,
            args.n_train, args.n_test,
            args.n_nodes, args.generator, args.p_edge,
            args.n_pairs, args.threshold, args.max_budget
        )

    ensure_converted   (args.run)
    ensure_preprocessed(args.run,args.workers)
    split = make_split (args.run, args.n_train)

    train_loop(args.run, split, device, args.epochs, args.batch_size)

if __name__ == "__main__":
    from src.utils import set_seed
    main()
