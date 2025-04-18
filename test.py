#!/usr/bin/env python3
"""
Solve every LP in runs/<RUN>/lp/ with:
 • GNN warm‑start  • QoSD lazy‑constraint callback
Results → runs/<RUN>/results/<lp>.json
"""

import argparse, os, json, pickle, networkx as nx, torch
import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum
from src.model import GNNPredictor, BipartiteNodeData
from tqdm import tqdm
import yaml
from types import SimpleNamespace

# ────────────────────────────────────────────────────────────────────────────────
def warm_start(lp_name, run, model, device):
    pkl = os.path.join("runs", run, "preprocess", "samples",
                       lp_name.replace(".lp",".pkl"))
    with open(pkl,"rb") as f:
        cons, idx, attr, var = pickle.load(f)
    data = BipartiteNodeData(
        torch.FloatTensor(cons).to(device),
        torch.LongTensor(idx).to(device),
        torch.FloatTensor(attr).to(device),
        torch.FloatTensor(var).to(device)
    )
    with torch.no_grad():
        logits,_ = model(data)
    return (torch.sigmoid(logits)>0.5).cpu().numpy().astype(float)

# ────────────────────────────────────────────────────────────────────────────────
def solve_one(lp_path, pairs, T, x0):
    m = gp.read(lp_path)
    m.Params.LazyConstraints = 1
    for i,v in enumerate(m.getVars()):
        v.Start = float(x0[i])

    # build NX skeleton
    G, xvars = nx.DiGraph(), {}
    for v in m.getVars():
        if v.VarName.startswith("x_"):
            _, us, vs = v.VarName.split("_")
            u,w = int(us), int(vs)
            xvars[(u,w)]=v
            G.add_edge(u,w)

    def cb(cb_m, where):
        if where != GRB.Callback.MIPSOL:
            return
        sol = {e: cb_m.cbGetSolution(v) for e,v in xvars.items()}
        for (u,v) in xvars:
            G[u][v]["weight"] = 1+sol[(u,v)]
        for s,t in pairs:
            try: dist,path = nx.single_source_dijkstra(G,s,t)
            except nx.NetworkXNoPath: continue
            if dist < T:
                expr = LinExpr(len(path)-1)
                expr += quicksum(xvars[(u,v)] for u,v in zip(path, path[1:]))
                cb_m.cbLazy(expr >= T)

    m.optimize(cb)
    return {
        "status": m.Status,
        "obj": m.ObjVal if m.SolCount>0 else None,
        "time": m.Runtime,
        "gap": m.MIPGap
    }

# ────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--time_limit", type=int, default=600)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    with open("config/model/QoSD.yaml") as f:
        yaml_cfg = yaml.safe_load(f)
    model_cfg = SimpleNamespace(**yaml_cfg)
    model = GNNPredictor(model_cfg).to(device)
    model.load_state_dict(torch.load(f"runs/{args.run}/model.pth", map_location=device))
    model.eval()

    pairs = json.load(open(f"runs/{args.run}/pairs.json"))
    thresh= json.load(open(f"runs/{args.run}/thresholds.json"))

    os.makedirs(f"runs/{args.run}/results", exist_ok=True)
    lp_dir = f"runs/{args.run}/lp"

    for lp_name in tqdm(sorted(os.listdir(lp_dir)), desc="Solve"):
        lp_path = os.path.join(lp_dir, lp_name)
        x0 = warm_start(lp_name, args.run, model, device)
        res = solve_one(lp_path, pairs[lp_name], thresh[lp_name], x0)
        json.dump(res, open(f"runs/{args.run}/results/{lp_name}.json","w"), indent=2)

if __name__ == "__main__":
    main()
