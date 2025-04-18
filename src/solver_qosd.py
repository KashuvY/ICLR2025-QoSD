# src/solver_qosd.py
import os, json
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum

def solve_instance(lp_path, pairs, T, time_limit=3600, threads=1, log_path=None):
    model = gp.read(lp_path)
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit       = time_limit
    model.Params.Threads         = threads
    if log_path:
        model.Params.LogFile = log_path

    # extract x_{u}_{v} vars + build NX structure
    G = nx.DiGraph()
    xvars = {}
    for v in model.getVars():
        if v.VarName.startswith("x_"):
            _, us, vs = v.VarName.split("_")
            u, w = int(us), int(vs)
            xvars[(u,w)] = v
            G.add_edge(u, w)

    def _lazy(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            sol = {e: cb_model.cbGetSolution(var) for e,var in xvars.items()}
            # update weights & check each (s,t)
            for (s,t) in pairs:
                for u,v in xvars:
                    G[u][v]['weight'] = 1 + sol[(u,v)]
                try:
                    dist, path = nx.single_source_dijkstra(G, s, t)
                except nx.NetworkXNoPath:
                    continue
                if dist < T:
                    expr = LinExpr()
                    expr += (len(path)-1)
                    expr += quicksum(xvars[(u,v)] for u,v in zip(path, path[1:]))
                    cb_model.cbLazy(expr >= T)

    model.optimize(_lazy)
    return {
        "status":  model.Status,
        "obj":     model.ObjVal    if model.SolCount>0 else None,
        "time":    model.Runtime,
        "nodes":   model.NodeCount,
        "gap":     model.MIPGap
    }

def batch_solve(lp_dir, out_json_dir, out_log_dir, time_limit, threads):
    pairs_map  = json.load(open(os.path.join("data/QoSD","pairs.json")))
    thresh_map = json.load(open(os.path.join("data/QoSD","thresholds.json")))
    os.makedirs(out_json_dir, exist_ok=True)
    os.makedirs(out_log_dir, exist_ok=True)

    for lp_name, pairs in pairs_map.items():
        lp_path  = os.path.join(lp_dir, lp_name)
        log_path = os.path.join(out_log_dir, lp_name + ".log")
        res = solve_instance(lp_path, pairs, thresh_map[lp_name],
                             time_limit=time_limit, threads=threads,
                             log_path=log_path)
        outp = os.path.join(out_json_dir, lp_name + ".json")
        json.dump(res, open(outp,"w"), indent=2)
        print(f"[QoSD] {lp_name}: status={res['status']} obj={res['obj']}")
