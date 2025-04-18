#!/usr/bin/env python3
"""
Turn the QoSD LP files that live in runs/<RUN>/lp/ into the
Graph‑tensors (<RUN>/preprocess/{samples,tensors}/ …) used by DiffILO.
"""

import argparse, os, pickle, multiprocessing as mp, logging
from functools import partial
import torch, numpy as np
import pyscipopt as scip

# ────────────────────────────────────────────────────────────────────────────────
def _process(lp_name: str, lp_dir: str, tens_dir: str,
             sample_dir: str, tensor_out_dir: str):
    """Preprocess a single LP → (sample.pkl, tensor.pkl)."""
    lp_path = os.path.join(lp_dir, lp_name)
    base    = os.path.splitext(lp_name)[0]          # instance_XX

    # ── fetch (A,b,c) that the converter already saved ────────────────────────
    A, b, c = torch.load(os.path.join(tens_dir, base + ".pt"))
    if b.ndim == 1:
        b = b.view(-1, 1)

    # ── mine bipartite features from the LP via SCIP ──────────────────────────
    m = scip.Model()
    m.hideOutput(True)
    m.readProblem(lp_path)

    # variables & mapping
    vars_ = m.getVars()
    v_map = {v.name: i for i, v in enumerate(vars_)}
    nvars = len(vars_)

    # prepare feat holders -----------------------------------------------------
    var_feats  = torch.zeros((nvars, 5), dtype=torch.float32)
    cons_feats, idx0, idx1, val = [], [], [], []

    # objective → var_feats[:,0] = coeff
    for e in m.getObjective():
        name = e.vartuple[0].name
        var_feats[v_map[name], 0] = float(m.getObjective()[e])

    # constraints loop ---------------------------------------------------------
    for r, cons in enumerate(m.getConss()):
        coeffs = m.getValsLinear(cons)
        if not coeffs:
            continue
        rhs = m.getRhs(cons)
        cons_feats.append([sum(coeffs.values())/len(coeffs), len(coeffs), rhs])

        for var, coef in coeffs.items():
            v_idx = v_map[var]
            idx0.append(r)
            idx1.append(v_idx)
            val.append(1.0)
            var_feats[v_idx, 1] += coef   # mean coef per var
            var_feats[v_idx, 2] += 1      # #constraints this var appears in

    if not cons_feats:            # a corner‑case we silently skip
        return
    cons_feats = torch.tensor(cons_feats)

    # normalise simple numeric columns (in‑place) ------------------------------
    for mat in [var_feats, cons_feats]:
        if mat.numel():
            mn, mx = mat.min(0).values, mat.max(0).values
            diff   = torch.where(mx==mn, torch.ones_like(mx), mx-mn)
            mat.sub_(mn).div_(diff)

    edge_idx  = torch.tensor([idx0, idx1], dtype=torch.long)
    edge_attr = torch.tensor(val,     dtype=torch.float32).view(-1, 1)

    # dump sample.pkl  (= graph for pyg) ---------------------------------------
    sample = [cons_feats, edge_idx, edge_attr, var_feats]
    with open(os.path.join(sample_dir, base + ".pkl"), "wb") as f:
        pickle.dump(sample, f)

    # dump tensor.pkl  (= A,b,c for loss) --------------------------------------
    with open(os.path.join(tensor_out_dir, base + ".pkl"), "wb") as f:
        pickle.dump((A, b, c), f)


# ────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run name (directory under runs/)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    run_dir   = os.path.join("runs", args.run)
    lp_dir    = os.path.join(run_dir, "lp")
    tens_dir  = os.path.join(run_dir, "tensors")     # created by converter
    prep_dir  = os.path.join(run_dir, "preprocess")
    sample_d  = os.path.join(prep_dir, "samples")
    tensor_d  = os.path.join(prep_dir, "tensors")
    os.makedirs(sample_d, exist_ok=True)
    os.makedirs(tensor_d, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.info("Pre‑processing LPs in %s …", lp_dir)

    fn = partial(_process, lp_dir=lp_dir, tens_dir=tens_dir,
                 sample_dir=sample_d, tensor_out_dir=tensor_d)
    files = [f for f in os.listdir(lp_dir) if f.endswith(".lp")]

    with mp.Pool(args.workers) as pool:
        for _ in pool.imap_unordered(fn, files):
            pass
    logging.info("All done (%d instances).", len(files))


if __name__ == "__main__":
    main()
