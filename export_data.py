"""Compute everything the Demo 1 UI shows and write web/data.json.

Includes selection-corrected statistics (max over layers vs a label-permutation null), a random-word control, 3-D projections
per layer, cross-language alignment, and a dial sweep (ring vs chord vs scrambled-ring interpolation).
Usage: uv run python export_data.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from calendar_spike import CONCEPTS, TEMPLATES, angles, circ_score, plane
from dial_spike import DAYS, Runner, make_patcher, ring_model

MODEL = "sarvamai/sarvam-1"
CACHE = Path("cache/sarvam1.npz")
OUT = Path("web/data.json")
LAYER_DIAL = 12
NOUNS = ("table river window garden pencil bridge ladder mirror candle basket violin pocket anchor button hammer ribbon jacket "
         "kitchen engine planet mountain cookie bottle carpet saddle fabric pillow whistle compass blanket lantern paddle "
         "helmet cabinet teapot blanket feather marble meadow harbor trumpet curtain pebble satchel").split()


def layer_scores(X, n_layers):
    out = []
    for L in range(1, n_layers):
        mu, b = plane(X[:, L])
        out.append(float(circ_score(angles(X[:, L], mu, b))))
    return out


def max_layer_perm_p(X, n_layers, max_perm=4000, seed=0):
    """p-value of the best-layer ring score against label permutations that ALSO take the best layer (corrects for picking a layer)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perms = list(itertools.permutations(range(n))) if n <= 7 else [rng.permutation(n) for _ in range(max_perm)]
    th = [angles(X[:, L], *plane(X[:, L])) for L in range(1, n_layers)]
    obs = max(circ_score(t) for t in th)
    null = np.array([max(circ_score(t[np.array(p)]) for t in th) for p in perms])
    return float(obs), float((null >= obs - 1e-12).mean())


def projections(acts, n_layers, layers):
    """3-D coordinates of every concept in a basis fit on its English counterpart (scaled by the English ring radius).
    Consecutive layers are Procrustes-aligned so the UI can morph smoothly between them."""
    from scipy.linalg import orthogonal_procrustes
    out = {str(L): {} for L in layers}
    for ref, keys in (("en_days", [k for k in acts if k.endswith("days")]), ("en_months", [k for k in acts if k.endswith("months")])):
        prev = None
        for L in layers:
            mu = acts[ref][:, L].mean(0)
            _, _, vt = np.linalg.svd(acts[ref][:, L] - mu, full_matrices=False)
            B = vt[:3]
            Pref = (acts[ref][:, L] - mu) @ B.T
            R = np.eye(3)
            if prev is not None:
                R, _ = orthogonal_procrustes(Pref, prev)
            radius = np.sqrt((Pref ** 2).sum(1)).mean()
            for k in keys:  # each language set is centred and scaled on its own so the *shapes* can be overlaid
                Pk = ((acts[k][:, L] - acts[k][:, L].mean(0)) @ B.T) @ R
                out[str(L)][k] = (Pk / np.sqrt((Pk ** 2).sum(1)).mean()).round(3).tolist()
            prev = Pref @ R
        # global view rotation: ring plane -> xy-plane at layer 20, days counter-clockwise (same rotation for every layer / language)
        ref_pts = np.array(out["20"][ref])
        _, _, vt = np.linalg.svd(ref_pts - ref_pts.mean(0), full_matrices=False)
        V = vt.T.copy()
        z = np.cross((ref_pts @ V)[:, :2].tolist() and np.c_[(ref_pts @ V)[:, :2], np.zeros(len(ref_pts))], np.roll(np.c_[(ref_pts @ V)[:, :2], np.zeros(len(ref_pts))], -1, 0))[:, 2].mean()
        if z < 0:
            V[:, 1] *= -1
        if np.linalg.det(V) < 0:
            V[:, 2] *= -1
        for L in layers:
            for k in keys:
                out[str(L)][k] = (np.array(out[str(L)][k]) @ V).round(3).tolist()
    return out


def alignment(acts, n_layers):
    res = {}
    for en, other in [("en_days", "hi_days"), ("en_days", "ta_days"), ("en_months", "hi_months")]:
        Xe, Xo = acts[en], acts[other]
        n = Xe.shape[0]
        per_layer = []
        for L in range(1, n_layers):
            mu, b = plane(Xe[:, L])
            d = angles(Xo[:, L], mu, b) - angles(Xe[:, L], mu, b)
            per_layer.append(float(np.abs(np.mean(np.exp(1j * d)))))
        res[f"{en}->{other}"] = per_layer
    return res


def collect_random(r, n_sets=30, n_templates=4, seed=1):
    rng = np.random.default_rng(seed)
    scores = []
    tpl = TEMPLATES["en"][:n_templates]
    for s in range(n_sets):
        words = list(rng.choice(sorted(set(NOUNS)), size=7, replace=False))
        acts = []
        for w in words:
            hs = []
            for t in tpl:
                ids = r.tok(t.format(w), return_tensors="pt").to(r.device)
                with torch.no_grad():
                    h = r.model(**ids, output_hidden_states=True).hidden_states
                hs.append(torch.stack([x[0, -1].float().cpu() for x in h]))
            acts.append(torch.stack(hs).mean(0))
        X = torch.stack(acts).numpy()
        scores.append(max(layer_scores(X, X.shape[1])))
        print(f"random set {s + 1}/{n_sets}: best-layer ring score {scores[-1]:.2f}", flush=True)
    return scores


def dial_sweep(r, acts):
    M = acts["en_days"][:, LAYER_DIAL]
    mu, Q, P = ring_model(M, [1, 2, 3])
    perm = np.array([0, 3, 6, 2, 5, 1, 4])  # scrambled day order for the control (same ring machinery)
    _, Qs, Ps = ring_model(M[perm], [1, 2, 3])
    grid = np.arange(0, 7.01, 0.25)
    sweep = {"ring": {}, "chord": {}, "scrambled": {}}
    for i, d in enumerate(DAYS):
        rows = {"ring": [], "chord": [], "scrambled": []}
        for t in grid:
            t0 = i + t
            rows["ring"].append(r.probs(d, LAYER_DIAL, make_patcher(mu, Q, P, t0, r.device)).round(4).tolist())
            lo, hi = int(np.floor(t0)) % 7, int(np.ceil(t0)) % 7
            frac = t0 - np.floor(t0)
            chord = lambda tt, lo=lo, hi=hi, fr=frac: (1 - fr) * M[lo] + fr * M[hi]
            rows["chord"].append(r.probs(d, LAYER_DIAL, make_patcher(mu, Q, chord, 0, r.device)).round(4).tolist())
            rows["scrambled"].append(r.probs(d, LAYER_DIAL, make_patcher(mu, Qs, lambda tt, tt0=t0: Ps(tt0), 0, r.device)).round(4).tolist())
        for k in rows:
            sweep[k][d] = rows[k]
        print(f"dial sweep {d} done", flush=True)

    def mass_on_neighbours(rows, half_only):
        vals = []
        for i in range(7):
            for gi, t in enumerate(grid):
                if t % 1 == 0.5 or (not half_only and t % 1 == 0):
                    t0 = i + t
                    exp = {(int(np.floor(t0)) + 1) % 7, (int(np.ceil(t0)) + 1) % 7}
                    vals.append(sum(rows[DAYS[i]][gi][j] for j in exp))
        return float(np.mean(vals))

    metrics = {k: {"neighbour_mass_half": mass_on_neighbours(sweep[k], True), "neighbour_mass_all": mass_on_neighbours(sweep[k], False)} for k in sweep}
    return {"layer": LAYER_DIAL, "harmonics": [1, 2, 3], "t_grid": grid.tolist(), "sweep": sweep, "metrics": metrics}


def main():
    import sys
    if "--proj-only" in sys.argv:
        acts_np = np.load(CACHE)
        acts = {k: acts_np[k] for k in acts_np.files}
        data = json.loads(OUT.read_text())
        data["proj"] = projections(acts, acts["en_days"].shape[1], list(range(2, 29, 2)))
        OUT.write_text(json.dumps(data, ensure_ascii=False))
        print("projections rewritten")
        return
    acts_np = np.load(CACHE)
    acts = {k: acts_np[k] for k in acts_np.files}
    n_layers = acts["en_days"].shape[1]
    data = {"model": MODEL, "n_layers": n_layers - 1, "names": CONCEPTS}

    data["ring"] = {k: layer_scores(v, n_layers) for k, v in acts.items()}
    data["ring_stats"] = {}
    for k, v in acts.items():
        obs, p = max_layer_perm_p(v, n_layers)
        data["ring_stats"][k] = {"best_R": obs, "p_layer_corrected": p}
        print(k, f"best R={obs:.2f}  layer-corrected p={p:.4f}", flush=True)
    data["align"] = alignment(acts, n_layers)
    data["proj"] = projections(acts, n_layers, list(range(2, 29, 2)))
    data["measured_dial_spike"] = {  # copied from dial_spike.py output (layer -> harmonics -> accuracy), random subspace was 0.00-0.02
        "6": {"1": 0.17, "2": 0.40, "3": 0.71}, "8": {"1": 0.31, "2": 0.52, "3": 0.74},
        "10": {"1": 0.29, "2": 0.62, "3": 0.90}, "12": {"1": 0.26, "2": 0.60, "3": 0.90}, "14": {"1": 0.29, "2": 0.55, "3": 0.90}}

    r = Runner(MODEL)
    data["random_words_best_R"] = collect_random(r)
    data["dial"] = dial_sweep(r, acts)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    print("dial metrics:", json.dumps(data["dial"]["metrics"], indent=1))


if __name__ == "__main__":
    main()
