"""Demo 1 intervention spike: rotate a day-name activation around the weekday circle and see if the model's answer moves.

Usage: uv run python rotate_spike.py [model_id] [cache_name]
For each (source day, offset n, rotation k): patch the residual stream at the day-name token (layer L) by rotating its
2-D circle coordinates by k/7 of a turn, then check whether the model answers (source + k + n).
Controls: no rotation (k=0 baseline accuracy), and rotation inside a random plane of the same size.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calendar_spike import CONCEPTS, plane, angles

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sarvamai/sarvam-1"
CACHE = Path("cache") / f"{sys.argv[2] if len(sys.argv) > 2 else 'sarvam1'}.npz"
DAYS = CONCEPTS["en_days"]
LAYERS = [8, 10, 12, 14, 16, 18, 20]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    acts = np.load(CACHE)["en_days"]  # [7, L+1, d]
    cand = [tok(" " + d, add_special_tokens=False).input_ids[0] for d in DAYS]
    assert len(set(cand)) == 7, f"day first-tokens collide: {cand}"

    def run(prompt, name_idx, layer, fn):
        ids = tok(prompt, return_tensors="pt").to(device)
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[0, name_idx] = fn(h[0, name_idx].float()).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hd = model.model.layers[layer - 1].register_forward_hook(hook)  # hidden_states[layer] = output of layer-1 module
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float()
        hd.remove()
        return logits[cand].softmax(0).cpu().numpy()

    rng = np.random.default_rng(0)
    print(f"{'layer':>5} {'k=0 acc':>8} {'rot acc':>8} {'rand-plane acc':>14} {'plane-sub':>10}  (chance 0.14)")
    for L in LAYERS:
        mu, basis = plane(acts[:, L])
        th = angles(acts[:, L], mu, basis)
        # orientation: does angle increase with day index?
        sgn = 1 if np.mean(np.sin(th[1:] - th[:-1])) > 0 else -1
        # random control plane, orthonormal, same dimensionality
        q, _ = np.linalg.qr(rng.normal(size=(acts.shape[-1], 2)))
        rbasis = q.T
        mu_t, basis_t, rbasis_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in (mu, basis, rbasis)]

        def rot_fn(angle, B):
            c, s = np.cos(angle), np.sin(angle)
            R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=device)
            def f(h):
                x = h - mu_t
                coords = B @ x
                return mu_t + x - B.T @ coords + B.T @ (R @ coords)
            return f

        hits = {"base": [], "rot": [], "rand": [], "sub": []}
        coords_cls = (torch.tensor(acts[:, L], dtype=torch.float32, device=device) - mu_t) @ basis_t.T  # [7,2]
        for i, d in enumerate(DAYS):
            for n in (1,):
                shots = "If today is Tuesday, then tomorrow is Wednesday.\nIf today is Saturday, then tomorrow is Sunday.\nIf today is Thursday, then tomorrow is Friday.\n"
                head = f"{shots}If today is {d}"
                prompt = f"{head}, then tomorrow is"
                name_idx = len(tok(head, return_tensors="pt").input_ids[0]) - 1
                p0 = run(prompt, name_idx, L, lambda h: h)
                hits["base"].append(int(p0.argmax() == (i + n) % 7))
                for k in (1, 2, 3, 4, 5, 6):
                    ang = sgn * 2 * np.pi * k / 7
                    p = run(prompt, name_idx, L, rot_fn(ang, basis_t))
                    hits["rot"].append(int(p.argmax() == (i + k + n) % 7))
                    tgt = coords_cls[(i + k) % 7]
                    sub = lambda h, tgt=tgt: mu_t + (h - mu_t) - basis_t.T @ (basis_t @ (h - mu_t)) + basis_t.T @ tgt
                    ps = run(prompt, name_idx, L, sub)
                    hits["sub"].append(int(ps.argmax() == (i + k + n) % 7))
                    pr = run(prompt, name_idx, L, rot_fn(ang, rbasis_t))
                    hits["rand"].append(int(pr.argmax() == (i + k + n) % 7))
        print(f"{L:>5} {np.mean(hits['base']):>8.2f} {np.mean(hits['rot']):>8.2f} {np.mean(hits['rand']):>14.2f} {np.mean(hits['sub']):>10.2f}", flush=True)


if __name__ == "__main__":
    main()
