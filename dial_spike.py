"""Demo 1: a smooth "clock dial" from the harmonic ring of weekday activations.

Day i's class-mean activation m_i is decomposed over the 7-cycle into harmonics h=1..3 (discrete Fourier transform
across the day index). A continuous position t on the ring is then
    P(t) = mu + (2/7) * sum_h Re( a_h * exp(2*pi*i*h*t/7) ),     a_h = sum_i (m_i - mu) * exp(-2*pi*i*h*i/7)
which reproduces class mean i exactly at integer t. Patching: replace the component of the day-token residual that lies in the
ring subspace S with P(t), leave the rest alone.

Usage: uv run python dial_spike.py [model_id] [cache_name]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calendar_spike import CONCEPTS

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sarvamai/sarvam-1"
CACHE = Path("cache") / f"{sys.argv[2] if len(sys.argv) > 2 else 'sarvam1'}.npz"
DAYS = CONCEPTS["en_days"]
SHOTS = "If today is Tuesday, then tomorrow is Wednesday.\nIf today is Saturday, then tomorrow is Sunday.\nIf today is Thursday, then tomorrow is Friday.\n"
N = 7


def ring_model(M, harmonics):
    """M: [7, d] class means at one layer. Returns mu, Q (d x 2H orthonormal), and fn t -> P(t)."""
    mu = M.mean(0)
    idx = np.arange(N)
    a = {h: ((M - mu) * np.exp(-2j * np.pi * h * idx / N)[:, None]).sum(0) for h in harmonics}
    cols = []
    for h in harmonics:
        cols += [a[h].real, a[h].imag]
    Q, _ = np.linalg.qr(np.stack(cols, 1))

    def P(t):
        return mu + (2 / N) * sum((a[h] * np.exp(2j * np.pi * h * t / N)).real for h in harmonics)

    return mu, Q, P


def make_patcher(mu, Q, P, t, device):
    mu_t, Q_t = (torch.tensor(a, dtype=torch.float32, device=device) for a in (mu, Q))
    target = torch.tensor(P(t), dtype=torch.float32, device=device)

    def f(h):
        return h - Q_t @ (Q_t.T @ (h - mu_t)) + Q_t @ (Q_t.T @ (target - mu_t))

    return f


class Runner:
    def __init__(self, model_id):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device).eval()
        self.cand = [self.tok(" " + d, add_special_tokens=False).input_ids[0] for d in DAYS]
        assert len(set(self.cand)) == 7

    def probs(self, day, layer=None, fn=None):
        head = f"{SHOTS}If today is {day}"
        ids = self.tok(f"{head}, then tomorrow is", return_tensors="pt").to(self.device)
        idx = len(self.tok(head, return_tensors="pt").input_ids[0]) - 1
        hd = None
        if fn is not None:
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[0, idx] = fn(h[0, idx].float()).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hd = self.model.model.layers[layer - 1].register_forward_hook(hook)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1].float()
        if hd:
            hd.remove()
        return logits[self.cand].softmax(0).cpu().numpy()


def main():
    r = Runner(MODEL)
    acts = np.load(CACHE)["en_days"]
    rng = np.random.default_rng(0)
    print(f"{'layer':>5} {'harm':>6} {'dial acc':>9} {'rand-subspace':>14}   (chance 0.14; k=1..6 shifts, 7 source days)")
    for L in (6, 8, 10, 12, 14):
        for harm in ([1], [1, 2], [1, 2, 3]):
            mu, Q, P = ring_model(acts[:, L], harm)
            rQ, _ = np.linalg.qr(rng.normal(size=Q.shape))
            hit, rhit = [], []
            for i, d in enumerate(DAYS):
                for k in range(1, 7):
                    t = i + k
                    p = r.probs(d, L, make_patcher(mu, Q, P, t, r.device))
                    hit.append(int(p.argmax() == (i + k + 1) % 7))
                    p = r.probs(d, L, make_patcher(mu, rQ, P, t, r.device))
                    rhit.append(int(p.argmax() == (i + k + 1) % 7))
            print(f"{L:>5} {str(harm):>6} {np.mean(hit):>9.2f} {np.mean(rhit):>14.2f}", flush=True)


if __name__ == "__main__":
    main()
