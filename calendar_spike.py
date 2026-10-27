"""Demo 1 spike: do day/month names form a circle inside the model, and is it shared across languages?

Usage: uv run python calendar_spike.py [model_id] [cache_name]
Writes cache/<cache_name>.npz (mean activations per concept/layer) and prints a summary table.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sarvamai/sarvam-1"
CACHE = Path("cache") / f"{sys.argv[2] if len(sys.argv) > 2 else 'sarvam1'}.npz"

# name sets, in true cyclic order (Monday first / January first)
CONCEPTS = {
    "en_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "hi_days": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
    "ta_days": ["திங்கட்கிழமை", "செவ்வாய்க்கிழமை", "புதன்கிழமை", "வியாழக்கிழமை", "வெள்ளிக்கிழமை", "சனிக்கிழமை", "ஞாயிற்றுக்கிழமை"],
    "en_months": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "hi_months": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
}
# the name is always the last thing in the string, so the last token is the name's last token
TEMPLATES = {
    "en": ["Today is {}", "See you on {}", "The meeting is on {}", "It happened on {}", "My birthday is in {}", "I was born in {}", "Let us meet on {}", "What I remember is {}"],
    "hi": ["आज {}", "हम {} को मिलेंगे", "बैठक {} को है", "यह {} को हुआ", "मेरा जन्मदिन {}", "मैं {} में पैदा हुआ", "चलो {} को मिलते हैं", "मुझे याद है {}"],
    "ta": ["இன்று {}", "{} சந்திப்போம்", "கூட்டம் {}", "இது {} நடந்தது", "என் பிறந்தநாள் {}", "நான் {} பிறந்தேன்", "{} சந்திக்கலாம்", "எனக்கு நினைவிருக்கிறது {}"],
}


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def collect(model, tok, device):
    out = {}
    for key, names in CONCEPTS.items():
        lang = key.split("_")[0]
        acts = []  # [n_names, n_layers+1, d]
        for name in names:
            per_t = []
            for t in TEMPLATES[lang]:
                ids = tok(t.format(name), return_tensors="pt").to(device)
                hs = model(**ids, output_hidden_states=True).hidden_states
                per_t.append(torch.stack([h[0, -1].float().cpu() for h in hs]))  # [L+1, d]
            acts.append(torch.stack(per_t).mean(0))
        out[key] = torch.stack(acts).numpy()
        print(f"collected {key}: {out[key].shape}", flush=True)
    return out


def plane(X):
    """top-2 PCA basis of class means X [n, d]."""
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return X.mean(0), vt[:2]


def circ_score(theta):
    """How well angles follow true cyclic order, up to rotation and reflection. 1 = perfect ring."""
    n = len(theta)
    true = 2 * np.pi * np.arange(n) / n
    best = 0.0
    for s in (1, -1):
        best = max(best, np.abs(np.mean(np.exp(1j * (theta - s * true)))))
    return best


def angles(X, mu, basis):
    p = (X - mu) @ basis.T
    return np.arctan2(p[:, 1], p[:, 0])


def perm_p(score_fn, n, observed, max_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    perms = list(itertools.permutations(range(n))) if n <= 7 else [rng.permutation(n) for _ in range(max_perm)]
    null = np.array([score_fn(np.array(p)) for p in perms])
    return float((null >= observed - 1e-12).mean())


def main():
    device = pick_device()
    print(f"model={MODEL} device={device}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    CACHE.parent.mkdir(exist_ok=True)
    acts = collect(model, tok, device)
    np.savez(CACHE, **acts)

    n_layers = next(iter(acts.values())).shape[1]
    print("\nper-layer ring score (1=perfect circle) and permutation p-value (lower p = more circular than shuffled labels)")
    for key, X in acts.items():
        n = X.shape[0]
        rows = []
        for L in range(1, n_layers):
            mu, b = plane(X[:, L])
            th = angles(X[:, L], mu, b)
            sc = circ_score(th)
            p = perm_p(lambda perm: circ_score(th[perm]), n, sc)
            rows.append((L, sc, p))
        top = sorted(rows, key=lambda r: -r[1])[:3]
        print(f"{key:10s} best layers: " + "  ".join(f"L{L} R={s:.2f} p={p:.4f}" for L, s, p in top))

    print("\ncross-language alignment (plane fit on English; projected non-English class means). R = consistency of angle shift; p vs shuffled labels")
    for en, other in [("en_days", "hi_days"), ("en_days", "ta_days"), ("en_months", "hi_months")]:
        Xe, Xo = acts[en], acts[other]
        n = Xe.shape[0]
        rows = []
        for L in range(1, n_layers):
            mu, b = plane(Xe[:, L])
            te, to = angles(Xe[:, L], mu, b), angles(Xo[:, L], mu, b)
            def sc(perm):
                d = to[perm] - te
                return max(np.abs(np.mean(np.exp(1j * d))), 0)
            obs = sc(np.arange(n))
            rows.append((L, obs, perm_p(sc, n, obs)))
        top = sorted(rows, key=lambda r: -r[1])[:3]
        print(f"{en}->{other:9s} " + "  ".join(f"L{L} R={s:.2f} p={p:.4f}" for L, s, p in top))


if __name__ == "__main__":
    main()
