"""Serve the demo: static files from web/ plus a live /api/dial endpoint that runs Sarvam-1 on this machine.

    uv run python serve.py            # http://localhost:8000
    uv run python serve.py --no-live  # static replay only (no model download or load)
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

LIVE = "--no-live" not in sys.argv
R = {}


@asynccontextmanager
async def lifespan(app):
    if LIVE and not Path("cache/sarvam1.npz").exists():
        print("cache/sarvam1.npz not found: run `uv run python calendar_spike.py` first. Serving replay mode only.")
    elif LIVE:
        from dial_spike import Runner, make_patcher, ring_model
        R["rt"] = Runner("sarvamai/sarvam-1")
        R["acts"] = np.load("cache/sarvam1.npz")["en_days"][:, 12]
        R["fn"] = (make_patcher, ring_model)
        R["models"] = {"ring": ring_model(R["acts"], [1, 2, 3])}
        perm = np.array([0, 3, 6, 2, 5, 1, 4])
        R["models"]["scrambled"] = ring_model(R["acts"][perm], [1, 2, 3])
        print("live model ready")
    yield


app = FastAPI(lifespan=lifespan)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@app.get("/api/dial")
def dial(day: int, t: float, mode: str = "ring"):
    if not LIVE or "rt" not in R:
        raise HTTPException(503, "live mode off")
    if not (0 <= day < 7 and 0 <= t <= 7 and mode in ("ring", "chord", "scrambled")):
        raise HTTPException(400, "bad args")
    make_patcher, _ = R["fn"]
    rt, M = R["rt"], R["acts"]
    t0 = day + t
    if mode == "chord":
        lo, hi, fr = int(np.floor(t0)) % 7, int(np.ceil(t0)) % 7, t0 - np.floor(t0)
        _, Q, _ = R["models"]["ring"]
        mu = M.mean(0)
        fn = make_patcher(mu, Q, lambda _: (1 - fr) * M[lo] + fr * M[hi], 0, rt.device)
    elif mode == "scrambled":
        mu, Q, P = R["models"]["scrambled"]
        fn = make_patcher(mu, Q, lambda _: P(t0), 0, rt.device)
    else:
        mu, Q, P = R["models"]["ring"]
        fn = make_patcher(mu, Q, P, t0, rt.device)
    return {"probs": rt.probs(DAYS[day], 12, fn).round(4).tolist()}


app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
