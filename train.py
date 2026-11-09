"""AlphaZero-style training loop for Wizard.

  .venv/bin/python train.py                   # full run (~1-1.5 h on 32 threads + GPU)
  .venv/bin/python train.py --quick           # 5-minute smoke test
  .venv/bin/python train.py --resume          # continue from models/latest.npz (reuses bootstrap data)

Phases:
  0. BOOTSTRAP  Thousands of games of plain ISMCTS self-play on every CPU core.
                Cached in models/bootstrap.npz so reruns skip it.
  1. ITERATE    Net-guided self-play -> train on a replay buffer -> evaluate vs MCTS-200.
                The checkpoint with the best margin is kept as models/best.npz.
  2. FINAL      Re-test best.npz with several playout-blend settings and bake the winner
                into the file, so `net` in tournaments uses the strongest setting.

Each decision stores the mover's encoded view, search's visit distribution (policy target)
and the mover's normalized round score (value target, blended with search's own estimate
via --q-weight, which is far less noisy). About a third of the time it also stores the same
moment from each other player's view as a value-only row: free extra value data.
"""
from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import random
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bots import ismcts
from engine import DECK, N_ROUNDS, RoundState
from features import FEATURE_SIZE, N_ACTIONS, encode_state

OTHER_VIEW_P = 0.34   # chance to also store a value-only row for each non-moving player

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROGRESS = os.path.join(MODELS, "progress.json")


class _Tee:
    """Mirror stdout into models/train_output.txt (read by dashboard.py)."""

    def __init__(self, path, mode):
        self.f = open(path, mode, buffering=1)
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def progress(phase, done=0, total=0, **extra):
    """Write models/progress.json atomically (read by dashboard.py)."""
    if not hasattr(progress, "t0") or progress.phase != phase:
        progress.t0, progress.phase = time.time(), phase
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "done": done, "total": total, "phase_start": progress.t0,
                   "now": time.time(), "pid": os.getpid(), **extra}, f)
    os.replace(tmp, PROGRESS)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden=(512, 512, 512, 256)):
        super().__init__()
        layers, d = [], FEATURE_SIZE
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(d, N_ACTIONS)
        self.value = nn.Linear(d, 1)  # this player's normalized round score

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h)


# ----------------------------------------------------------------------------- self-play
def selfplay_game(job):
    seed, model_path, sims, mix, q_weight = job
    rng = random.Random(seed)
    np.random.seed(seed % 2**32)
    net = None
    if model_path:
        from netbot import NumpyNet, net_search
        net = NumpyNet(model_path)
    xs, pis, masks, vs = [], [], [], []
    n = 4
    for r in range(1, N_ROUNDS + 1):
        deck = list(DECK); rng.shuffle(deck)
        s = RoundState(n, r, (r - 1) % n, [deck[i * r:(i + 1) * r] for i in range(n)], deck[n * r])
        samples = []
        while not s.is_over():
            obs = s.observe(s.to_move)
            legal = obs.legal_actions
            if len(legal) == 1:
                s.apply(legal[0])
                continue
            if net is None:
                root = ismcts(obs, sims, rng=rng)
                stats = {a: (ch.visits, ch.total) for a, ch in root.children.items()}
            else:
                root = net_search(obs, net, sims, rng=rng, root_noise=0.25, mix=mix)
                stats = {a: (ch.N, ch.W) for a, ch in root.children.items()}
            off = 0 if obs.phase == "bid" else 14
            pi = np.zeros(N_ACTIONS, np.float32)
            mask = np.zeros(N_ACTIONS, bool)
            mask[[off + a for a in legal]] = True
            for a, (v, _w) in stats.items():
                pi[off + a] = v
            pi /= pi.sum()
            q = sum(w for _v, w in stats.values()) / max(1, sum(v for v, _w in stats.values()))
            samples.append((encode_state(s, s.to_move), pi, mask, s.to_move, q))
            for p in range(n):   # free extra value data: the same moment seen by other players
                if p != s.to_move and rng.random() < OTHER_VIEW_P:
                    samples.append((encode_state(s, p), None, None, p, None))
            acts = list(stats)
            if obs.phase == "bid" or not obs.history:   # explore on bids + first trick
                a = rng.choices(acts, weights=[stats[x][0] for x in acts])[0]
            else:
                a = max(acts, key=lambda x: stats[x][0])
            s.apply(a)
        scale = 20 + 10 * r
        z = [x / scale for x in s.scores()]
        for x, pi, m, p, q in samples:
            if pi is None:  # value-only row
                pi, m, target = np.zeros(N_ACTIONS, np.float32), np.zeros(N_ACTIONS, bool), z[p]
            else:
                target = (1 - q_weight) * z[p] + q_weight * q
            xs.append(x); pis.append(pi); masks.append(m); vs.append(target)
    return (np.array(xs, np.float16), np.array(pis, np.float16), np.array(masks, bool),
            np.array(vs, np.float32)[:, None])


def run_selfplay(n_games, seed0, model_path, sims, mix, q_weight, workers, phase="self-play"):
    jobs = [(seed0 + g, model_path, sims, mix, q_weight) for g in range(n_games)]
    games = []
    progress(phase, 0, n_games)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(selfplay_game, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            games.append(fut.result())
            if k % 4 == 0 or k == n_games:
                progress(phase, k, n_games)
    return tuple(np.concatenate(parts) for parts in zip(*games))


# ----------------------------------------------------------------------------- training
def losses(net, X, P, M, V):
    logits, v = net(X.float())
    logits = logits.masked_fill(~M, -1e9)
    P = P.float()
    n_policy = P.sum().clamp(min=1.0)   # each policy row sums to 1; value-only rows are all zero
    lp = -(P * F.log_softmax(logits, -1)).sum() / n_policy
    lv = F.mse_loss(v, V)
    return lp, lv


def fit(net, opt, data, device, max_epochs, batch, value_weight=2.0, patience=2):
    """Train with a 5% hold-out; stop when validation loss stops improving, keep best weights."""
    n = len(data[0])
    perm = np.random.permutation(n)
    k = max(1, int(n * 0.05))
    va_idx, tr_idx = perm[:k], perm[k:]
    T = [torch.from_numpy(d).to(device) for d in data]
    va = [t[torch.from_numpy(va_idx).to(device)] for t in T]
    tr_idx = torch.from_numpy(tr_idx).to(device)
    best_loss, best_state, bad, hist = 1e9, None, 0, None
    for ep in range(max_epochs):
        net.train()
        sh = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(sh), batch):
            idx = sh[i:i + batch]
            lp, lv = losses(net, *(t[idx] for t in T))
            opt.zero_grad()
            (lp + value_weight * lv).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vlp, vlv = (x.item() for x in losses(net, *va))
        total = vlp + value_weight * vlv
        if total < best_loss - 1e-4:
            best_loss, bad, hist = total, 0, (ep + 1, vlp, vlv)
            best_state = {k_: v_.detach().clone() for k_, v_ in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return hist, float(va[3].var().item())


def export(net, path, mix):
    w = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    np.savez(path, meta_mix=np.float32(mix), **w)


def set_mix(path, mix):
    w = dict(np.load(path))
    w["meta_mix"] = np.float32(mix)
    np.savez(path, **w)


def evaluate(model_path, sims, deals, workers, seed=4242, label="eval"):
    from arena import run_match
    progress(label, 0, deals * 4)
    st = run_match([f"net:{model_path}:{sims}", "mcts:200", "heuristic", "heuristic"],
                   deals=deals, seed=seed, workers=workers)
    return st[0]["mean"], st[1]["mean"], st[0]["ms"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap-games", type=int, default=4096, help="plain-ISMCTS games for phase 0")
    ap.add_argument("--bootstrap-sims", type=int, default=100)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=512, help="net self-play games per iteration")
    ap.add_argument("--sims", type=int, default=200, help="search sims per move in net self-play")
    ap.add_argument("--buffer", type=int, default=1_500_000, help="replay buffer size (samples)")
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--q-weight", type=float, default=0.7, help="search-estimate share of the mover's value target")
    ap.add_argument("--mix-start", type=float, default=0.5)
    ap.add_argument("--mix-end", type=float, default=0.0)
    ap.add_argument("--mix-iters", type=int, default=6, help="iterations to go from mix-start to mix-end")
    ap.add_argument("--eval-deals", type=int, default=24, help="x4 seat rotations")
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.bootstrap_games, a.iters, a.games, a.sims, a.eval_deals, a.max_epochs = 128, 2, 64, 100, 4, 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(16, os.cpu_count()))
    os.makedirs(MODELS, exist_ok=True)
    sys.stdout = _Tee(os.path.join(MODELS, "train_output.txt"), "a" if a.resume else "w")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}  args: {' '.join(sys.argv[1:]) or '(defaults)'}", flush=True)
    print(f"training device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else
          "  (no CUDA - install the NVIDIA driver to train on GPU)"), flush=True)
    latest, best = os.path.join(MODELS, "latest.npz"), os.path.join(MODELS, "best.npz")
    boot_path = os.path.join(MODELS, "bootstrap.npz" if not a.quick else "bootstrap_quick.npz")
    log_path = os.path.join(MODELS, "log.tsv")

    net = PolicyValueNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    start, best_margin = 0, -1e9
    if a.resume and os.path.exists(latest):
        w = np.load(latest)
        net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if not k.startswith("meta_")})
        rows = [l.split("\t") for l in open(log_path) if l[0].isdigit()] if os.path.exists(log_path) else []
        if rows:
            start = int(rows[-1][0]) + 1
            best_margin = max(float(r[3]) for r in rows)
    else:
        open(log_path, "w").write("iter\tnetbot\tmcts\tmargin\tsamples\tmix\tval_policy\tval_value\tseconds\n")

    # ---- phase 0: bootstrap data
    if os.path.exists(boot_path):
        d = np.load(boot_path)
        boot = (d["x"], d["pi"], d["mask"], d["v"])
        print(f"loaded {len(boot[0])} bootstrap samples from {boot_path}", flush=True)
    else:
        t0 = time.time()
        print(f"\n=== bootstrap: {a.bootstrap_games} games of plain ISMCTS-{a.bootstrap_sims} self-play ===", flush=True)
        boot = run_selfplay(a.bootstrap_games, 7_000_000, None, a.bootstrap_sims, 0.0, a.q_weight, a.workers,
                            phase="bootstrap self-play")
        np.savez(boot_path, x=boot[0], pi=boot[1], mask=boot[2], v=boot[3])
        print(f"  {len(boot[0])} rows in {time.time() - t0:.0f}s", flush=True)
    buffer = [boot]

    def buffer_data():
        allx = tuple(np.concatenate(parts) for parts in zip(*buffer))
        if len(allx[0]) > a.buffer:  # keep the newest samples
            allx = tuple(x[-a.buffer:] for x in allx)
        return allx

    if start == 0:
        t0 = time.time()
        progress("training on bootstrap")
        (ep, vlp, vlv), var = fit(net, opt, buffer_data(), device, a.max_epochs, a.batch)
        print(f"  trained on bootstrap: {ep} epochs, val policy {vlp:.3f}, val value {vlv:.4f} "
              f"(target variance {var:.4f}) in {time.time() - t0:.0f}s", flush=True)
        mix0 = a.mix_start
        export(net, latest, mix0)
        shutil.copy(latest, os.path.join(MODELS, "iter_boot.npz"))
        nb, mc, ms = evaluate(latest, a.eval_sims, a.eval_deals, a.workers)
        best_margin = nb - mc
        shutil.copy(latest, best)
        print(f"  eval: netbot {nb:.1f} vs mcts:200 {mc:.1f} (margin {nb - mc:+.1f}, {ms:.0f} ms/move)", flush=True)

    # ---- phase 1: net-guided iterations
    for it in range(start, start + a.iters):
        t0 = time.time()
        frac = min(1.0, it / max(1, a.mix_iters))
        mix = a.mix_start + (a.mix_end - a.mix_start) * frac
        print(f"\n=== iteration {it}: {a.games} games of net-guided self-play, blend {mix:.2f}, "
              f"{a.sims} sims/move ===", flush=True)
        data = run_selfplay(a.games, (it + 1) * 1_000_003, latest, a.sims, mix, a.q_weight, a.workers,
                            phase=f"iteration {it} self-play")
        buffer.append(data)
        t_sp = time.time() - t0
        for g in opt.param_groups:
            g["lr"] = a.lr * max(0.15, 0.85 ** (it + 1))
        t1 = time.time()
        progress(f"iteration {it} training")
        (ep, vlp, vlv), var = fit(net, opt, buffer_data(), device, a.max_epochs, a.batch)
        next_mix = a.mix_start + (a.mix_end - a.mix_start) * min(1.0, (it + 1) / max(1, a.mix_iters))
        path = os.path.join(MODELS, f"iter_{it:03d}.npz")
        export(net, path, next_mix)
        shutil.copy(path, latest)
        nb, mc, ms = evaluate(path, a.eval_sims, a.eval_deals, a.workers, label=f"iteration {it} eval")
        margin = nb - mc
        tag = ""
        if margin > best_margin:
            best_margin = margin
            shutil.copy(path, best)
            tag = "  ** new best **"
        n_buf = len(buffer_data()[0])
        print(f"  +{len(data[0])} samples (buffer {n_buf}) | self-play {t_sp:.0f}s | train {ep} ep "
              f"{time.time() - t1:.0f}s | val policy {vlp:.3f} value {vlv:.4f} (var {var:.4f})", flush=True)
        print(f"  eval: netbot {nb:.1f} vs mcts:200 {mc:.1f} (margin {margin:+.1f}, {ms:.0f} ms/move)"
              f"{tag}   [{time.time() - t0:.0f}s]", flush=True)
        with open(log_path, "a") as f:
            f.write(f"{it}\t{nb:.1f}\t{mc:.1f}\t{margin:.1f}\t{n_buf}\t{next_mix:.2f}\t{vlp:.4f}\t{vlv:.5f}\t"
                    f"{time.time() - t0:.0f}\n")

    # ---- phase 2: pick the best blend for the best checkpoint
    print("\n=== final: tuning playout blend for models/best.npz ===", flush=True)
    results = {}
    for m in (0.0, 0.25, 0.5):
        set_mix(best, m)
        nb, mc, ms = evaluate(best, a.eval_sims, a.eval_deals * 2, a.workers, seed=777, label=f"final: blend {m:.2f}")
        results[m] = nb - mc
        print(f"  blend {m:.2f}: netbot {nb:.1f} vs mcts:200 {mc:.1f} (margin {nb - mc:+.1f}, {ms:.0f} ms/move)", flush=True)
    m_best = max(results, key=results.get)
    set_mix(best, m_best)
    progress("done")
    print(f"best.npz uses blend {m_best:.2f}. Play it: python tournament.py net:models/best.npz mcts:200 heuristic heuristic")


if __name__ == "__main__":
    main()
