"""AlphaZero-style training loop for Wizard.

  .venv/bin/python train.py --iters 8 --games 96 --sims 100

Each iteration:
  1. SELF-PLAY  — every CPU core plays full games against itself with tree search
                  (iteration 0 uses plain ISMCTS; afterwards the search is guided by the net).
                  We record (what the player saw, how search spread its visits, final round score).
  2. TRAIN      — a policy/value MLP learns to predict search's choice and the outcome.
                  Runs on the GPU if CUDA is available, else on CPU.
  3. EVALUATE   — the new NetBot plays a table vs plain MCTS with the same budget + 2 heuristics.

Checkpoints go to models/iter_XXX.npz and models/latest.npz (used by netbot.py).
"""
from __future__ import annotations

import argparse
import os

# self-play workers do tiny matmuls; stop each of them from spawning a BLAS thread per core
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bots import ismcts
from engine import DECK, N_ROUNDS, RoundState
from features import FEATURE_SIZE, N_ACTIONS, encode

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class PolicyValueNet(nn.Module):
    def __init__(self, hidden=(512, 512, 256)):
        super().__init__()
        layers, d = [], FEATURE_SIZE
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(d, N_ACTIONS)
        self.value = nn.Linear(d, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


# ----------------------------------------------------------------------------- self-play
def selfplay_game(job):
    seed, model_path, sims = job
    rng = random.Random(seed)
    np.random.seed(seed % 2**32)
    net = None
    if model_path:
        from netbot import NumpyNet, net_search
        net = NumpyNet(model_path)
    xs, pis, masks, zs = [], [], [], []
    n = 4
    for r in range(1, N_ROUNDS + 1):
        deck = list(DECK); rng.shuffle(deck)
        s = RoundState(n, r, (r - 1) % n, [deck[i * r:(i + 1) * r] for i in range(n)], deck[n * r])
        round_samples = []
        while not s.is_over():
            obs = s.observe(s.to_move)
            legal = obs.legal_actions
            if len(legal) == 1:
                s.apply(legal[0])
                continue
            if net is None:
                root = ismcts(obs, sims, rng=rng)
                visits = {a: ch.visits for a, ch in root.children.items()}
            else:
                root = net_search(obs, net, sims, rng=rng, root_noise=0.25)
                visits = {a: ch.N for a, ch in root.children.items()}
            pi = np.zeros(N_ACTIONS, np.float32)
            mask = np.zeros(N_ACTIONS, bool)
            off = 0 if obs.phase == "bid" else 14
            for a in legal:
                mask[off + a] = True
            for a, v in visits.items():
                pi[off + a] = v
            pi /= pi.sum()
            round_samples.append((encode(obs), pi, mask, obs.player))
            acts = list(visits)
            explore = obs.phase == "bid" or not obs.history  # bids + first trick: sample
            if explore:
                a = rng.choices(acts, weights=[visits[x] for x in acts])[0]
            else:
                a = max(acts, key=visits.get)
            s.apply(a)
        scale = 20 + 10 * r
        z = [x / scale for x in s.scores()]
        for x, pi, m, p in round_samples:
            xs.append(x); pis.append(pi); masks.append(m); zs.append(z[p])
    return np.array(xs), np.array(pis), np.array(masks), np.array(zs, np.float32)


# ----------------------------------------------------------------------------- training
def train_net(net, data, device, epochs, batch=512, lr=1e-3):
    X, P, M, Z = (torch.from_numpy(d).to(device) for d in data)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot_p = tot_v = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits, v = net(X[idx])
            logits = logits.masked_fill(~M[idx], -1e9)
            lp = -(P[idx] * F.log_softmax(logits, -1)).sum(-1).mean()
            lv = F.mse_loss(v, Z[idx])
            opt.zero_grad()
            (lp + lv).backward()
            opt.step()
            tot_p += lp.item() * len(idx); tot_v += lv.item() * len(idx)
        print(f"    epoch {ep + 1}/{epochs}  policy loss {tot_p / n:.3f}  value loss {tot_v / n:.4f}")


def export(net, path):
    np.savez(path, **{k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--games", type=int, default=64, help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=100, help="search simulations per move")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--buffer", type=int, default=3, help="iterations of data to keep")
    ap.add_argument("--eval-deals", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--resume", action="store_true", help="start from models/latest.npz")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(16, os.cpu_count()))
    print(f"training device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else
          "  (no CUDA - install the NVIDIA driver to train on GPU)"))
    os.makedirs(MODELS, exist_ok=True)
    net = PolicyValueNet().to(device)
    latest = os.path.join(MODELS, "latest.npz")
    model_path = None
    if a.resume and os.path.exists(latest):
        w = np.load(latest)
        net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files})
        model_path = latest
    buffer = []
    from arena import print_table, run_match
    for it in range(a.iters):
        t0 = time.time()
        mode = "net-guided search" if model_path else "plain ISMCTS (bootstrap)"
        print(f"\n=== iteration {it}: self-play {a.games} games with {mode}, {a.sims} sims/move ===")
        jobs = [(it * 1_000_003 + g, model_path, a.sims) for g in range(a.games)]
        with ProcessPoolExecutor(a.workers) as ex:
            games = list(ex.map(selfplay_game, jobs))
        data = tuple(np.concatenate(parts) for parts in zip(*games))
        buffer = (buffer + [data])[-a.buffer:]
        allx = tuple(np.concatenate(parts) for parts in zip(*buffer))
        print(f"  {len(data[0])} new samples ({len(allx[0])} in buffer), self-play took {time.time() - t0:.0f}s")
        t1 = time.time()
        train_net(net, allx, device, a.epochs)
        print(f"  training took {time.time() - t1:.1f}s on {device}")
        model_path = os.path.join(MODELS, f"iter_{it:03d}.npz")
        export(net, model_path)
        export(net, latest)
        os.environ["NETBOT_MODEL"] = model_path
        os.environ["NETBOT_SIMS"] = str(a.sims)
        stats = run_match(["netbot.py", f"mcts:{a.sims}", "heuristic", "heuristic"], deals=a.eval_deals,
                          seed=999, workers=a.workers)
        print(f"  eval (iteration {it}):")
        print_table(stats, [f"netbot{a.sims} (iter {it})", f"mcts:{a.sims}", "heuristic", "heuristic"])
        with open(os.path.join(MODELS, "log.tsv"), "a") as f:
            f.write(f"{it}\t{stats[0]['mean']:.1f}\t{stats[1]['mean']:.1f}\t{len(allx[0])}\n")


if __name__ == "__main__":
    main()
