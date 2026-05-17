"""Entrenador offline de la Q-table de Group C (self-play + vs aleatorio).

Uso:  cd tournament && python train_group_c.py
Genera groups/Group C/qtable.npz . NO se descubre como agente (sin subclase
Policy). Reutiliza el MISMO encoding canónico que el agente (import por ruta)
para que entrenamiento e inferencia sean idénticos.
"""
import importlib.util
import pathlib
from collections import defaultdict
import numpy as np

# === Variable de configuración del curso (rúbrica criterio 2) ===
TRAIN_EPISODES = 200_000

ALPHA = 0.30
GAMMA = 0.95
EPS_START, EPS_END = 1.0, 0.05
SELFPLAY_FRAC = 0.70
SEED = 7

ROOT = pathlib.Path(__file__).parent
GC = ROOT / "groups" / "Group C" / "policy.py"
OUT = ROOT / "groups" / "Group C" / "qtable.npz"

_spec = importlib.util.spec_from_file_location("_gc_policy", GC)
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)
canonical_key = gc.canonical_key
_drop_row = gc._drop_row
_legal = gc._legal
_wins_at = gc._wins_at

rng = np.random.default_rng(SEED)
Q: dict[tuple[int, int], np.ndarray] = defaultdict(
    lambda: np.zeros(7, dtype=np.float32))


def canon_cols(legal, mirrored):
    return [(6 - c if mirrored else c) for c in legal]


def choose(board, me, eps):
    legal = _legal(board)
    key, mirrored = canonical_key(board, me)
    if rng.random() < eps:
        c = int(rng.choice(legal))
    else:
        row = Q[key]
        best_c, best_q = legal[0], -np.inf
        for cc in legal:
            qc = 6 - cc if mirrored else cc
            if row[qc] > best_q:
                best_q, best_c = row[qc], cc
        c = best_c
    qc = 6 - c if mirrored else c
    return c, key, qc, canon_cols(legal, mirrored)


def apply_move(board, c, p):
    r = _drop_row(board, c)
    board[r, c] = p
    return r


def update_trajectory(traj, terminal_reward):
    # traj: lista de (key, qcol, legal_canon_next) en orden de juego del jugador.
    n = len(traj)
    for i in range(n):
        key, qcol, _ = traj[i]
        if i == n - 1:
            target = terminal_reward
        else:
            nkey, _, nlegal = traj[i + 1]
            nrow = Q[nkey]
            target = GAMMA * max(nrow[lc] for lc in nlegal)
        Q[key][qcol] += ALPHA * (target - Q[key][qcol])


def play_episode(eps):
    board = np.zeros((6, 7), dtype=int)
    cur = -1  # Rojo mueve primero
    selfplay = rng.random() < SELFPLAY_FRAC
    learner_color = None if selfplay else (-1 if rng.random() < 0.5 else 1)
    traj = {-1: [], 1: []}

    while True:
        legal = _legal(board)
        if not legal:
            winner = 0
            break
        learner_turn = selfplay or cur == learner_color
        if learner_turn:
            c, key, qc, lc_next = choose(board, cur, eps)
            traj[cur].append((key, qc, lc_next))
        else:
            c = int(rng.choice(legal))
        r = apply_move(board, c, cur)
        if _wins_at(board, r, c, cur):
            winner = cur
            break
        cur = -cur

    for color in (-1, 1):
        if not traj[color]:
            continue
        if winner == 0:
            rew = 0.0
        elif winner == color:
            rew = 1.0
        else:
            rew = -1.0
        update_trajectory(traj[color], rew)


def eval_vs_random(games=200):
    wins = losses = 0
    for g in range(games):
        board = np.zeros((6, 7), dtype=int)
        cur = -1
        learner = -1 if g % 2 == 0 else 1
        while True:
            legal = _legal(board)
            if not legal:
                break
            if cur == learner:
                key, mirrored = canonical_key(board, cur)
                row = Q.get(key)
                if row is None:
                    c = int(rng.choice(legal))
                else:
                    best_c, best_q = legal[0], -np.inf
                    for cc in legal:
                        qc = 6 - cc if mirrored else cc
                        if row[qc] > best_q:
                            best_q, best_c = row[qc], cc
                    c = best_c
            else:
                c = int(rng.choice(legal))
            r = apply_move(board, c, cur)
            if _wins_at(board, r, c, cur):
                if cur == learner:
                    wins += 1
                else:
                    losses += 1
                break
            cur = -cur
    return wins, losses, games


def main():
    for ep in range(TRAIN_EPISODES):
        frac = ep / TRAIN_EPISODES
        eps = EPS_START + (EPS_END - EPS_START) * frac
        play_episode(eps)
        if (ep + 1) % 10_000 == 0:
            w, l, n = eval_vs_random()
            print(f"ep {ep + 1:>7d}  eps={eps:.3f}  |Q|={len(Q):>8d}  "
                  f"vs random: {w}/{n} W, {l} L  (winrate={w / n:.3f})")

    keys = list(Q.keys())
    keys_hi = np.array([k[0] for k in keys], dtype=np.int64)
    keys_lo = np.array([k[1] for k in keys], dtype=np.int64)
    qvals = np.stack([Q[k] for k in keys]).astype(np.float32)
    # Podar estados ~irrelevantes (|Q| despreciable -> el agente usa el piso
    # heurístico para ellos igual). Reduce mucho el tamaño en disco.
    mask = np.abs(qvals).max(axis=1) > 1e-4
    keys_hi, keys_lo, qvals = keys_hi[mask], keys_lo[mask], qvals[mask]
    np.savez_compressed(OUT, keys_hi=keys_hi, keys_lo=keys_lo, qvals=qvals)
    print(f"Guardado {OUT}  ({mask.sum()} estados, de {len(keys)})")


if __name__ == "__main__":
    main()
