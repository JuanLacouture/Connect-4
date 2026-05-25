#!/usr/bin/env python3
"""
Round-robin tournament - inter-group Connect-4 championship.
  Group A: Hydra        (Lacouture) - MCTS + UCB1 + persistent tree
  Group B: Minimax      (Sanchez)   - Minimax + Alpha-Beta, depth 6
  Group C: MCTSSoler    (Soler)     - MCTS + UCB1, no tree reuse

50 games per matchup, alternating first player.
Output: versus/roundrobin_results.json
"""

import sys, os, time, json, importlib.util
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
ROWS, COLS = 6, 7
GAMES_PER_MATCHUP = 50

AGENT_CONFIGS = {
    'Hydra':    ('groups/Group A/policy.py', 'Hydra'),
    'Minimax':  ('groups/Group B/policy.py', 'MinimaxPolicy'),
    'MCTSSoler':('groups/Group C/policy.py', 'MCTSAgentTorneo'),
}

# ── board helpers ─────────────────────────────────────────────────────────────
def _valid(b):
    return [c for c in range(COLS) if b[0, c] == 0]

def _drop(b, col, p):
    b2 = b.copy()
    for r in range(ROWS - 1, -1, -1):
        if b2[r, col] == 0:
            b2[r, col] = p
            return b2
    return b2

def _wins(b, p):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if b[r,c] == b[r,c+1] == b[r,c+2] == b[r,c+3] == p:
                return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if b[r,c] == b[r+1,c] == b[r+2,c] == b[r+3,c] == p:
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if b[r,c] == b[r+1,c+1] == b[r+2,c+2] == b[r+3,c+3] == p:
                return True
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if b[r,c] == b[r-1,c+1] == b[r-2,c+2] == b[r-3,c+3] == p:
                return True
    return False

# ── agent loader ──────────────────────────────────────────────────────────────
def load_agent(path, class_name):
    spec = importlib.util.spec_from_file_location('_agent_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()

def safe_mount(agent, timeout=2.0):
    try:
        agent.mount(timeout)
    except TypeError:
        try:
            agent.mount()
        except Exception:
            pass

# ── single game ───────────────────────────────────────────────────────────────
def play_game(agent_neg1, name_neg1, agent_pos1, name_pos1):
    """
    Play one game. Convention:
      agent_neg1 plays as -1 (first mover).
      agent_pos1 plays as +1 (second mover).
    Returns (winner_name_or_None, moves_list, duration_seconds).
    """
    board = np.zeros((ROWS, COLS), dtype=int)
    turn_order = [(-1, agent_neg1, name_neg1), (1, agent_pos1, name_pos1)]
    moves = []
    t_game = time.time()

    for move_num in range(ROWS * COLS):
        player_id, agent, name = turn_order[move_num % 2]

        t0 = time.time()
        col = int(agent.act(board))
        dt = time.time() - t0

        move_data = {
            'player': name,
            'col': col,
            'decision_time': round(dt, 4),
        }
        if hasattr(agent, 'last_iterations'):
            move_data['iterations'] = int(getattr(agent, 'last_iterations', 0))
        if hasattr(agent, 'last_depth'):
            move_data['depth'] = int(getattr(agent, 'last_depth', 0))
        moves.append(move_data)

        board = _drop(board, col, player_id)

        if _wins(board, player_id):
            return name, moves, time.time() - t_game

        if not _valid(board):
            return None, moves, time.time() - t_game   # draw

    return None, moves, time.time() - t_game

# ── matchup runner ────────────────────────────────────────────────────────────
def run_matchup(name_a, agent_a, name_b, agent_b, n=GAMES_PER_MATCHUP):
    games = []
    summary = {f'{name_a}_wins': 0, f'{name_b}_wins': 0, 'draws': 0}
    t_matchup = time.time()

    for i in range(n):
        # Reset agent state before each game
        safe_mount(agent_a)
        safe_mount(agent_b)

        # Alternate who plays first (-1)
        if i % 2 == 0:
            neg1_name, neg1_ag = name_a, agent_a
            pos1_name, pos1_ag = name_b, agent_b
        else:
            neg1_name, neg1_ag = name_b, agent_b
            pos1_name, pos1_ag = name_a, agent_a

        winner, moves, duration = play_game(neg1_ag, neg1_name, pos1_ag, pos1_name)

        if winner == name_a:
            summary[f'{name_a}_wins'] += 1
        elif winner == name_b:
            summary[f'{name_b}_wins'] += 1
        else:
            summary['draws'] += 1

        result_str = winner if winner else 'draw'
        elapsed = time.time() - t_matchup
        print(f"  Game {i+1:2d}/{n}: {neg1_name}(-1) vs {pos1_name}(+1) -> "
              f"{result_str} ({len(moves)} moves, {duration:.1f}s) "
              f"[total: {elapsed/60:.1f}min]")

        games.append({
            'game_idx': i,
            'first_player': neg1_name,
            'winner': winner,
            'total_moves': len(moves),
            'duration_seconds': round(duration, 2),
            'moves': moves,
        })

    return {
        'player_a': name_a,
        'player_b': name_b,
        'summary': summary,
        'games': games,
    }

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print("=" * 60)
    print("Connect-4 Inter-Group Round-Robin Tournament")
    print(f"  {GAMES_PER_MATCHUP} games per matchup  |  3 matchups  |  150 games total")
    print("  Estimated time: ~1.5 hours")
    print("=" * 60)

    print("\nLoading agents...")
    agents = {}
    for name, (path, cls_name) in AGENT_CONFIGS.items():
        agents[name] = load_agent(path, cls_name)
        print(f"  OK  {name:12s} <- {cls_name} ({path})")

    matchups = [
        ('Hydra',    'Minimax'),
        ('Hydra',    'MCTSSoler'),
        ('Minimax',  'MCTSSoler'),
    ]

    results = {'matchups': []}
    t_total = time.time()

    for name_a, name_b in matchups:
        print(f"\n{'='*60}")
        print(f"  {name_a}  vs  {name_b}  ({GAMES_PER_MATCHUP} games)")
        print(f"{'='*60}")

        matchup_result = run_matchup(name_a, agents[name_a], name_b, agents[name_b])
        results['matchups'].append(matchup_result)

        s = matchup_result['summary']
        print(f"\n  ── Result: {name_a}={s[f'{name_a}_wins']}  "
              f"{name_b}={s[f'{name_b}_wins']}  draws={s['draws']} ──")

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"Tournament complete in {total_min:.1f} minutes.")

    os.makedirs('versus', exist_ok=True)
    out = 'versus/roundrobin_results.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved -> {out}")
    print("=" * 60)

if __name__ == '__main__':
    main()
