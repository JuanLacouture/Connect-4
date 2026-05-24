"""
Round-robin tournament runner with per-move metrics collection.
Outputs: versus/roundrobin_results.json
"""

import sys
import os
import time
import json
import runpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connect4.connect_state import ConnectState


def load_agent(group_folder, class_name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groups", group_folder, "policy.py")
    namespace = runpy.run_path(path)
    cls = namespace[class_name]
    agent = cls()
    agent.mount()
    return agent


AGENTS = [
    ("Group A", "Titan"),
    ("Group B", "Hydra"),
    ("Group C", "Kronos"),
]

GAMES_PER_MATCHUP = 50


def print_board(board):
    """ASCII board. R=Red(-1), Y=Yellow(+1), .=empty."""
    sym = {-1: "R", 1: "Y", 0: "."}
    print("    0 1 2 3 4 5 6")
    for r in range(6):
        print("    " + " ".join(sym[int(board[r][c])] for c in range(7)))
    print("    0 1 2 3 4 5 6")


def play_game(agent_a, name_a, agent_b, name_b, a_goes_first):
    """
    Play one game. Returns dict with full move log and outcome.
    a_goes_first=True  -> agent_a plays as Red (-1, goes first)
    a_goes_first=False -> agent_b plays as Red (-1, goes first)
    """
    state = ConnectState()
    moves = []
    t_game_start = time.time()

    while not state.is_final():
        if a_goes_first:
            current_agent = agent_a if state.player == -1 else agent_b
            current_name = name_a if state.player == -1 else name_b
        else:
            current_agent = agent_b if state.player == -1 else agent_a
            current_name = name_b if state.player == -1 else name_a

        t0 = time.time()
        col = int(current_agent.act(state.board))
        decision_time = round(time.time() - t0, 4)

        move_record = {
            "player": current_name,
            "col": col,
            "decision_time": decision_time,
        }
        if hasattr(current_agent, "last_depth"):
            move_record["depth"] = current_agent.last_depth
        if hasattr(current_agent, "last_iterations"):
            move_record["iterations"] = current_agent.last_iterations

        moves.append(move_record)
        state = state.transition(col)

    duration = round(time.time() - t_game_start, 2)
    raw_winner = state.get_winner()

    if raw_winner == 0:
        winner = "draw"
    elif a_goes_first:
        winner = name_a if raw_winner == -1 else name_b
    else:
        winner = name_b if raw_winner == -1 else name_a

    first_player = name_a if a_goes_first else name_b

    return {
        "first_player": first_player,
        "winner": winner,
        "total_moves": len(moves),
        "duration_seconds": duration,
        "moves": moves,
        "final_board": state.board.tolist(),
    }


def run_matchup(group_a, cls_a, group_b, cls_b, n_games):
    print(f"\n{'='*50}")
    print(f"  {cls_a}  vs  {cls_b}  ({n_games} games)")
    print(f"{'='*50}")

    summary = {cls_a: 0, cls_b: 0, "draws": 0}
    games = []

    for i in range(n_games):
        agent_a = load_agent(group_a, cls_a)
        agent_b = load_agent(group_b, cls_b)

        a_goes_first = (i % 2 == 0)
        result = play_game(agent_a, cls_a, agent_b, cls_b, a_goes_first)
        result["game_idx"] = i

        w = result["winner"]
        if w == "draw":
            summary["draws"] += 1
        else:
            summary[w] += 1

        avg_time = round(
            sum(m["decision_time"] for m in result["moves"]) / len(result["moves"]), 3
        ) if result["moves"] else 0

        print(
            f"  Game {i+1:>2}/{n_games}: first={result['first_player']:<8} "
            f"winner={result['winner']:<8} "
            f"moves={result['total_moves']:>2}  "
            f"dur={result['duration_seconds']:>5.1f}s  "
            f"avg_decision={avg_time:.3f}s"
        )

        hydra_in_match = "Hydra" in [cls_a, cls_b]
        hydra_lost = hydra_in_match and result["winner"] not in ["Hydra", "draw"]
        if hydra_lost:
            loser_moves = [m for m in result["moves"] if m["player"] == "Hydra"]
            avg_iters = round(
                sum(m.get("iterations", 0) for m in loser_moves) / len(loser_moves), 1
            ) if loser_moves else 0
            print(f"  !! HYDRA LOST — avg_iters={avg_iters}  last 8 moves:")
            for m in result["moves"][-8:]:
                tag = f"iters={m['iterations']}" if "iterations" in m else f"depth={m.get('depth', '?')}"
                print(f"     {m['player']:<8} col={m['col']}  t={m['decision_time']:.3f}s  {tag}")
            print("  Final board (R=Red first player, Y=Yellow):")
            print_board(result["final_board"])
            print()

        games.append(result)

    print(f"\n  RESULT: {cls_a}={summary[cls_a]}  {cls_b}={summary[cls_b]}  draws={summary['draws']}")
    return {"player_a": cls_a, "player_b": cls_b, "summary": summary, "games": games}


def main():
    matchups_config = [
        (AGENTS[0], AGENTS[1]),
        (AGENTS[0], AGENTS[2]),
        (AGENTS[1], AGENTS[2]),
    ]

    all_results = {"matchups": []}

    for (group_a, cls_a), (group_b, cls_b) in matchups_config:
        result = run_matchup(group_a, cls_a, group_b, cls_b, GAMES_PER_MATCHUP)
        all_results["matchups"].append(result)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "versus", "roundrobin_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")

    print("\n" + "="*50)
    print("  LEADERBOARD")
    print("="*50)
    scores = {}
    for m in all_results["matchups"]:
        a, b = m["player_a"], m["player_b"]
        scores.setdefault(a, 0)
        scores.setdefault(b, 0)
        scores[a] += m["summary"][a]
        scores[b] += m["summary"][b]
    for name, wins in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name:<10} {wins} wins")


if __name__ == "__main__":
    main()
