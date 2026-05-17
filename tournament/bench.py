"""Harness del gate del reto. Standalone: NO importa tournament.py (pydantic).

Uso:  cd tournament && python bench.py [N]
Valida cada agente vs aleatorio (N por color, ambos colores):
  GATE = 0 derrotas Y winrate >= 0.5 sobre 2*N partidas.
Imprime PASS/FAIL + tiempo máx por jugada. Matriz self-play para el reporte.
"""
import sys
import time
import importlib.util
import inspect
import pathlib
import numpy as np

from connect4.policy import Policy
from connect4.connect_state import ConnectState

ROOT = pathlib.Path(__file__).parent
GROUPS = ROOT / "groups"


def load_agents() -> dict[str, type]:
    agents: dict[str, type] = {}
    for py in sorted(GROUPS.rglob("policy.py")):
        group = py.parent.name
        spec = importlib.util.spec_from_file_location(
            f"_bench_{group.replace(' ', '_')}", py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Policy) and obj is not Policy:
                agents[group] = obj
                break
    return agents


class RandomPolicy(Policy):
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        cols = [c for c in range(7) if s[0, c] == 0]
        return int(self.rng.choice(cols))


def play_game(red: Policy, yellow: Policy) -> tuple[int, float]:
    red.mount()
    yellow.mount()
    st = ConnectState()
    max_t = 0.0
    while not st.is_final():
        agent = red if st.player == -1 else yellow
        t0 = time.perf_counter()
        a = int(agent.act(st.board))
        max_t = max(max_t, time.perf_counter() - t0)
        st = st.transition(a)  # lanza ValueError si col ilegal -> FALLA ruidoso
    return st.get_winner(), max_t


def evaluate(name: str, AgentCls: type, n: int) -> bool:
    overall_w = overall_l = overall_d = 0
    max_t = 0.0
    for color, label in ((-1, "Rojo"), (1, "Amarillo")):
        w = l = d = 0
        for i in range(n):
            agent = AgentCls()
            rnd = RandomPolicy(seed=10_000 + i)
            if color == -1:
                res, mt = play_game(agent, rnd)
            else:
                res, mt = play_game(rnd, agent)
            max_t = max(max_t, mt)
            if res == color:
                w += 1
            elif res == 0:
                d += 1
            else:
                l += 1
        print(f"  {name} como {label:8s}: {w}W {l}L {d}D  "
              f"winrate={w / n:.3f}")
        overall_w += w
        overall_l += l
        overall_d += d
    tot = 2 * n
    wr = overall_w / tot
    ok = overall_l == 0 and wr >= 0.5
    print(f"  {name} TOTAL: {overall_w}W {overall_l}L {overall_d}D  "
          f"winrate={wr:.3f}  max_move={max_t * 1000:.1f}ms  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def selfplay_matrix(agents: dict[str, type], n: int = 50) -> None:
    print("\n=== Matriz self-play (para el reporte) ===")
    names = list(agents)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            aw = bw = dd = 0
            for k in range(n):
                if k % 2 == 0:
                    res, _ = play_game(agents[a](), agents[b]())
                    if res == -1:
                        aw += 1
                    elif res == 1:
                        bw += 1
                    else:
                        dd += 1
                else:
                    res, _ = play_game(agents[b](), agents[a]())
                    if res == -1:
                        bw += 1
                    elif res == 1:
                        aw += 1
                    else:
                        dd += 1
            print(f"  {a} vs {b}: {aw}-{bw} ({dd}D)")


def check_winner_helpers(agents: dict[str, type]) -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        st = ConnectState()
        steps = int(rng.integers(0, 20))
        for _ in range(steps):
            if st.is_final():
                break
            free = st.get_free_cols()
            st = st.transition(int(rng.choice(free)))
        ref = st.get_winner()
        for group, cls in agents.items():
            mod = sys.modules[cls.__module__]
            if hasattr(mod, "_winner"):
                assert mod._winner(st.board) == ref, (
                    f"{group}._winner != ConnectState.get_winner")
    print("assert _winner == ConnectState.get_winner: OK")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    only = sys.argv[2] if len(sys.argv) > 2 else None
    agents = load_agents()
    if only == "matrix":
        check_winner_helpers(agents)
        selfplay_matrix(agents, n)
        return
    print(f"Agentes: {list(agents)}  |  N={n} por color  "
          f"{'(solo ' + only + ')' if only else ''}\n", flush=True)
    check_winner_helpers(agents)
    print(flush=True)
    all_ok = True
    for name, cls in agents.items():
        if only and name != only:
            continue
        ok = evaluate(name, cls, n)
        all_ok &= ok
    if not only:
        selfplay_matrix(agents)
    print(f"\n=== GATE {'PASS' if all_ok else 'FAIL'} ===", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
