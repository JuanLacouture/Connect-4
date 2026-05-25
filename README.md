# Connect-4 — Torneo Inter-Grupal de Agentes IA

**Curso:** Fundamentos de Inteligencia Artificial  
**Universidad de La Sabana — 2026-1**  
**Profesor:** Sergio Andres Amortegui

---

## Integrantes del grupo

| Integrante | Branch individual | Agente |
|------------|-------------------|--------|
| Juan Lacouture | [`Juan-Lacouture`](https://github.com/JuanLacouture/Connect-4/tree/Juan-Lacouture) | Hydra |
| Andres Sanchez | [`Andres-Sanchez`](https://github.com/JuanLacouture/Connect-4/tree/Andres-Sanchez) | Minimax |
| Santiago Soler | [`Santiago-Soler`](https://github.com/JuanLacouture/Connect-4/tree/Santiago-Soler) | MCTSSoler |

---

## Descripcion general

Este repositorio contiene el trabajo completo del grupo para el torneo de Connect-4. Cada integrante desarrollo de forma independiente sus agentes en su propia rama, eligio al mejor de ellos, y lo contribuyo a `main` para el torneo inter-grupal.

El torneo consiste en un **round-robin de 150 juegos** (50 por enfrentamiento, 3 matchups) entre los tres agentes finalistas. Los resultados completos y el analisis se encuentran en este branch (`main`).

---

## Ramas del repositorio

### `Juan-Lacouture` — Lacouture
Desarrollo intra-grupal de Juan: tres agentes propios (Titan, Hydra, Kronos) con Negamax + tabla de transposicion, MCTS + UCB1 + reutilizacion de arbol, y Negamax + heuristica de amenazas. Torneo interno de 50 partidas para seleccionar el mejor.

**Agente ganador:** **Hydra** — MCTS + UCB1 + rollouts heuristicos + persistent tree reuse.  
Ver detalles: [`Juan-Lacouture/tournament/README.md`](https://github.com/JuanLacouture/Connect-4/blob/Juan-Lacouture/tournament/README.md)

---

### `Andres-Sanchez` — Sanchez
Desarrollo intra-grupal de Andres: tres agentes (Minimax, MCTS, Q-Learning). Analisis parametrico de profundidad (depths 1-8), self-play, y estudio completo en notebook de entrega.

**Agente ganador:** **MinimaxPolicy** — Minimax + poda Alpha-Beta, profundidad 6, heuristica de ventanas de 4 celdas, orden centro-primero.  
Ver detalles: [`Andres-Sanchez/tournament/groups/Group B/readme.md`](https://github.com/JuanLacouture/Connect-4/blob/Andres-Sanchez/tournament/groups/Group%20B/readme.md)

---

### `Santiago-Soler` — Soler
Desarrollo intra-grupal de Santiago: agentes MCTS con distintas variantes y ajuste de pesos heuristicos.

**Agente ganador:** **MCTSAgentTorneo** — MCTS + UCB1 con rollouts heuristicos.  
Ver detalles: [`Santiago-Soler`](https://github.com/JuanLacouture/Connect-4/tree/Santiago-Soler)

---

### `main` — Torneo inter-grupal (este branch)
Contiene los tres agentes finalistas, el runner del torneo, los resultados en JSON y el notebook de analisis completo.

---

## Agentes en el torneo inter-grupal

| Grupo | Agente | Algoritmo | Clase |
|-------|--------|-----------|-------|
| A | **Hydra** (Lacouture) | MCTS + UCB1 + reutilizacion de arbol | `Hydra` |
| B | **Minimax** (Sanchez) | Minimax + Alpha-Beta, d=6 | `MinimaxPolicy` |
| C | **MCTSSoler** (Soler) | MCTS + UCB1 | `MCTSAgentTorneo` |

---

## Estructura del proyecto (branch main)

```
tournament/
  groups/
    Group A/policy.py        # Hydra — MCTS con persistent tree
    Group B/policy.py        # Minimax — Alpha-Beta d=6
    Group C/policy.py        # MCTSSoler — MCTS sin reutilizacion
  connect4/                  # Infraestructura base (Policy, estado, utils)
  run_roundrobin.py          # Runner del torneo (150 juegos)
  analisis_torneo.ipynb      # Notebook de analisis y graficas
  versus/
    roundrobin_results.json  # Resultados completos con metricas por movimiento
    elo_curve.png            # Evolucion Elo a lo largo del torneo
    ranking_comparativo.png  # Puntos vs Elo vs Win Rate
    h2h_heatmap.png          # Matriz de victorias directas
    decision_times.png       # Distribucion de tiempos de decision
    first_player_per_agent.png  # Ventaja del primer jugador por agente
    wr_by_role.png           # Win rate como 1ro vs 2do jugador
    mcts_iterations.png      # Iteraciones MCTS por juego
    tabla_tecnica.png        # Tabla comparativa de caracteristicas
```

---

## Requisitos

```bash
pip install numpy matplotlib jupyter
```

Python 3.10+

---

## Ejecutar el torneo

```bash
cd tournament
python run_roundrobin.py
```

Corre los 3 matchups (50 juegos cada uno) y guarda los resultados en `versus/roundrobin_results.json`.  
Tiempo estimado: ~1.5 horas. El runner alterna quien mueve primero en cada juego y registra por movimiento: columna, tiempo de decision, iteraciones MCTS y profundidad de busqueda.

---

## Ver el analisis

```bash
cd tournament
jupyter notebook analisis_torneo.ipynb
```

El notebook genera todas las graficas en `versus/` y cubre:

1. Tabla de puntos chess-style (W / L / D / Win%)
2. Matriz Head-to-Head con heatmap de calor
3. Curva Elo juego a juego — pondera calidad de victorias
4. Ranking consolidado por tres metricas (Pts, Elo, Win%)
5. Ventaja del primer jugador por agente (barras apiladas + comparativa)
6. Distribucion de tiempos de decision — violin + boxplot
7. Iteraciones MCTS a lo largo del torneo con media movil
8. Analisis tecnico — pros/contras de cada algoritmo, tabla comparativa
9. Resumen ejecutivo final

---

## Convencion del tablero

```
Jugador -1  →  primer jugador (mueve primero)
Jugador +1  →  segundo jugador
0           →  celda vacia
```

La asignacion de rol alterna entre juegos para eliminar el sesgo de turno.

---

## Notas tecnicas

- Los agentes implementan la interfaz `Policy` de `connect4/policy.py` — metodos `mount(timeout)` y `act(board)`.
- `mount()` se llama antes de cada juego para reiniciar estado interno (critico para el tree reuse de Hydra).
- MCTSSoler tiene un bug conocido: hardcodea `player=-1`, jugando de forma incorrecta cuando le corresponde el rol `+1`.
