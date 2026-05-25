# Connect-4 — Torneo de Agentes IA
**Fundamentos de Inteligencia Artificial**

---

## Descripción

Torneo round-robin entre tres agentes de Connect-4. Cada agente implementa algoritmos distintos fundamentados en los módulos del curso. Se corrieron 50 partidas por emparejamiento (150 total), alternando quién va primero.

---

## Agentes

### Grupo A — Titan
**Algoritmo:** Negamax con poda Alpha-Beta + Tabla de Transposición + Profundización Iterativa

| Parámetro | Valor |
|-----------|-------|
| `TIME_LIMIT` | 1.75 s por jugada |
| Profundidad máxima | 13 niveles |
| Ordenación de movimientos | Centro-primero (`[3,2,4,1,5,0,6]`) |

**Fundamento de curso (≥60 %):**
- Árbol de juego (GCS) — Módulo 2
- Optimalidad de Bellman / value iteration — Módulo 9
- Tablas de transposición (path-independent quality) — Módulo 4
- GPI greedy `argmax q*(s,a)` — Módulo 8
- Zero-sum flip — Módulo 12

**Técnicas externas (≤40 %):** poda alpha-beta, profundización iterativa anytime.

**Clase:** `Titan` — `groups/Group A/policy.py`

---

### Grupo B — Hydra *(agente principal — 89 % win rate)*
**Algoritmo:** MCTS + UCB1 + Rollouts Heurísticos + Reutilización de Árbol + Bloqueo de Tres en Raya

| Parámetro | Valor |
|-----------|-------|
| `TIME_LIMIT` | 1.75 s por jugada |
| `C_UCB` | √2 ≈ 1.414 |
| Rollout | Win → Block → Centro-biased random |
| Tree reuse | Reutiliza subárbol del nodo sucesor entre turnos |

**Fundamento de curso (≥60 %):**
- MCTS — 4 fases (selección/expansión/simulación/backprop) — Módulo 13
- UCB1 como tree policy — Módulo 10
- Zero-sum self-play con signo alternado — Módulo 12
- GPI online (el árbol acumula estimaciones de valor) — Módulo 9 / 13

**Técnicas externas (≤40 %):** rollouts heurísticos, reutilización persistente del árbol, detección y bloqueo de amenazas de tres en raya.

**Clase:** `Hydra` — `groups/Group B/policy.py`

---

### Grupo C — Kronos
**Algoritmo:** Negamax Alpha-Beta + Heurística de Secuencias de Amenaza + UCB1 en raíz

| Parámetro | Valor |
|-----------|-------|
| `TIME_LIMIT` | 1.75 s por jugada |
| Bonus doble amenaza | +500 puntos en evaluación |
| Profundidad máxima | 13 niveles |

**Fundamento de curso (≥60 %):**
- Árbol de juego (GCS) — Módulo 2
- Optimalidad de Bellman — Módulo 9
- Teorema de Mejora de Política — Módulo 8
- UCB1 para selección en raíz — Módulo 10
- Zero-sum flip — Módulo 12

**Técnicas externas (≤40 %):** poda alpha-beta, análisis explícito de doble amenaza, profundización iterativa.

**Diferenciador vs Titan:** la heurística de Kronos detecta cuándo un jugador tiene ≥2 amenazas simultáneas accesibles (doble amenaza — el oponente solo puede bloquear una), otorgando un gran bonus de evaluación.

**Clase:** `Kronos` — `groups/Group C/policy.py`

---

## Resultados del Torneo

**50 partidas por emparejamiento, 150 totales.**

### Marcador global

| Agente | Victorias | Win Rate |
|--------|:---------:|:--------:|
| **Hydra** | **89 / 100** | **89 %** |
| Titan | 47 / 100 | 47 % |
| Kronos | 14 / 100 | 14 % |

### Head-to-head

| Emparejamiento | Resultado |
|----------------|-----------|
| Hydra vs Titan | 43 – 7 (86 %) |
| Hydra vs Kronos | 46 – 4 (92 %) |
| Titan vs Kronos | 40 – 10 (80 %) |

Resultados completos: `versus/roundrobin_results.json`

---

## Requisitos

```
Python >= 3.10
numpy
matplotlib
jupyter
```

Instalar dependencias:
```bash
pip install numpy matplotlib jupyter
```

---

## Estructura de Archivos

```
tournament/
├── connect4/                  # Framework del juego (no modificar)
│   ├── connect_state.py
│   ├── policy.py
│   ├── environment_state.py
│   ├── dtos.py
│   └── utils.py
├── groups/
│   ├── Group A/
│   │   └── policy.py          # Titan — Negamax + Alpha-Beta
│   ├── Group B/
│   │   └── policy.py          # Hydra — MCTS + UCB1
│   └── Group C/
│       └── policy.py          # Kronos — Negamax + Threat Heuristic
├── versus/
│   └── roundrobin_results.json
├── run_roundrobin.py           # Ejecuta el torneo completo
├── entrega.ipynb               # Notebook de análisis y validación
└── README.md                   # Este archivo
```

---

## Cómo Ejecutar

### Torneo completo (150 partidas, ~2 horas)

```bash
cd tournament
python run_roundrobin.py
```

Salida: `versus/roundrobin_results.json`

### Notebook de análisis

```bash
cd tournament
jupyter notebook entrega.ipynb
```

O abrir `entrega.ipynb` directamente desde VS Code y ejecutar todas las celdas.

> **Nota:** Las celdas de experimentos en vivo (vs aleatorio, sweep de TIME_LIMIT, comparación de versiones) toman aproximadamente 15–25 minutos en ejecutarse.

### Usar un agente en Python

```python
import sys, runpy, numpy as np
sys.path.insert(0, 'tournament')

ns = runpy.run_path('tournament/groups/Group B/policy.py')
hydra = ns['Hydra']()
hydra.mount()

board = np.zeros((6, 7), dtype=int)
col = hydra.act(board)
print(f"Hydra juega en columna {col}")
```

---

## Notas de Implementación

- **Sin `@override` ni `from typing import override`** — garantiza compatibilidad con el grader de Gradescope.
- Todos los agentes implementan la interfaz `Policy` (`connect4/policy.py`): métodos `mount()` y `act(board)`.
- El método `mount()` inicializa estado persistente (tabla de transposición para Titan/Kronos, árbol MCTS para Hydra).
- `act(board)` recibe el tablero como `numpy.ndarray` de forma `(6, 7)` con valores `{-1, 0, 1}` y retorna un entero `[0, 6]`.
