# Connect-4 — Agente MCTSAgentTorneo
**Fundamentos de Inteligencia Artificial · Santiago Soler Prado · 2026.1**

---

## Descripción

Agente de Connect-4 basado en **Monte Carlo Tree Search (MCTS)** con política **UCB1** y reutilización persistente del árbol entre turnos. Implementa búsqueda online estocástica: el árbol de decisión se construye durante la partida sin ningún entrenamiento offline. La técnica de *tree reuse* produce un crecimiento de más de 500× en iteraciones a lo largo de una partida (de ~300 en el turno 1 a +160 000 en el turno 17).

---

## Agente — MCTSAgentTorneo

**Algoritmo:** MCTS + UCB1 + Rollout Heurístico + Tree Reuse

| Parámetro | Valor |
|-----------|-------|
| `simulations` | 2 000 por movimiento |
| `time_limit` | 5.5 s (ajustable via `mount(timeout)`) |
| `C_UCB` | √2 ≈ 1.414 |
| Rollout | Win → Block → Centro-biased random |
| Tree reuse | Subárbol del sucesor persiste entre turnos |
| Entrenamiento offline | Ninguno |

**Fundamento de curso (≥60 %):**
- MCTS — 4 fases (selección / expansión / simulación / backpropagación) — Módulo 13
- UCB1 como tree policy — Módulo 10
- Zero-sum self-play con signo alternado en backpropagación — Módulo 12
- GPI online: el árbol acumula estimaciones (W, N) por nodo — Módulos 9 / 13
- MDP con espacio de estados ~4.5×10¹² — Módulo 2

**Técnicas externas (≤40 %):** rollouts heurísticos (win → block → centro), reutilización persistente del árbol entre turnos, priorización de movimientos inmediatos antes de MCTS.

**Priorización del método `act()`:**
1. Una sola columna libre → jugarla directamente
2. Movimiento ganador propio inmediato
3. Bloqueo de victoria inmediata del oponente
4. MCTS completo → hijo con más visitas

**Clase:** `MCTSAgentTorneo` (alias `AgenteMCTS`) — `groups/Group_A/policy.py`

---

## Resultados del Análisis

Análisis completo disponible en `entrega.ipynb`.

### Desempeño por oponente

| Oponente | Win Rate | Notas |
|----------|:--------:|-------|
| Aleatorio | **85 %** | Convergencia desde partida 10–15, sin deriva |
| Minimax (α-β) | 25 % | Búsqueda exhaustiva supera MCTS en profundidad ≥6 |
| Heurística | **95 %** | Domina completamente |

### Desempeño por rol

| Rol | Win Rate | Diferencial |
|-----|:--------:|:-----------:|
| Primer movimiento (−1, Rojo) | 85 % | +6 pp |
| Segundo movimiento (+1, Amarillo) | 79 % | — |

> El diferencial de +6 pp es excepcional frente a +18–20 pp típico de agentes determinísticos. El *tree reuse* compensa la desventaja de turno.

### Desempeño por cuartil de iteraciones

| Cuartil | Iteraciones medias | Win Rate |
|---------|--------------------|:--------:|
| Q1 | ~990 | 88 % |
| Q2 | ~2 600 | **100 %** |
| Q3 | ~13 400 | **100 %** |
| Q4 | ~28 800 | 68 % |

> Zona óptima Q2–Q3. La degradación en Q4 es un confound: en partidas largas el oponente también refina su búsqueda.

---

## Requisitos

```
Python >= 3.10
numpy
matplotlib
seaborn
reportlab
jupyter
```

Instalar dependencias:

```bash
python -m pip install numpy matplotlib seaborn reportlab jupyter
```

> **Nota Windows:** Si `pip` no está en PATH usar `python -m pip install ...`

---

## Estructura de Archivos

```
tournament/
├── connect4/                    # Framework del juego (no modificar)
│   ├── connect_state.py
│   ├── policy.py
│   ├── environment_state.py
│   ├── dtos.py
│   └── utils.py
├── groups/
│   └── Group_A/
│       └── policy.py            # MCTSAgentTorneo — MCTS + UCB1
├── entrega.ipynb                # Notebook de análisis y validación
├── generar_graficas.py          # Genera 6 gráficas PNG del análisis
├── generar_pdf_reto.py          # Genera el informe PDF (2 páginas)
└── README.md                    # Este archivo
```

---

## Cómo Ejecutar

### Usar el agente en Python

```python
import sys, numpy as np
sys.path.insert(0, 'tournament')

from groups.Group_A.policy import MCTSAgentTorneo

agente = MCTSAgentTorneo(simulations=2000, time_limit=5.5)
agente.mount()

board = np.zeros((6, 7), dtype=int)
col = agente.act(board)
print(f"MCTSAgentTorneo juega en columna {col}")
```

### Notebook de análisis

```bash
cd tournament
jupyter notebook entrega.ipynb
```

O abrir `entrega.ipynb` desde VS Code y ejecutar todas las celdas.

> **Nota:** Las celdas de experimentos en vivo (vs aleatorio, sweep de configuración, análisis de iteraciones) toman aproximadamente 15–25 minutos en ejecutarse.

### Generar gráficas e informe PDF

```bash
cd tournament

# Generar 6 gráficas PNG
python generar_graficas.py

# Generar informe PDF (2 páginas)
python generar_pdf_reto.py
```

Salida: carpeta `graficas_informe/` con 6 PNG e `Informe_Agente_MCTS.pdf`.

---

## Notas de Implementación

- **Sin `@override`** — garantiza compatibilidad con el grader de Gradescope.
- Todos los agentes implementan la interfaz `Policy` (`connect4/policy.py`): métodos `mount()` y `act(board)`.
- `mount(timeout)` acepta argumento opcional. Si el torneo pasa un `timeout`, el agente calibra `time_limit = timeout * 0.90` para nunca exceder el tiempo asignado.
- `act(board)` recibe el tablero como `numpy.ndarray` de forma `(6, 7)` con valores `{-1, 0, 1}` y retorna un entero `[0, 6]`.
- **`__slots__` en `MCTSNode`**: elimina el `__dict__` por objeto, reduciendo overhead de memoria y acelerando la creación de miles de nodos por segundo.
- Las operaciones de tablero (`get_free_cols`, `make_move`, `check_winner`, `winning_move`, `heuristic_rollout`) son **funciones puras** separadas de las clases para máxima velocidad.
- `AgenteMCTS` es alias de `MCTSAgentTorneo` para compatibilidad con el sistema de descubrimiento automático del torneo.
