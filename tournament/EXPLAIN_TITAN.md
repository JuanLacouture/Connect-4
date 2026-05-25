# Titan — Cómo Explicarlo en la Presentación

## ¿Qué hace este agente?

Titan es un agente que **piensa hacia el futuro**. Analiza el tablero de Connect-4 como si fuera un árbol de posibilidades: "si yo pongo aquí, el rival puede responder así, y yo puedo responder asá..." Titan evalúa todas esas ramas y elige la jugada que lo lleva al mejor futuro posible, asumiendo que el rival también juega de manera óptima. No improvisa — calcula.

---

## Técnica 1: Árbol de Juego + Negamax

**Analogía:** Imagina un laberinto donde tú y un rival se turnan para elegir el camino. Tú quieres llegar al tesoro (+1); el rival quiere que caigas en la trampa (-1). En cada bifurcación, **el que tiene el turno elige el camino que más le conviene a él**. Negamax dice: *"mi mejor jugada es aquella que hace que el mejor camino de mi rival sea lo peor para mí."*

Matemáticamente: `v*(s) = max_a [ -v*(s') ]` — en cada nivel se invierte el signo. Lo que es bueno para mí es malo para ti.

**Cómo aplica al Connect-4:** Titan construye un árbol donde cada nodo es un tablero y cada rama es una ficha colocada en una columna. Baja por el árbol asumiendo que ambos jugadores eligen su mejor opción. Al llegar al fondo, usa una función que puntúa el tablero.

**Frase clave:** *"Juego óptimo contra un rival óptimo."*

---

## Técnica 2: Poda Alpha-Beta

**Analogía:** Estás contratando empleados. Ya encontraste uno que saca 8/10. Evaluás al siguiente candidato y el primer panel le da 3/10. **¿Seguís entrevistándolo?** No — aunque los demás paneles le den 10/10 no puede superar al que ya tenés. Cortás ahí.

Alpha-beta hace exactamente eso con ramas del árbol: si ya sé que esta rama no puede ser mejor que lo que ya encontré, **ni la analizo**. Mismo resultado, mucho menos trabajo.

**Cómo aplica al Connect-4:** Con 7 columnas posibles en cada turno, el árbol crece muy rápido. Alpha-beta elimina ramas que nunca serán elegidas, permitiéndole a Titan llegar más profundo en el mismo tiempo.

**Frase clave:** *"Saltá lo que nunca vas a elegir."*

---

## Técnica 3: Profundización Iterativa

**Analogía:** Google Maps calculando una ruta. Primero calcula la ruta mirando 1 minuto hacia adelante. Luego 2 minutos. Luego 3. Si el tiempo se acaba, te da la mejor ruta que encontró hasta ese momento. **Siempre tenés una respuesta disponible.**

Titan funciona igual: primero busca a profundidad 1 (un turno hacia adelante), luego 2, luego 3... Si el tiempo de 1.75 segundos se agota, usa la respuesta de la última profundidad completada.

**Cómo aplica al Connect-4:** Al inicio de la partida puede llegar a profundidad 10+ porque hay pocos patrones útiles. Al final, puede llegar más profundo porque hay menos columnas válidas.

**Frase clave:** *"Siempre tengo una respuesta de respaldo."*

---

## Técnica 4: Tabla de Transposición

**Analogía:** Un libro de problemas de ajedrez. Si el problema de la página 47 ya lo resolviste, y en otra partida llegás a la misma posición por un camino distinto, **no lo volvés a resolver — consultás el libro**.

En Connect-4, el mismo tablero se puede alcanzar por diferentes secuencias de jugadas (ej: poner en columna 3 antes o después que en columna 5). La tabla de transposición guarda los resultados ya calculados, indexados por el estado del tablero.

**Cómo aplica al Connect-4:** Titan guarda cada tablero analizado en un diccionario (`tt`). Si llega al mismo tablero por otro camino y ya lo analizó a suficiente profundidad, reutiliza el resultado instantáneamente.

**Frase clave:** *"Memoria de tableros ya resueltos."*

---

## WAY TO GO — Script de Presentación

**Palabras clave:** árbol de juego · negamax · poda alpha-beta · profundización iterativa · tabla de transposición · Bellman · sign flip

**Flujo hablado:**

1. *"Titan piensa el juego como un árbol: cada rama es una jugada posible. El algoritmo se llama Negamax y aplica directamente la ecuación de optimalidad de Bellman — en cada nivel se invierte el signo porque lo que es bueno para mí es malo para el rival."*

2. *"Para no analizar todo el árbol — que es enorme — usamos poda alpha-beta. Es como cuando en una entrevista de trabajo ya tenés un candidato con 8/10: si el siguiente saca 3 en el primer panel, lo descartás sin escuchar más."*

3. *"La profundización iterativa nos da flexibilidad de tiempo: buscamos primero a 1 nivel, luego 2, luego 3... Si el tiempo se acaba, usamos la última respuesta completa. Siempre hay algo para jugar."*

4. *"La tabla de transposición evita recalcular tableros que ya vimos. Como un libro de problemas: si ya resolví esta posición, no la re-analizo."*

5. *"El resultado: Titan ganó el 100% de los juegos contra un jugador aleatorio en ambos colores, y en el torneo interno fue competitivo contra Hydra y Kronos."*
