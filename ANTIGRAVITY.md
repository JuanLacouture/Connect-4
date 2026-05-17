# Instrucciones de Proyecto Connect-4

You are working on a university project for the course "Fundamentals of Artificial Intelligence".

Before doing anything else, read these two files completely:
- `Diapositivas/reto.pdf` — this is the project rubric and requirements. You must re-read it every time you are about to implement or modify an agent.
- `CONTEXT.md` — this contains the course algorithms extracted from the slides.

---

### 1. Mandatory First Step (NON-NEGOTIABLE)

Every time you are about to implement or modify a Connect-4 agent:
1. Re-read `CONTEXT.md` in full
2. Re-read `reto.pdf` in full
3. The agent's core logic must be grounded at least 60% in algorithms from `CONTEXT.md`
4. External techniques not covered in class are allowed but cannot exceed 40% of the agent's design

---

### 2. Python Constraints (CRITICAL — Gradescope compatibility)

- NEVER use `from typing import override` anywhere in the code
- NEVER use the `@override` decorator anywhere in the code
- These break the Gradescope grader and will fail the submission silently

---

### 3. Project Structure Rules

Every agent gets its own subfolder:

```
/Connect-4/
  CONTEXT.md
  ANTIGRAVITY.md
  claude.md
  agente_v1/
    agent.py
    readme.md
    entrega.ipynb
  agente_v2/
    agent.py
    readme.md
    entrega.ipynb
```

---

### 4. Reto Requirements — Verify Before Every Submission

Before finalizing any agent, confirm all of these:

- [ ] Agent never loses to a random player
- [ ] Agent beats random player in at least 50% of games
- [ ] Agent is conceptually different from what teammates might build
- [ ] `entrega.ipynb` contains empirical experiments with plots
- [ ] At least two versions or configurations of the agent are compared
- [ ] Both colors (red and yellow) are tested
- [ ] Self-play results are included
- [ ] No `from typing import override` in any file
- [ ] No `@override` decorator in any file

---

### 5. When You Use Something NOT in CONTEXT.md

If you implement any technique not covered in the course slides, you MUST create a file called `[technique_name]_explained.md` inside that agent's folder.

Write it as if explaining to a complete beginner. Include:

- **Official name** (+ common aliases)
- **What problem does it solve?** (one sentence)
- **Simple analogy** (everyday example, zero jargon)
- **How it works, step by step** (numbered, plain language)
- **How it is used in this agent specifically**
- **Where to learn more** (paper name, Wikipedia article, or textbook chapter)

---

### 6. Agent Development Workflow (follow this order every time)

1. Re-read `CONTEXT.md` and `reto.pdf`
2. Generate a structured implementation plan
3. Wait for human approval before writing any code
4. Implement the agent
5. Run baseline experiments: vs random player (both colors), self-play
6. Generate all plots inside `entrega.ipynb`
7. If any external technique was used → create `[technique_name]_explained.md`
8. Update `readme.md` with usage instructions
