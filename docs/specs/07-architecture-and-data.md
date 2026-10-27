# Architecture, Data Management, Agent Budgets, and Replay

## 8. Architecture & Data Management

### 8.1 Data Model

Each match run is persisted in SQLite or Parquet.

Minimal model:

```sql
MATCHES ( id, seed, bot_version, opponent_hash, hands_played, ev_bb_100, ci_low, ci_high )

HANDS ( id, match_id, street, hero_cards, villain_cards, board, actions_json, pot, ev_change )
```

For scientific reproducibility, the following are additionally stored:

```text
experiment_id
game_config_hash
abstraction_version
engine_version
python_version
rng_version
git_commit
starting_stack
small_blind
big_blind
rake
population_id
population_seed
opponent_parameters
bot_configuration
statistical_method
sample_size
precision_target
```

The complete experiment configuration is stored as an immutable configuration and hashed.

The configuration hash is computed as **SHA-256 over a canonical JSON representation** of the frozen configuration. Keys are sorted and values are normalised deterministically so that equivalent configurations produce the same hash across platforms.

The project uses **Poetry** for Python dependency management. `pyproject.toml` and `poetry.lock` are stored with the project and the relevant lockfile state is included in experiment metadata.

For SQLite persistence, the schema includes a `schema_version` table. Schema changes are versioned through explicit migrations; **Alembic** is used for schema migration once the SQLite schema is established. A database is never modified by an implicit or undocumented schema change.

The reference benchmark uses the developer's **physical local machine**, not rented cloud instances. Hardware metadata records a platform-independent fingerprint derived from CPU model, core count, and architecture. The OS identifier is the output of `platform.platform()`. Exact machine identifiers that are unnecessary for reproducibility are not required.

This allows a result to be reconstructed not only by a seed, but by the entire relevant experiment environment.

---

### 8.2 Action Validation, Illegal Actions, and Agent Budgets

Every action emitted by an agent is validated against the current legal-action set before it reaches the game engine.

**Illegal-action policy:** an illegal action is a deterministic agent error. The engine rejects it and records the complete state, requested action, legal action set, and agent version. The benchmark does **not** silently substitute the nearest legal action, because doing so would change the agent's requested behaviour and make results difficult to interpret. Development runs may raise an exception; benchmark runs terminate the affected run with a defined failure status. The frozen benchmark policy is `illegal_action_handling = exclude_run`: the failed run is excluded from the inferential dataset rather than counted as a lost pot or silently repaired. It is never automatically replaced with a new seed. The failure is retained in the experiment log and reported separately. The planned power/precision analysis is based on the number of valid independent statistical units; if failures reduce the valid sample below the pre-specified target, the benchmark is incomplete and additional valid runs are scheduled before final analysis.

Every non-trivial agent also declares an execution budget. At minimum:

```text
agent_simulation_budget
agent_time_budget_ms   (optional safety limit)
timeout_policy
```

If an agent exceeds `agent_time_budget_ms`, the engine executes the `timeout_policy`. The default benchmark policy is `fold`: the hand is terminated and the agent forfeits the pot. Other policies (e.g., `check_if_legal_else_fold`) must be explicitly declared and versioned.

For search agents, the primary scientific budget is the number of simulations per decision. A time limit may be used as a technical safety bound but is not silently substituted for the simulation budget. Budget, timeout, and failure behaviour are stored with the agent configuration.

### 8.3 Performance Target

The Python engine is optimised to a speed of at least 10,000 simulated simple engine hands per second in a single‑thread benchmark **as an engineering target**.

The actual performance is measured on defined hardware.

**Reference hardware:** the physical local development/benchmark machine used for the project. The exact CPU model, core count, architecture, RAM, Python version, and OS identifier are recorded with each benchmark. No cloud instance is required for the scientific baseline; parallelisation may use the local machine's cores when needed.

For more complex agents such as MCTS or NFSP, a significantly lower simulation rate is to be expected.

The benchmark speed is not a scientific target metric for agent quality.

Large benchmark matrices can be parallelised over multiple cores or a cluster if needed.

Parallelisation is performed at the match level: each match (defined by agent, opponent, seed) is executed as an independent job in a job queue. Results are merged into a central database. Parallelisation is purely for runtime optimisation and does not affect the statistical evaluation.

---

### 8.4 Debugging & Replay Tool

A mandatory CLI module serves to analyse large losing pots:

```bash
poker inspect \
  --match-id 42 \
  --filter "hero_loss > 50 AND street == 'river'"
```

This allows critical hands to be examined in detail.

Additional filters are provided:

```bash
poker inspect --match-id 42 --street river
poker inspect --match-id 42 --action call
poker inspect --match-id 42 --min-pot 50
```

This prevents the developer from relying solely on aggregated JSON or database data.

The replay mechanism shall display the complete decision chain. Internally, **true game state** and **agent-visible observation** are stored separately so replay data cannot accidentally leak hidden information into a policy:

- Hole cards
- Board
- Pot
- Stack
- Position
- Action history
- Available actions
- Bot evaluation
- Chosen action
- Resulting profit
- Estimated decision EV, if applicable

---
