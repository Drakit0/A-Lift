# A-Lift

Reinforcement learning agents that navigate a simulated warehouse, pick up objects and deliver them, written for a course project at Universidad Pontificia Comillas ICAI (Aprendizaje por refuerzo, 2025-2026).

## What it does

The agent moves in a continuous 10 m x 10 m map with three fixed shelves, in steps of 0.25 m. There are four custom gym-style environments (Gymnasium API), all in `src/`:

- `navigation_environment.py`: reach a target zone without hitting walls or shelves (used for the first part of the course work).
- `warehouse_environment.py`, three variants:
  1. Environment 1: pick up any of the objects, which sit at fixed positions on the shelves.
  2. Environment 2: pick up an object and drop it in the delivery zone. Dropping it elsewhere or colliding ends the episode as a failure.
  3. Environment 3: same task as 2, but the object positions on the shelves change every episode.

`A-Lift.mp4` is a recording of a trained agent. `environment.png` shows the navigation map.

## How it works

The state is encoded with tile coding (`src/tiles3.py`, `src/representation.py`): 10 x 10 tiles over the map and 8 overlapping tilings. In environments 2 and 3 the representation also includes direction and distance to the current goal (the object or the delivery zone). Q(s, a) is linear in the active tiles, with one weight row per action, and only the active weights are updated.

Agents implemented:

| File | Algorithm |
|---|---|
| `src/qlearning.py` | Q-learning with tile coding |
| `src/sarsa.py` | SARSA(0) with tile coding |
| `src/sarsa_lambda.py` | SARSA(lambda) with tile coding and eligibility traces |
| `src/dqn.py` | DQN (PyTorch): replay buffer, optional prioritized replay, target network, n-step returns |

Exploration is epsilon-greedy with exponential decay. The decay start and end, the learning rate, gamma and lambda are set per environment at the bottom of each script.

## Results

The numbers below are from the project report (`analysis/`, in Spanish), section "Comparativa de enfoques".

| Model | Success env 1 | Success env 2 | Success env 3 | Episodes |
|---|---|---|---|---|
| SARSA(lambda) + tiles | 95%+ | 90%+ | 85%+ | 3k to 30k |
| DQN | 20% | not reported | not reported | 50k+ |

The report also says that a linear model on the 11 raw observation variables reached under 6% success, and that DQN took 10 to 20 times longer to train than SARSA(lambda) with tiles. The report concludes that DQN did not learn a working policy in the time available, so the tile coding agent was used for the three environments.

For the navigation task, the report states that Q-learning and SARSA both reach very high success rates, with Q-learning converging slightly faster. Training curves are in `images/`. The trained agents are in `models/`; the file names encode environment, number of episodes, learning rate, initial epsilon, evaluation success rate and average return.

## How to run

Python 3 with the packages in `requirements.txt` (the file was exported from a pip environment and includes a CUDA build of torch, so edit that line if you have no GPU).

```
pip install -r requirements.txt
```

Run everything from the repository root, since scripts read and write `models/` and `images/` with relative paths.

Train and evaluate SARSA(lambda). The environment is selected by the `env_variant` variable ("1", "2" or "3") in the `__main__` block:

```
python src/sarsa_lambda.py
```

The same holds for DQN (`env_str` in `src/dqn.py`):

```
python src/dqn.py
```

Navigation task with Q-learning or SARSA:

```
python src/qlearning.py
python src/sarsa.py
```

Watch a saved agent in the pygame window. The script lists the files in `models/` and asks which one to run:

```
python src/visualizer.py
```

## Structure

```
src/        environments, tile coding, agents, visualizer
models/     trained agents (pickle)
images/     training curves, value heatmaps, evaluation plots
analysis/   project report (LaTeX source and PDF)
```

## Authors

- Pablo Tuñón Laguna
- Lydia Ruiz Martínez
- Alberto Velasco Rodríguez

Course project for Aprendizaje por refuerzo, Universidad Pontificia Comillas ICAI, academic year 2025-2026.
