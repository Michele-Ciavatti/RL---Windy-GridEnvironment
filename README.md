# Monte Carlo and TD(0) Control in a Windy GridWorld

Tabular **Monte Carlo** and **TD(0)** prediction and control on a 5×5 GridWorld with stochastic wind, implemented from scratch in a single Jupyter notebook (NumPy + Matplotlib only, no RL libraries).

![Convergence comparison](figures/convergence_comparison.png)

*Smoothed episodic return during the early episodes of training: TD(0) control (Q-learning) gets to good performance faster than Monte Carlo control.*

## Problem

An agent starts in the bottom-left corner `(0, 0)` and must reach the goal in the top-right corner `(4, 4)`. At each step it picks one of four actions (up, down, left, right) and moves one cell, while a random wind may push it off course.

| Element | Details |
|---|---|
| **Wind** | With probability 0.9 per step, a push whose direction and magnitude (1 or 2 cells) depend on the parity of the current cell's row and column. Otherwise no wind. |
| **Walls** | The agent cannot leave the grid: it is clipped back inside and penalised. |
| **Rewards** | +1 for reaching the goal, −0.1 for hitting a wall, 0 otherwise |
| **Episodes** | End at the goal or after a fixed amount of steps |
| **Discount** | γ = 0.9 |

## What's implemented

- **Environment**: `GridEnvironment` (no wind) and `StochasticWindGridEnvironment`
- **Estimators**: tabular state-value `Value` and action-value `ActionValue`, with either sample-average or constant-α updates
- **Policy**: ε-soft policy with optional ε decay
- **Monte Carlo** (`MCAgent`): first-visit or every-visit prediction and control (Sutton & Barto §5.1, §5.4)
- **TD(0)** prediction and control (Sutton & Barto §6.1, §6.5), with three bootstrap targets:
  - `QLearningAgent`: off-policy, bootstraps on `max_a Q(s', a)`
  - `SarsaAgent`: on-policy, bootstraps on `Q(s', a')` with `a'` sampled from the policy
  - `ExpectedSarsaAgent`: bootstraps on the expectation of `Q(s', ·)` under the policy

## Results

Training for 5000 episodes per agent, with ε-decay. Mean and standard deviation of the episodic return over training:

| Agent | Mean return | Std dev |
|---|---|---|
| Monte Carlo | 0.82 | 0.18 |
| Q-learning | 0.79 | 0.16 |
| Expected SARSA | 0.80 | 0.13 |

Takeaways:

- **TD(0) learns faster early on.** It updates at every step, while MC has to wait for the end of each episode.
- **MC has higher variance**, since it learns from full sampled returns.
- **Expected SARSA has the lowest variance**, because averaging over next actions removes the noise of sampling a single one.

Results come from a single run and will vary with the random seed.

## How to run

```bash
git clone https://github.com/Michele-Ciavatti/RL---Windy-GridEnvironment.git
cd RL---Windy-GridEnvironment
pip install -r requirements.txt
jupyter notebook Control_GridWorld.ipynb
```

Requires **Python 3.11 or newer** (the code uses `StrEnum` and `match` statements).

Run the cells top to bottom. The notebook saves value-function heatmaps and the convergence plot to `figures/`.

## Repository structure

```
.
├── Control_GridWorld.ipynb   # environment, agents, experiments
├── figures/                  # generated plots
├── requirements.txt
├── LICENSE
└── README.md
```

## Possible extensions

- Compare constant-α and sample-average updates for MC
- Plot the learned greedy policy as arrows on the grid
- Average results over many seeds and add confidence bands

## Acknowledgements

The GridWorld setup and parts of the code are adapted from the Reinforcement Learning Laboratories by Alberto Sinigaglia, Unipd, 2025–2026.

Algorithm descriptions follow R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018).
