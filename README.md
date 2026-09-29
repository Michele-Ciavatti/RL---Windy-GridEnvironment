# Monte Carlo Control in a Windy GridWorld

Tabular Monte Carlo prediction and control on a 5×5 GridWorld with stochastic wind,
implemented in a Jupyter notebook.

## Problem

An agent starts in the bottom-left corner (0, 0) and must reach the goal in the
top-right corner (4, 4). Each step it moves up, down, left or right, while a random
wind pushes it off course. Reaching the goal gives +1, hitting a wall gives -0.1. More details in the notebook.

## How to run
```bash
git clone https://github.com/Michele-Ciavatti/RL---Windy-GridEnvironment.git
cd RL---Windy-GridEnvironment
pip install -r requirements.txt
jupyter notebook mc_control_gridworld.ipynb
```

Requires Python 3.11 or newer

## Acknowledgements

The GridWorld setup and parts of the code are adapated from the Reinforcement Learning Laboratoris by Alberto Sinigaglia, Unipd, 2025-2026. The Control part has been made entirely by me.