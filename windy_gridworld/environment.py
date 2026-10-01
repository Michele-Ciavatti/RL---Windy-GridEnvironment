from .policy import Policy
from .types import BOARD_SIZE, GridState, Action
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.patches import Circle
import numpy as np



class GridEnvironment:
    """Standard GridEnvironment without wind."""
    def __init__(self, size = BOARD_SIZE):
        self.state = GridState(0, 0) # start in bottom-left corner
        self.size = size
        self.goal = GridState(size - 1, size - 1)

    def wind(self, action: Action) -> GridState:
        return GridState(0, 0)  # no wind at all

    def move(self, action: Action) -> tuple[GridState, float, bool]:
        self.state += self.wind(action) + action.delta
        hit_wall: bool = not self.state.is_valid()
        self.state = self.state.clip()
        done: bool = self.state == self.goal
        reward: float = int(done) - hit_wall * 0.1
        return self.state, reward, done

    def reset(self) -> GridState:
        self.state = GridState(0, 0)
        return self.state

    def generate_episode(self, policy: Policy, s: GridState, max_steps: int = 200):
        for _ in range(max_steps):
            a = policy.act(s)
            next_s, r, done = self.move(a)
            yield s, a, r
            if done: break
            s = next_s

    def draw(self):
        fig, ax = plt.subplots()
        # Draw grid
        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], color='black')
            ax.plot([i, i], [0, self.size], color='black')
        # Add coordinates inside the grid in the top-right corner of each cell
        for i, j in product(range(self.size), repeat = 2):
            ax.text(j + 0.95, i + 0.95, f"({j},{i})", fontsize=8, color='gray', ha='right', va='top')
        # Draw goal (circle)
        ax.add_patch(Circle((self.goal.col + 0.5, self.goal.row + 0.5), 0.3, color = 'red', label = 'Goal'))
        # Draw player (circle)
        ax.add_patch(Circle((self.state.col + 0.5, self.state.row + 0.5), 0.3, color = 'blue', label = 'Player'))
        # Set limits and grid settings
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.show()  


class StochasticWindGridEnvironment(GridEnvironment):
    """GridEnvironment with stochastic wind, i.e. wind has a probability to push the player at each step."""
    def __init__(self, size = BOARD_SIZE, prob = 0.9):
        super().__init__(size)
        self.prob = prob
    
    def wind(self, action: Action) -> GridState:
        if np.random.rand() >= self.prob: return GridState(0, 0) # no wind
        row, col = self.state
        match (row % 2, col % 2):
            case (0, 0): return GridState(0, np.random.choice([-2, -1, 1]))
            case (0, 1): return GridState(np.random.choice([-1, 1, 2]), 0)
            case (1, 0): return GridState(np.random.choice([-2, -1, 1]), 0)
            case _:      return GridState(0, np.random.choice([-1, 1, 2]))