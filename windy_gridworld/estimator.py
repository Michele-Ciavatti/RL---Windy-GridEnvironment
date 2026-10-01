from abc import ABC, abstractmethod
from .types import BOARD_SIZE, GridState, Action
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path


class Estimator(ABC):
    """
        Tabular estimator: this is the basis for value and 
        action-value functions in tabular form.
    """
    def __init__(
            self,
            shape: tuple[int, ...],
            size = BOARD_SIZE,
            gamma = 0.9,
            alpha: Optional[float] = None,  
    ):
        self.values, self.counts = np.zeros(shape), np.zeros(shape, dtype = int)
        self.gamma, self.alpha, self.size = gamma, alpha, size
        self._step = (self._sample_average_update if alpha is None
                      else self._constant_alpha_update)

    def _sample_average_update(self, key: tuple[int, ...], target: float) -> None:
            self.counts[key] += 1
            self.values[key] += (target - self.values[key]) / self.counts[key]
    
    def _constant_alpha_update(self, key: tuple[int, ...], target: float) -> None:
        self.counts[key] += 1
        self.values[key] += (target - self.values[key]) * self.alpha

    def update(self, key: tuple[int, ...], target: float) -> None:
        self._step(key, target)
    
    @staticmethod
    @abstractmethod
    def get_key(s: GridState, a: Action) -> tuple[int, ...]:
        """Table index for this (state, action); a state-only table ignores a."""
        pass

class Value(Estimator):
    """State-value table V(s), indexed by [row, col]."""
    def __init__(self, size = BOARD_SIZE, **kwargs):
        super().__init__(shape = (size, size), size = size, **kwargs)

    @staticmethod
    def get_key(s: GridState, a: Action): return s.idx

    def draw(self, save_to: Optional[str] = None):
        fig, ax = plt.subplots()
        # Create a color map based on the value grid
        cax = ax.imshow(self.values, cmap='coolwarm', origin='lower', 
                        extent=(0, self.size, 0, self.size), vmin = -0.5, vmax = 1.0)
        # Add a color bar to indicate value levels
        fig.colorbar(cax)
        # Draw grid lines
        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], color='black', lw=0.5)
            ax.plot([i, i], [0, self.size], color='black', lw=0.5)
        # Add coordinates and values inside the grid in the top-right corner of each cell
        for i, j in product(range(self.size), repeat = 2):
            val = self.values[i, j] # i = row, j = col
            ax.text(j + 0.9, i + 0.9, f"({j},{i})", fontsize=8, color='gray', ha='right', va='top')
            # Optionally display the value itself at the center of the cell
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", fontsize=8, color='black', ha='center', va='center')
        # Set limits and grid settings
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        if save_to is not None:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.show()

class ActionValue(Estimator):
    """Action-value table Q(s, a), indexed by [row, col, action]."""
    def __init__(self, size = BOARD_SIZE, **kwargs):
        super().__init__(shape = (size, size, len(Action)), size = size, **kwargs)

    @staticmethod
    def get_key(s: GridState, a: Action): return (*s.idx, a.idx)