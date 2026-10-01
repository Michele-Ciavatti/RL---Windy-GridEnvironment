from abc import ABC, abstractmethod
from .types import BOARD_SIZE, GridState, Action
from .estimator import ActionValue
import numpy as np
from numpy.typing import NDArray

class Policy(ABC):
    def __init__(self, action_value: ActionValue, size = BOARD_SIZE):
        self.size = size
        self.action_value = action_value

    @abstractmethod
    def act(self, s: GridState) -> Action:
        pass
    
class EpsilonSoftPolicy(Policy):
    def __init__(self, 
                 action_value: ActionValue,
                 epsilon = 0.1,
                 decay = 1.0,
                 min_epsilon = 0.01,
                 size = BOARD_SIZE):
        super().__init__(action_value, size = size)
        self.epsilon, self.decay, self.min_epsilon = epsilon, decay, min_epsilon

    def get_action_probabilities(self, s: GridState) -> NDArray[np.float64]:
        """
        Returns a probability array over all actions for state s under the 
        epsilon-soft policy.
        """
        probs = np.full(len(Action), self.epsilon / len(Action))
        Q_s = self.action_value.values[s.idx]
        max_q = np.max(Q_s)
        best_action_idxs = np.flatnonzero(Q_s == max_q)
        # add the remaining (1 - epsilon) weight equally among all best actions
        probs[best_action_idxs] += (1.0 - self.epsilon) / len(best_action_idxs)
        return probs

    def act(self, s: GridState) -> Action:
        probs = self.get_action_probabilities(s)
        chosen_idx = np.random.choice(len(probs), p = probs)
        return Action.from_idx(chosen_idx)
    
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)    