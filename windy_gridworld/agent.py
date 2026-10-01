from abc import ABC, abstractmethod
from .types import GridState, Action
from .environment import GridEnvironment
from typing import Optional
from .estimator import Estimator, ActionValue, Value
from .policy import EpsilonSoftPolicy
from tqdm import trange
import numpy as np


NUM_EPISODES = 5000

class Agent(ABC):
    """
        Agent that encapsulates a EpsilonSoftPolicy and ActionValue estimator
        to run prediction and control. The specific strategy to be used is decided 
        by the subclasses.
    """
    def __init__(self,
                 env: GridEnvironment,
                 initial_action_value: Optional[ActionValue] = None,
                 alpha: Optional[float] = None,
                 epsilon: float = 0.1,
                 decay: float = 0.999):
        self.env = env
        self.action_value = (initial_action_value 
                             if initial_action_value is not None 
                             else ActionValue(alpha = alpha))
        self.policy = EpsilonSoftPolicy(self.action_value, epsilon = epsilon, decay = decay)
        self.alpha = alpha

    @abstractmethod
    def _step_evaluate(self, V: Value) -> None:
        pass

    @abstractmethod
    def _step_training(self) -> float:
        pass

    def evaluate(self,
                 initial_value: Optional[Value] = None,
                 num_episodes = NUM_EPISODES) -> Value:
        V = initial_value if initial_value is not None else Value(alpha = self.alpha)
        for _ in trange(num_episodes): self._step_evaluate(V)
        return V

    def train(self, num_episodes = NUM_EPISODES) -> list[float]:
        returns_history = []
        for _ in trange(num_episodes): returns_history.append(self._step_training())
        return returns_history

class MCAgent(Agent):
    """
        Agent that encapsulates an EpsilonSoftPolicy and ActionValue estimator
        to run Monte Carlo Control and Prediction.
    """
    def __init__(self, *args, use_first_visit = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_first_visit = use_first_visit

    @staticmethod
    def _returns(rewards, gamma) -> list[float]:
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = gamma * G + r
            returns.append(G)
        return returns[::-1]

    def _update_table(self, table: Estimator, trajectory, actions, rewards) -> None:
        returns = self._returns(rewards, table.gamma)
        seen: set[tuple[int, ...]] = set()
        for s, a, g in zip(trajectory, actions, returns):
            key = table.get_key(s, a)
            if self.use_first_visit:
                if key in seen: continue
                seen.add(key)
            table.update(key, g)

    def _generate_trajectory(self) -> tuple[list[GridState], list[Action], list[float]]:
        """
        Generates a trajectory for the environment.
        
        Returns:
            trajectory: list of states visited
            actions:    list of actions performed
            rewards:    list of rewards obtained
        """
        s = self.env.reset()
        steps = self.env.generate_episode(self.policy, s)
        trajectory, actions, rewards = zip(*steps)
        return list(trajectory), list(actions), list(rewards) 

    def _step_evaluate(self, V: Value) -> None:
        self._update_table(V, *self._generate_trajectory())

    def _step_training(self) -> float:
        trajectory, actions, rewards = self._generate_trajectory()
        self._update_table(self.action_value, trajectory, actions, rewards)
        self.policy.decay_epsilon()
        return sum(rewards)

class TD0Agent(Agent, ABC):
    """
        Agent that encapsulates an EpsilonSoftPolicy and ActionValue estimator
        to run TD(0) Control and Prediction.
        Subclasses must decided what bootstrap value to use in the update.
    """
    def __init__(self, *args, alpha: float = 0.1, **kwargs):
        assert(alpha is not None and 0 < alpha < 1)     # must use alpha here
        super().__init__(*args, alpha = alpha, **kwargs)

    @abstractmethod
    def _bootstrap_value(self, next_s: GridState) -> float:
        """Compues the bootstrap value for next_s."""
        pass

    def _step_evaluate(self, V: Value, max_steps: int = 200) -> None:
        s = self.env.reset()
        for _ in range(max_steps):
            a = self.policy.act(s)
            next_s, r, done = self.env.move(a)
            target = r + (0.0 if done else V.gamma * V.values[next_s.idx])
            V.update(V.get_key(s, a), target)
            if done: break
            s = next_s
    
    def _step_training(self, max_steps: int = 200) -> float:
        s = self.env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = self.policy.act(s)
            next_s, r, done = self.env.move(a)
            target = r + (0.0 if done 
                          else self.action_value.gamma * self._bootstrap_value(next_s))
            self.action_value.update(self.action_value.get_key(s, a), target)
            total += r
            if done: break
            s = next_s
        self.policy.decay_epsilon()
        return total

class QLearningAgent(TD0Agent):
    def _bootstrap_value(self, next_s: GridState) -> float:
        # max_a Q(next_s, a)
        return np.max(self.action_value.values[next_s.idx])
        
class ExpectedSarsaAgent(TD0Agent):
    def _bootstrap_value(self, next_s: GridState) -> float:
        # sum[pi(next_s | a) * Q(next_s, a)] over all possible actions a's
        probs = self.policy.get_action_probabilities(next_s)
        Q_s = self.action_value.values[next_s.idx]
        return np.dot(probs, Q_s)