from typing import NamedTuple
from enum import StrEnum

BOARD_SIZE = 5

class GridState(NamedTuple):
    row: int
    col: int

    # it's useful to add GridStates automatically
    def __add__(self, other: object) -> 'GridState':
        if isinstance(other, GridState):
            return GridState(self.row + other.row, self.col + other.col)
        elif isinstance(other, tuple) and len(other) == 2:
            return GridState(self.row + other[0], self.col + other[1])
        raise NotImplementedError

    def is_valid(self, board_size: int = BOARD_SIZE) -> bool:
        """Checks if the state lies within the valid board grid"""
        return 0 <= self.row < board_size and 0 <= self.col < board_size

    def clip(self, board_size: int = BOARD_SIZE) -> 'GridState':
        """Clips the coordinates within [0, board_size - 1]"""
        clipped_row = max(0, min(self.row, board_size - 1))
        clipped_col = max(0, min(self.col, board_size - 1))
        return GridState(row = clipped_row, col = clipped_col)

    @property
    def idx(self) -> tuple[int, int]:
        """Returns the numerical index of the state for the V and Q table lookup."""
        return (self.row, self.col)

class Action(StrEnum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'

    @property
    def delta(self) -> GridState:
        """Returns the (row, col) shift for this action"""
        match self:
            case Action.UP:     return GridState(1, 0)
            case Action.DOWN:   return GridState(-1, 0)
            case Action.LEFT:   return GridState(0, -1)
            case Action.RIGHT:  return GridState(0, 1)
    @property
    def idx(self) -> int:
        """Returns the numerical index of the action for Q-table lookup."""
        match self:
            case Action.UP:     return 0
            case Action.DOWN:   return 1
            case Action.LEFT:   return 2
            case Action.RIGHT:  return 3
    @classmethod
    def from_idx(cls, idx: int) -> "Action":
        """Converts an integer index back to its corresponding Action enum."""
        assert(0 <= idx < 4)
        return list(cls)[idx]