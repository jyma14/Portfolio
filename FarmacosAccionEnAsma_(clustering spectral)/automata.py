#!/usr/bin/python
"""
    An abstract base class for implementing cellular automata.
"""
__author__ = 'Daniel Rodgers-Pryor'
__copyright__ = "Copyright (c) 2014, Daniel Rodgers-Pryor\nAll rights reserved."
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = __author__
__email__ = "djrodgerspryor@gmail.com"

import numpy as np


class CellularAutomata(object):
    def __init__(self, n, states, dim=2, log_states=False, log_counts=True, boundary_conditions=None):
        """
            n                   The side-length of the grid.
            states              An iterator of state labels - these can be any hashable objects.
                                    there must be fewer than 2**32 states
            dim                 Grid dimension. *Must* be 2 for now
            boundary_conditions Does nothing for now
            log_states [False]  Triggers the logging of the full grid at each step in a list: myAutomata.state_log
            log_counts [True]   Triggers the logging of the frequency-counts (in a state-keyed dict) at each step
                                    in a list: myAutomata.count_log
        """
        self.nstates = len(states)

        # Set grid data-type
        if self.nstates <= (2 ** 8 - 1):  # One value is reserved for special signalling
            self.dtype = np.uint8
        elif self.nstates <= (2 ** 16 - 1):
            self.dtype = np.uint16
        elif self.nstates <= (2 ** 32 - 1):
            self.dtype = np.uint32
        else:
            raise AssertionError(
                'Too many states: %d\nWhat do you need with more than 2^32 states anyway!?' % self.nstates)

        self.grid = np.zeros([n] * dim, dtype=self.dtype)
        self.rows = self.grid.shape[0]
        self.cols = self.grid.shape[1]
        self.dim = dim

        # A mapping the given state objects to ints for internal storage
        int_states = range(len(states))
        self.internal_values = dict(zip(states, int_states))
        self.external_values = dict(zip(int_states, states))

        # Neighbor indexes:
        pos_basis = np.eye(self.dim, dtype=np.int8)
        self.neighbor_displacements = np.vstack((pos_basis, - pos_basis))  # Relative indices of neighbors

        # Log
        self.state_log = [] if log_states else None
        self.count_log = [] if log_counts else None

        #TODO: boundary conditions

    def get(self, i, j):
        """
            Get the (external) label of the cell-state (i, j).
        """
        return self.external_values.get(self.grid[i, j])

    def set(self, i, j, state):
        """
            Set a cell's state using external state labels.
        """
        self.grid[i, j] = self.internal_values[state]
        return 0

    def evaluate_cell(self, i, j):
        """Returns new state of the given cell"""
        raise NotImplementedError('You need to subclass CellularAutomata and implement this yourself.')

    def neighbors(self, i, j):
        """Return an array of the values of the neighbors of the given cell."""
        neighbor_positions = np.copy(self.neighbor_displacements)
        neighbor_positions[:, 0] += i
        neighbor_positions[:, 1] += j

        # Return neighbors which are within the grid:
        return [pos for pos in neighbor_positions if
                (pos[0] >= 0) and (pos[0] < self.rows) and
                (pos[1] >= 0) and (pos[1] < self.cols)]

    def log(self):
        """
            Log current state (grid state and or state-counts) to class-attributes.
        """
        if self.state_log:
            # Copy and log the whole grid (expensive):
            self.state_log.append(np.copy(self.grid))

        if self.log:
            counts = np.bincount(self.grid.flatten())
            states = [self.external_values.get(val) for val in np.nonzero(counts)[0]]

            # Append a new dict mapping state objects to frequency counts:
            self.count_log.append(dict(zip(states, counts)))

            # Fill empty entries with zero counts
            for val in self.external_values.values():
                self.count_log[-1][val] = self.count_log[-1].get(val, 0)

    def _step(self):
        """
            Loop through grid and update each cell.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.grid[i, j] = self.evaluate_cell(i, j)

    def step(self):
        """
            Transparently log with each step.
        """
        self._step()
        self.log()

