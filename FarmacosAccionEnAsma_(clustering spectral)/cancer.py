#!/usr/bin/python
"""
    A subclass of CellularAutomata which implements the cancer-growth model described by Qi et al. (1993)
"""
__author__ = 'Daniel Rodgers-Pryor'
__copyright__ = "Copyright (c) 2014, Daniel Rodgers-Pryor\nAll rights reserved."
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = __author__
__email__ = "djrodgerspryor@gmail.com"


# Import cython is it's available:
try:
    import pyximport

    pyximport.install()
    import cancer_utils  # cython import should be auto-compiled
except ImportError:
    cython_available = False
else:
    cython_available = True

from automata import CellularAutomata
import random
import numpy as np
from functools import partial
from warnings import warn


def rand_bool(p):
    """p is a probability in [0,1]"""
    return random.random() < p


# These strings will be used to label the possible cell-states
state_names = ('N', 'C', 'E', 'D')  # Normal, Cancerous, Effector (cytotoxic complex), Dead


class CancerAutomata(CellularAutomata):
    def __init__(self, *args, **kwargs):
        """
            Arguments:
                n                           The side-length of the grid.
                cython_optimise [True]      Cython-optimised functions if available.
                parallelise [True]          If cython optimise is True, switches to *much* faster nogil parallelised code.
                growth_competition [True]   Reduce growth rate as: k1' = k1 * (1 - C_neighbors_count/4)
        """
        
        # Transition rate parameters. These can be redefined locally for each automata instance:
        self.parameters = {
            'mutation_rate': 0.00,  # N->C
            'growth_rate': 0.4,  # C->2C
            'effector_binding_rate': 0.1,  # C->E
            'assassination_rate': 0.35,  # E->D
            'rebirth_rate': 0.35  # D->N
        }

        # Modify arguments for passing to base class:
        cython_optimise = kwargs.pop('cython_optimise', True)
        parallelise = kwargs.pop('parallelise', True)  # Only applies if cython_optimise if True
        self.growth_competition = kwargs.pop('growth_competition', True)
        kwargs['states'] = state_names

        super(CancerAutomata, self).__init__(*args, **kwargs)  # Base init

        if cython_optimise:
            if cython_available and (self.dim == 2) and (self.dtype == np.uint8):
                # Switch to optimised Cython implementations by overriding default methods
                self._step = partial(cancer_utils.cstep2D, grid=self.grid, grid_values=self.internal_values,
                                     parameters=self.parameters, get_neighbors=self.neighbors,
                                     parallelise=parallelise, growth_competition=self.growth_competition)

                self.evaluate_cell = partial(cancer_utils.cevaluate_cell2D, grid=self.grid,
                                             grid_values=self.internal_values, parameters=self.parameters,
                                             get_neighbors=self.neighbors, growth_competition=self.growth_competition)

                self.attempt_growth = partial(cancer_utils.cevaluate_cell2D, grid=self.grid,
                                              grid_values=self.internal_values)

                # Initialise the C random module:
                cancer_utils.crand_seed(random.randint(0, 2 ** 16))
            else:
                # Warn user about lack of cython optimisation
                msg = ["Cython unavailable, reverting to python. This will be much slower!"]
                msg.append("\tCython import and compile: %s" % ("OK" if cython_available else
                                                                "ERROR"))
                msg.append(
                    "\tGrid dimension %d: %s" % (self.dim,
                                                 "OK" if self.dim == 2 else
                                                 "ERROR: dim must be 2 for cython"))
                msg.append(
                    "\tGrid dtype %s: %s" % (str(self.dtype),
                                             "OK" if self.dtype == np.uint8 else
                                             "ERROR: dtype must be uint8 (<=> fewer than 255 possible cell states)"))

                warn("\n".join(msg), RuntimeWarning)

        # Convenient translations of state names to internal representations:
        for n in state_names:
            setattr(self, n, self.internal_values[n])

    def effective_growth_rate(self, neighbors):
        rate = self.parameters['growth_rate']  # Base growth rate
        if self.growth_competition:
            # Limit by local density of cancer cells:
            neighboring_C_cells = sum(1 for ni, nj in neighbors if self.grid[ni, nj] == self.C)
            rate *= (1 - (neighboring_C_cells / (self.dim * 2.0)))
            # Note: effective_growth_rate is a simplification of the model given in Qi et al.

        return rate

    def evaluate_cell(self, i, j):
        """
            The logic for updating a cell.
        """
        current_state = self.grid[i, j]
        new_state = current_state  # Default is that nothing happens

        # Possible cancerous mutation
        if current_state == self.N:
            if rand_bool(self.parameters['mutation_rate']):
                new_state = self.C  # Mutate to cancer

        # Possible cyto-toxicity and growth
        elif current_state == self.C:
            neighbors = self.neighbors(i, j)
            # Possible growth into neighboring cell:
            if rand_bool(self.effective_growth_rate(neighbors)):
                self.attempt_growth(neighbors)
                # Note: The fact that cells are updated in order means that low-index cells will be systematically
                #   slightly different to high-index cells. THis is probably only a minor problem with the model.
            
            elif rand_bool(self.parameters['effector_binding_rate']):  # Possible cyto-toxicity
                # Note: can only become cytotoxic if growth fails
                new_state = self.E  # Bind with effector to form cytotoxic complex

        # Possible death
        elif current_state == self.E:
            if rand_bool(self.parameters['assassination_rate']):
                new_state = self.D  # Destruction by immune system

        # Possible rebirth
        elif current_state == self.D:
            if rand_bool(self.parameters['rebirth_rate']):
                new_state = self.N  # Rebirth of dead cell

        return new_state

    def attempt_growth(self, neighbors):
        """
            Choose a random normal neighbor and expand into it. Do nothing if no normal neighbors are available.
        """
        # Filter neighbors to keep only the ones which are normal:
        neighbors = [(ni, nj) for ni, nj in neighbors if self.grid[ni, nj] == self.N]

        if not neighbors:
            return  # Growth isn't allowed if there are no normal neighbors

        # Choose a random neighbor to grow into:
        i, j = random.choice(neighbors)
        self.grid[i, j] = self.C




