#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
"""
    This a library of sped-up, but more rigid cython C functions for processing the cancer automata.
"""
__author__ = 'Daniel Rodgers-Pryor'
__copyright__ = "Copyright (c) 2014, Daniel Rodgers-Pryor\nAll rights reserved."
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = __author__
__email__ = "djrodgerspryor@gmail.com"


import cython
from cython.parallel cimport prange
cimport numpy as np
import numpy as np
from libc.stdlib cimport rand, RAND_MAX, srand, malloc, free
from libc.stdio cimport printf, fflush, stdout # For debugging inside nogil
from libc.math cimport floor
from cpython cimport bool


### Structs (for pure-c, nogil operation) ###

cdef struct pos2D:
    int x
    int y

cdef struct pos2Dlist:
    pos2D* l
    int n


### Utility Functions ###

cpdef int crand_seed(int seed):
    srand(seed)
    return 0

cpdef char crand_bool(float p) nogil:
    """p is a probability in [0,1]"""
    cdef double r = rand()
    return r/RAND_MAX < p

cpdef int crand_int(unsigned int minval, unsigned int maxval) nogil:
    """Return a roughly uniform random int in the range [minval, maxval)"""
    return minval + (rand() % (maxval - minval))  # Not actually uniform, but fine for our purposes


### Main Cython Functions ###

cpdef cstep2D(np.ndarray[np.uint8_t, ndim=2] grid, dict grid_values, dict parameters, object get_neighbors,
        bool parallelise=True, bool growth_competition=True):
    cdef:
        char[:, :] grid_view
        int i, j
        char N, C, E, D
        float k0, k1, k2, k3, k4
        pos2Dlist *neighbors
        pos2D *neighbor
        bint g_comp

    if parallelise:
        grid_view = grid
        # lots of ugly, non-extensible, explictly named arguments:
        N = grid_values['N']
        C = grid_values['C']
        E = grid_values['E']
        D = grid_values['D']
        k0 = parameters['mutation_rate']
        k1 = parameters['growth_rate']
        k2 = parameters['effector_binding_rate']
        k3 = parameters['assassination_rate']
        k4 = parameters['rebirth_rate']
        g_comp = growth_competition
        
        for i in range(grid.shape[0]):
            for j in prange(grid.shape[1], nogil=True): # Parallel, nogil loop
                grid_view[i, j] = cevaluate_cell2D_p(grid_view, i, j, N, C, E, D, k0, k1, k2, k3, k4, g_comp)
                    
    else:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                grid[i, j] = cevaluate_cell2D(grid, i, j, grid_values, parameters, get_neighbors)

cpdef char cevaluate_cell2D(np.ndarray[np.uint8_t, ndim=2] grid, int i, int j, dict grid_values, dict parameters,
                            object get_neighbors, bool growth_competition=True):
    cdef:
        char current_state = grid[i, j]
        char new_state = current_state  # Default is that nothing happens
        unsigned int cancerous_neighbors = 0
        list neighbors
        pos2Dlist *cneighbors
        char[:, :] grid_view


    if current_state == grid_values['N']: # Possible cancerous mutation
        if crand_bool(parameters['mutation_rate']):
            new_state = grid_values['C']  # Mutate to cancer

    elif current_state == grid_values['C']: # Possible cyto-toxicity and growth
        neighbors = get_neighbors(i, j)

        # Convert to pure-C:
        cneighbors = <pos2Dlist *>malloc(sizeof(pos2Dlist))
        cneighbors.l = <pos2D *>malloc(sizeof(pos2D)*4) # 4 possible neighbors
        cneighbors.n = 0
        for nx, ny in neighbors:
            cneighbors.l[cneighbors.n].x = nx
            cneighbors.l[cneighbors.n].y = ny
            cneighbors.n += 1
        grid_view = grid

        # Attempt to grow into neighboring cell:
        if cgrow2D_p(grid_view, cneighbors, parameters['growth_rate'], grid_values['C'], <int>growth_competition):
            cattempt_growth(grid, neighbors, grid_values)

        elif crand_bool(parameters['effector_binding_rate']):
            # Note: can only become cytotoxic if growth fails
            new_state = grid_values['E']  # Bind with effector to form cytotoxic complex

        # Note: The fact that cells are updated in order means that low-index cells will be systematically
        # slightly different to high-index cells. THis is probably only a minor problem with the model.

        free(cneighbors.l)
        free(cneighbors)

    elif current_state == grid_values['E']: # Possible death
        if crand_bool(parameters['assassination_rate']):
            new_state = grid_values['D']  # Destruction by immune system

    elif current_state == grid_values['D']: # Possible
        if crand_bool(parameters['rebirth_rate']):
            new_state = grid_values['N']  # Rebirth of dead cell

    return new_state

cpdef int cattempt_growth(np.ndarray[np.uint8_t, ndim=2] grid, list neighbors, dict grid_values):
    # Filter neighbors to keep only the ones which are normal:
    neighbors = [(ni, nj) for ni, nj in neighbors if grid[ni, nj] == grid_values['N']]

    if not neighbors: return 0 # Growth isn't allowed if there are no normal neighbors

    # Choose a random neighbor to grow into:
    i, j = neighbors[crand_int(0, len(neighbors))]
    grid[i, j] = grid_values['C']

    return 0


### Parallelisable, Pure-C, nogil Functions ###

cdef pos2D* cattempt_growth2D_p(char[:, :] grid, pos2Dlist* neighbors, char N) nogil:
    """
        Choose a random normal neighbor to mutate into. Return their position as a pos2D.
        Return NULL if no neighbor can be found.
    """
    cdef:
        int normal_neighbors[4]
        int i, k = 0
        pos2D *neighbor = NULL

    # Filter neighbors to keep only the ones which are normal:
    for i in range(neighbors.n):
        if grid[neighbors.l[i].x, neighbors.l[i].y] == N:
            normal_neighbors[k] = i
            k += 1

    if k == 0: return neighbor # Growth isn't allowed if there are no normal neighbors

    # Choose a random neighbor to grow into:
    neighbor = &(neighbors.l[normal_neighbors[crand_int(0, k)]])

    return neighbor

cdef pos2Dlist* cneighbors2D_p(char[:, :] grid, int i, int j) nogil:
    """
        Filter neighbors within the grid.
    """
    cdef:
        pos2D *npos_list = <pos2D *>malloc(sizeof(pos2D)*4) # 4 possible neighbors
        pos2Dlist *neighbors = <pos2Dlist *>malloc(sizeof(pos2Dlist))
    neighbors.l = npos_list
    neighbors.n = 0

    if i-1 >= 0:
        neighbors.l[neighbors.n].x = i-1
        neighbors.l[neighbors.n].y = j
        neighbors.n += 1
    
    if j-1 >= 0:
        neighbors.l[neighbors.n].x = i
        neighbors.l[neighbors.n].y = j-1
        neighbors.n += 1

    if i+1 < grid.shape[0]:
        neighbors.l[neighbors.n].x = i+1
        neighbors.l[neighbors.n].y = j
        neighbors.n += 1

    if j+1 < grid.shape[1]:
        neighbors.l[neighbors.n].x = i
        neighbors.l[neighbors.n].y = j+1
        neighbors.n += 1

    return neighbors

cdef char cgrow2D_p(char[:, :] grid, pos2Dlist* neighbors, float k1, char C, bint growth_comptetition) nogil:
    """
        Choose stochastically whether to grow or not.
    """
    cdef:
        int i
        int cn = 0
        float effective_growth_rate

    if not growth_comptetition:
        # No competition
        effective_growth_rate = k1
    else:
        # Number of cancerous neighbors:
        for i in range(neighbors.n):
            if grid[neighbors.l[i].x, neighbors.l[i].y] == C:
                cn += 1

        # Growth limited by local density of cancer cells:
        effective_growth_rate = k1 * (1 - (cn / 4.0))
        # Note: the number of neighbors above is hard-coded for 2D (as is this whole function)
        # Note: effective_growth_rate is a simplification of the model given in Qi et al.

    return crand_bool(effective_growth_rate)

cdef char cevaluate_cell2D_p(char[:, :] grid, int i, int j, char N, char C, char E, char D,
                             float mutation_r, float growth_r, float binding_r, float death_r, float rebirth_r,
                             bint g_comp) nogil:
    """
        nogil version for parallelisation. The lack of python types (dict) means that this function is much more rigid
        than the gil version.
    """
    cdef:
        char current_state = grid[i, j]
        char new_state = current_state  # Default is that nothing happens
        unsigned int cancerous_neighbors = 0

    if current_state == N: # Possible cancerous mutation
        if crand_bool(mutation_r):
            new_state = C  # Mutate to cancer

    elif current_state == C: # Possible cyto-toxicity and growth
        # Possible growth and/or effector-binding
        neighbors = cneighbors2D_p(grid, i, j)
        if cgrow2D_p(grid, neighbors, growth_r, C, g_comp):  # Mutate?
            neighbor = cattempt_growth2D_p(grid, neighbors, N)  # Choose neighbor

            if neighbor != NULL:  # If neighbor is valid
                grid[neighbor.x, neighbor.y] = C  # Grow into chosen neighbor

        elif crand_bool(binding_r):
            # Note: can only become cytotoxic if growth fails
            new_state = E  # Bind with effector to form cytotoxic complex

        # Make sure that there are no memory leaks:
        free(neighbors.l)
        free(neighbors)

    elif current_state == E: # Possible death
        if crand_bool(death_r):
            new_state = D  # Destruction by immune system

    elif current_state == D: # Possible
        if crand_bool(rebirth_r):
            new_state = N  # Rebirth of dead cell

    return new_state
