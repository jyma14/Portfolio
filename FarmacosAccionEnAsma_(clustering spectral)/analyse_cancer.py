#!/usr/bin/python
"""
    Set-up, run and plot the cancer cellular-automata described by Qi et. al. (1993)

    Usage:
        analyse_cancer.py [--noshow]

    Arguments:
        -n --noshow  Plot to file rather than interactive terminal
"""
__author__ = 'Daniel Rodgers-Pryor'
__copyright__ = "Copyright (c) 2012, Daniel Rodgers-Pryor\nAll rights reserved."
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = __author__
__email__ = "djrodgerspryor@gmail.com"
import cancer
import matplotlib
import numpy as np
from docopt import docopt


### Config ###

N = 100
c = cancer.CancerAutomata(N, growth_competition=True)
#c.parameters['mutation_rate'] = 10**-6
show = True
max_days = 400
days_step = 1

c.parameters['mutation_rate'] = 10**-5
c.parameters['growth_rate'] = 0.5
c.parameters['effector_binding_rate'] = 0.1
c.parameters['assassination_rate'] = 0.1
c.parameters['rebirth_rate'] = 0.1

# Command-line override:
arguments = docopt(__doc__, version="Daniel's Cancer Automata V1.0")
if arguments['--noshow']:
    show = False


### Setup ###

# Dependent imports:
if show:
    matplotlib.use('Qt4Agg')  # Interactive
else:
    matplotlib.use('Agg')  # Headless
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

significant_mutation = abs(c.parameters['mutation_rate']) > 1.0/(100 * N**2 * max_days)  # Is there significant mutation?
cmap = matplotlib.colors.ListedColormap(['#FFFFFF', '#D3649F', '#A3043F', '#000000'], 'custom_cmap')
fig, ax = plt.subplots(1, 1, figsize=(7, 7))


def plot():
    # Clear:
    ax.cla()

    # Plot:
    newim = ax.imshow(np.transpose(c.grid), cmap=cmap, vmin=0, vmax=c.nstates-1, interpolation='none', animated=True)

    newdivider = make_axes_locatable(ax)
    newcax = newdivider.append_axes('right', size="5%", pad=0.1)
    newcbar = fig.colorbar(newim, cax=newcax, cmap=cmap, ticks=range(0, c.nstates))
    newcax.set_yticklabels(cancer.state_names)
    newcbar.set_label('Cell Type')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return newim, newdivider, newcax, newcbar

if show:
    im, divider, cax, cbar = plot()
    title = ax.set_title('Cells (Day %d)' % 0, fontsize=17)
    fig.show()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    fig.canvas.draw()

# Make a single, central cancer cell if there is not random mutation:
c.set(int(N/2), int(N/2), 'C')
# Note: explicit int casts make intent clear (rather than relying implicitly on deprecated division)


### Simulate and plot ###
try:
    t = 0  # Time counter
    while t < max_days:
        if show:
            plt.pause(0.1)  # Process user-interaction with window

        # Run for a few steps:
        for i in xrange(days_step):
            c.step()
            t += 1
        
        # Output:
        print('day %d' % t)

        title = ax.set_title('Cells (Day %d)' % t, fontsize=17)

        if show:
            #plt.draw()
            fig.canvas.restore_region(background)
            im.set_data(np.transpose(c.grid))
            ax.draw_artist(im)
            fig.canvas.blit(ax.bbox)
        else:
            plot()
            try:
                fig.savefig('cells.png', dpi=200)
            except IOError:
                pass

        # Quit if nothing is expected to happen:
        if not significant_mutation:
            # If all cells are healthy:
            if np.all(c.grid == c.internal_values['N']):
                # The sim has reached a steady-state and nothing will happen
                print('All tissue healthy!')
                break
    print('Done!')

except KeyboardInterrupt:
    plt.close('all')
