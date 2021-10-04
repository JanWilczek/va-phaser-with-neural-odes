import matplotlib.pyplot as plt
from matplotlib import rc
import tikzplotlib


def setup_pyplot_for_latex():
    # Use LaTeX font to save the figures in the .png format
    # (they are too big for a tikzfigure)
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams.update({'font.size': 12})

def save_tikz(filename):
    tikz_filename = append_ending(str(filename), '.tex')
    tikzplotlib.save(tikz_filename)

def save_png(filename):
    tikz_filename = append_ending(str(filename), '.png')
    plt.savefig(tikz_filename, bbox_inches='tight', dpi=400)

def append_ending(filename, ending):
    if not filename.endswith(ending):
        filename += ending
    return filename
