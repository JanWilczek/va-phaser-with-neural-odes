import matplotlib.pyplot as plt
from matplotlib import rc
import tikzplotlib


def setup_pyplot_for_latex():
    # Use LaTeX font to save the figures in the .png format
    # (they are too big for a tikzfigure)
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams.update({'font.size': 20})

def save_tikz(filename):
    tikz_filename = append_ending(str(filename), '.tex')
    tikzplotlib.save(tikz_filename)

def append_ending(filename, ending):
    if not filename.endswith(ending):
        filename += ending
    return filename
