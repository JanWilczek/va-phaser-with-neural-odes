import numpy as np
import matplotlib.pyplot as plt
from common import setup_pyplot_for_latex, save_tikz
from diode_ode_numerical import DiodeParameters
from visualize_ode import plot_ode


def diode_equation_rhs(v_out, v_in, d):
    return (v_in - v_out) * d.c1 - d.c2 * np.sinh(v_out / d.v_t)

def main():
    setup_pyplot_for_latex()
    
    step = 1e-3
    amplitude = 1
    value_range = np.arange(-amplitude, amplitude, step)
    derivative_magnitude = np.zeros((value_range.shape[0], value_range.shape[0]))

    d = DiodeParameters()

    for i in range(value_range.shape[0]):
        for j in range(value_range.shape[0]):
            derivative_magnitude[i, j] = 10*np.log10(1e-6 + np.abs(diode_equation_rhs(value_range[i], 20 * value_range[j], d)))

    plot_ode(derivative_magnitude)
    plt.savefig('analytical_derivative.png', bbox_inches='tight', dpi=300)
    save_tikz('analytical_derivative')

if __name__ == '__main__':
    main()
