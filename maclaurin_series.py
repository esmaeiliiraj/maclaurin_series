import math
import numpy as np
import scipy
# Symbolic Python
from sympy import *
import matplotlib.pyplot as plt

# Defining the function for Maclauring Series
def mls(X, nth=2):
    # Defining x as a symbol
    x = symbols('x')
    # Defining the function. Modify the f for your desired function.
    #f = math.e**x
    f = (x**2) - (0.7 * x**3) + math.e**x
    # Summing the values of Maclauring into g
    g = np.zeros(X.shape)
    for n in range(nth+1):
        # Taking the nth derivative of f respect to x
        dnf_dxn = diff(f, x, n)
        # lambdify method to convert the function to take values for x
        dnf_dxn = lambdify(x, dnf_dxn)
        gnx = dnf_dxn(0) * np.power(X,n) * (1/math.factorial(n))
        g += gnx

    # The real function
    f = lambdify(x, f)
    fx = f(X)
    return g, fx


# Plotting the real function vs. the nth order Maclaurin
def plot_mls(X, order=0):
    function_vals = mls(X, 0)[1]
    n_plots = order + 1

    fig, axs = plt.subplots(n_plots, 1, figsize=(6, 18))
    maclaurin_vals = []
    for i in range(order+1):
        maclaurin_vals.append(mls(X, i)[0])

    for i in range(n_plots):
        axs[i].plot(X, maclaurin_vals[i], label=f"{i} Order Maclaurin Series", linestyle='--', color='red')
        axs[i].plot(X, function_vals, label='Original function', color='black')
        axs[i].legend()
        
    plt.tight_layout()
    plt.savefig('maclaurin.png', format='png')
    plt.show();


# An example. The output plot is shown in 'maclauring.png' file.
x = np.linspace(-3, 3, 50)
plot_mls(x, 5)
