import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root_scalar
from functools import partial



def langevin(x):
    # Computes langevin function:
    #   L(x) = coth(x) - 1/x
    return 1/np.tanh(x) - 1/x


def anhysteretic(M, H, Ms, a, alpha):
    # Computes anhysteretic function for M
    #   y == M/Ms - L(He/a)
    He = H + alpha*M
    y = M/Ms - langevin(He/a)
    return y

def bracket_anhysteretic(H, Ms, a, alpha):
    # Find bracket for solving for anhysteretic curve at H
    Man = partial(anhysteretic, H=H, Ms=Ms, a=a, alpha=alpha)
    N = 20
    limits = [-1 + 0.1*x for x in range(N+1)]
    for i in range(N):
        Mlo = Man(limits[i]*Ms)
        Mhi = Man(limits[i+1]*Ms)
        if Mlo * Mhi < 0:
            return [limits[i]*Ms, limits[i+1]*Ms]
    return None


def solve_anhysteretic(H, Ms, a, alpha):
    if H == 0 or a == 0 or Ms == 0:
        M = 0
    else:
        bracket = bracket_anhysteretic(H, Ms, a, alpha)
        if bracket is not None:
            Man = partial(anhysteretic, H=H, Ms=Ms, a=a, alpha=alpha)
            sol = root_scalar(Man, bracket=bracket, method='bisect')
            M = sol.root
            if not sol.converged:
                print('Root finding did not converge for H={}'.format(H))
        else:
            print('Could not find bracket for H={}'.format(H))
            M = 0
    return M


def plot_anhysteretic(axes, Hmin, Hmax, Ms, a, alpha):
    H = np.linspace(Hmin, Hmax, 100)
    M = [solve_anhysteretic(h, Ms, a, alpha) for h in H]
    M = np.array(M)
    axes.plot(H, M/Ms)


def anhysteretic_test1():
    Ms = 1.6e6
    a = 1100
    alphas = [0, 0.8e-3, 1.6e-3]
    Hmin = -5000
    Hmax = 5000
    fig, ax = plt.subplots()
    for alpha in alphas:
        plot_anhysteretic(ax, Hmin, Hmax, Ms, a, alpha)
    ax.set_xlim([Hmin, Hmax])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('H (A/m)')
    ax.set_ylabel('M/Ms')
    ax.set_title('Anhysteretic Magnetization\nMs=1.6e6 A/m, a=1100 A/m')
    ax.legend(['alpha=0.0e-3', 'alpha=0.8e-3', 'alpha=1.6e-3'])
    plt.grid()
    plt.show()
    

def anhysteretic_test2():
    Ms = 1.6e6
    a_s = [1100, 2200, 3300]
    alpha = 1.6e-3
    Hmin = -5000
    Hmax = 5000
    fig, ax = plt.subplots()
    for a in a_s:
        plot_anhysteretic(ax, Hmin, Hmax, Ms, a, alpha)
    ax.set_xlim([Hmin, Hmax])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('H (A/m)')
    ax.set_ylabel('M/Ms')
    ax.set_title('Anhysteretic Magnetization\nMs=1.6e6 A/m, alpha=1.6e-3')
    ax.legend(['a=1100 A/m', 'a=2200 A/m', 'a=3300 A/m'])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    anhysteretic_test2()
