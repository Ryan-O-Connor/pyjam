import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import minimize
from functools import partial


###############################################################################
#
# Jiles-Atherton Equation solving functions
#
###############################################################################


def coth(x):
    # Hyperbolic cotangent (syntactic sugar)
    return 1 / np.tanh(x)


def L(x):
    # Langevin function
    if x == 0:
        return 0
    else:
        return coth(x) - 1 / x


def dLdx(x):
    # Derivative of langevin function
    if x == 0:
        return 1 / 3
    else:
        return 1 - coth(x) ** 2 + 1 / x ** 2


def dMdH(M, H, Ms, a, alpha, k, c, delta):
    # Derivative of magnetization
    He = H + alpha * M
    Man = Ms * L(He / a)
    dM = Man - M
    dMdH_num = dM / (delta * k - alpha * dM) + c * Ms / a * dLdx(He / a)
    dMdH_den =  (1 + c - c * alpha * Ms / a * dLdx(He / a))
    return dMdH_num / dMdH_den


def euler(dMdH, M0, H):
    # Euler ODE integrator for J-A equation
    M = [M0]
    for i in range(len(H) - 1):
        dH_i = H[i + 1] - H[i]
        dMdH_i = dMdH(M[i], H[i + 1], delta=np.sign(dH_i))
        M.append(M[i] + dMdH_i * dH_i)
    return M


def H_arr(Hlimit, curve_type):
    # External field intensity input
    if curve_type == 'initial':
        H = np.linspace(0, Hlimit, 500, endpoint=True)
    elif curve_type == 'loop':
        H1 = np.linspace(Hlimit, -Hlimit, 1000, endpoint=False)
        H2 = np.linspace(-Hlimit, Hlimit, 1000, endpoint=True)
        H = np.append(H1, H2)
    elif curve_type == 'full':
        H1 = np.linspace(0, Hlimit, 500, endpoint=False)
        H2 = np.linspace(Hlimit, -Hlimit, 1000, endpoint=False)
        H3 = np.linspace(-Hlimit, Hlimit, 1000, endpoint=True)
        H = np.append(H1, np.append(H2, H3))
    else:
        print('Invalid curve type')
        H = None
    return H


###############################################################################
#
# Curve-fit optimization function
#
###############################################################################


def read_BHdata(filename):
    Hdata, Bdata = [], []
    with open(filename, 'rt') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i != 0:
                Hdata.append(float(row[0]))
                Bdata.append(float(row[1]))
    return Hdata, Bdata


def fit_error(Hdata, Bdata, Ms, a, alpha, k, c):
    # Computes sum-of-squares error between test data and J-A curve defined by
    # given model parameters
    pass



###############################################################################
#
# Test case functions
#
###############################################################################


def test1():
    # Initializate parameters
    Hmin = 0
    Hmax = 5000
    Ms = 1.6e6
    a = 1100
    alpha = 1.6e-3
    k = 400
    c = 0.2
    delta = 1
    # Setup and solve model
    H = np.linspace(Hmin, Hmax, 1000)
    M0 = 0
    dydt = partial(dMdH, Ms=Ms, a=a, alpha=alpha, k=k, c=c, delta=delta)
    M = np.array(euler(dydt, M0, H))
    plt.plot(H, M / Ms)
    plt.show()


def test2():
    # Initializate parameters
    Hmax = 6000
    Ms = 1.6e6
    a = 1100
    alpha = 1.6e-3
    k = 400
    c = 0.2
    # Setup and solve model
    H = H_arr(Hmax, curve_type='full')
    M0 = 0
    dydt = partial(dMdH, Ms=Ms, a=a, alpha=alpha, k=k, c=c)
    M = np.array(euler(dydt, M0, H))
    plt.plot(H, M / Ms)
    plt.xlim([-Hmax-1000, Hmax+1000])
    plt.ylim([-1, 1])
    plt.title('Fe-C 0.06 wt%\nInitial Magnetization and Hysteresis Loop')
    plt.xlabel('H [A/m]')
    plt.ylabel('M/Ms')
    plt.grid()
    plt.show()


def test3():
    # Import data
    filename = 'init.csv'
    Hdata, Bdata = read_BHdata(filename)
    # Create model
    mu0 = 4*np.pi*10**-7
    Hmax = max(Hdata)
    Ms = 1.5e6
    a = 30
    alpha = 1e-5
    k = 30
    c = 0.1
    H = H_arr(Hmax, curve_type='initial')
    M0 = 0
    dydt = partial(dMdH, Ms=Ms, a=a, alpha=alpha, k=k, c=c)
    M = np.array(euler(dydt, M0, H))
    B = mu0*(M + H)
    # Plot data
    plt.plot(Hdata, Bdata)
    plt.plot(H, B)
    plt.legend(['Data', 'Model'])
    plt.xlim([0, 1.1*Hmax])
    plt.ylim()
    plt.title('Hiperco 50\nInitial Magnetization')
    plt.xlabel('H [A/m]')
    plt.ylabel('B [T]')
    plt.grid()
    plt.show()





if __name__ == '__main__':
    test3()

