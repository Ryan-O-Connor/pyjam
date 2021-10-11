import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def coth(x):
    # Hyperbolic cotangent (syntactic sugar)
    return 1/np.tanh(x)


def L(x):
    # Langevin function
    if x == 0:
        return 0
    else:
        return coth(x) - 1/x


def dLdx(x):
    # Derivative of langevin function
    if x == 0:
        return 1/3
    else:
        return 1 - coth(x)**2 + 1/x**2


def dMdH(M, H, Ms, a, alpha, k, c, delta):
    # Derivative of magnetization
    He = H + alpha*M
    Man = Ms*L(He/a)
    dM = Man - M
    dMdH_num = dM / (delta*k - alpha*dM) + c*Ms/a*dLdx(He/a)
    dMdH_den = (1+c)*(1 - c*alpha*Ms/a*dLdx(He/a))
    return dMdH_num / dMdH_den


def euler(dMdH, M0, H):
    # Euler ODE integrator
    M = [M0]
    for i in range(len(H)-1):
        dH_i = H[i+1] - H[i]
        dMdH_i = dMdH(M[i], H[i+1], delta=np.sign(dH_i))
        M.append(M[i] + dMdH_i*dH_i)
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
    return H


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
    plt.plot(H, M/Ms)
    plt.show()



def test2():
    # Initializate parameters
    Hmax = 5000
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
    plt.plot(H, M/Ms)
    plt.show()

    
test2()
