import numpy as np
import scipy as sp
import cvxpy as cvx
import matplotlib.pyplot as plt

def optRectifierSwtichingThreephase(N=2048, outputVoltageSineHarmonicNums=[1,2,4,6],
                          outputVoltageSinevals=[0,0,0,0],
                          outputVoltagecosinharmonicnums=[1,2,4,6],
                          outputVoltagecosinevals=[0,0,0,0],
                          outputVoltageDCval=0.8, gamma=10,
                          solver='ECOS'):
    """
    This function computes the optimal switching of a three phase rectifier
    which minimizes a weighted sum of the THDs of the input current and the output
    voltage.
    :param N: Time discretizations. Must be much larger than the highest harmonic number in the constraints,
    :param outputVoltageSineHarmonicNums: Sine harmonic numbers of the output voltage to be controlled
    :param outputVoltageSinevals: Desired sine harmonic values of the output voltage.
    :param outputVoltagecosinharmonicnums: Cosine harmonic numbers of the output voltage to be controlled
    :param outputVoltagecosinevals: Desired cosine harmonic values of the output voltage.
    :param outputVoltageDCval: Desired DC of the output voltage.
    :param gamma: The weight for the weighted sum of THDs of the input current and the output voltage.
    :param solver: One of the CVX solver. Default is set to ECOS.
    :return: The input currents from the three phases (which also indicates the optimal switching states) and the output voltage.
    """

    Fs = np.zeros([len(outputVoltageSineHarmonicNums), N])
    Fc = np.zeros([len(outputVoltagecosinharmonicnums), N])

    for k in range(len(outputVoltageSineHarmonicNums)):
        Fs[k, :] = np.sin(2 * np.pi * np.linspace(0, N - 1, N) / N * outputVoltageSineHarmonicNums[k])
    for k in range(len(outputVoltagecosinharmonicnums)):
        Fc[k, :] = np.cos(2 * np.pi * np.linspace(0, N - 1, N) / N * outputVoltagecosinharmonicnums[k])

    sinew12 = np.sin(2*np.pi*np.linspace(0,N-1,N)/N) - np.sin(2*np.pi/3 + 2*np.pi*np.linspace(0,N-1,N)/N)
    sinew23 = np.sin(2*np.pi/3 + 2*np.pi*np.linspace(0,N-1,N)/N) - np.sin(4*np.pi/3 + 2*np.pi*np.linspace(0,N-1,N)/N)
    sinew31 = np.sin(4*np.pi/3 + 2*np.pi*np.linspace(0,N-1,N)/N) - np.sin(2*np.pi*np.linspace(0,N-1,N)/N)

    Z12 = cvx.Variable([N,3])
    Z23 = cvx.Variable([N,3])
    Z31 = cvx.Variable([N,3])
    s = np.array([[-1],[0],[1]])
    prob = cvx.Problem(cvx.Minimize( np.ones([1,N])*((Z12+Z23)*(s**2))/N + np.ones([1,N])*((Z23+Z31)*(s**2))/N + np.ones([1,N])*((Z31+Z12)*(s**2))/N + 10*(sinew12**2)*(Z12*(s**2))/N + 10*(sinew23**2)*(Z23*(s**2))/N + gamma*(sinew31**2)*(Z31*(s**2))/N ),
                       [(Fc*(np.diag(sinew12)*(Z12*s)+np.diag(sinew23)*(Z23*s)+np.diag(sinew31)*(Z31*s))).flatten() == outputVoltagecosinevals,
                        (Fs*(np.diag(sinew12)*(Z12*s)+np.diag(sinew23)*(Z23*s)+np.diag(sinew31)*(Z31*s))).flatten() == outputVoltageSinevals,
                        sinew12*(Z12*s)/N + sinew23*(Z23*s)/N + sinew31*(Z31*s)/N == outputVoltageDCval,
                        np.ones([1,N])*(((Z12+Z23)*s)/N) == 0,
                        np.ones([1,N])*(((Z23+Z31)*s)/N) == 0,
                        np.ones([1,N])*(((Z31+Z12)*s)/N) == 0,
                        Z12 >= 0,
                        Z23 >= 0,
                        Z31 >= 0,
                        Z12*np.ones([3,1]) + Z23*np.ones([3,1]) + Z31*np.ones([3,1]) == 1])
    prob.solve(solver=solver)

    time_labels = np.linspace(0, 20, 2048)

    plt.figure()
    plt.plot(time_labels, np.matmul((Z12.value),s),linewidth=3)
    plt.plot(time_labels, np.matmul((Z23.value),s),linewidth=3)
    plt.plot(time_labels, np.matmul((Z31.value),s),linewidth=3)
    plt.title('Plot of the Switching Scheme/Normalized Current')

    plt.figure()
    plt.plot(time_labels, np.matmul(np.diag(sinew12),(np.matmul(Z12.value,s)))+ np.matmul(np.diag(sinew23),(np.matmul(Z23.value,s))) + np.matmul(np.diag(sinew31),(np.matmul(Z31.value,s))), linewidth=3)
    plt.title('Plot of the Output Voltage')

    t = np.matmul(np.diag(sinew12),(np.matmul(Z12.value,s)))+ np.matmul(np.diag(sinew23),(np.matmul(Z23.value,s))) + np.matmul(np.diag(sinew31),(np.matmul(Z31.value,s)))

    plt.figure()
    plt.plot(np.abs(np.matmul(sp.linalg.dft(N),t))[0:int(N/2+1)]/N)
    plt.title('Discrete Fourier Transform of the Output Voltage')