#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predicting reddening and variation through chi_sq minimization of diff. between obs. and pred. mag
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import sys
import silence_tensorflow.auto
import tensorflow as tf
from scipy.optimize import minimize


def read_out(path):
    # read out the test data set
    with h5py.File(path, 'r') as f:
        d = f['data'][:]
        r = f['r_fit'][:]
        if 'mock' in path:
            R = f['R'][:]
            dr = f['dr_fit'][:]
            dR = f['dR'][:]
        else:
            R = []
            dr = []
            dR = []

    return d, r, R, dr, dR


def save(path, r, R, dr, dR, l, m , C, chi_sq):
    # save the mock data run
    with h5py.File(path, 'w') as f:
        f.create_dataset('/E', data=r, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dE', data=dr, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/R', data=R, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dR', data=dR, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/l', data=l, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dm', data=m, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/C', data=C, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/chi', data=chi_sq, chunks=True, compression='gzip', compression_opts=3)

def guess_dR(BR, B, B_inv):
    # read out the mean wavelengths of the 13 filters we use
    with open('mean_wavelengths.json', 'r') as f:
        mean = json.load(f)

    x = np.array(mean)
    R = np.einsum('ij,j->i',B_inv,BR)

    # calculate the offset of our linear dR-model to be perpendicular to R
    b = np.sum(x*R)/np.sum(R)
    y = x[:] - b[None]

    # norm it to 1 and transform it to mag/color space
    BdR = np.einsum('ij,j->i',B,y)
    BdR /= np.dot(BdR,BdR)**0.5

    return BdR


def calculate_pre_loop(dm, C):
    # calculate some terms of chi_sq before which don't change over the iterations
    mcm = np.einsum('ni,nij,nj->n',dm,C,dm)
    mc = np.einsum('ni,nij->nj',dm,C)
    cm = np.einsum('nij,nj->ni',C,dm)
    mcm_sum = np.sum(mcm)

    return mcm_sum, mc, cm


def chi_sq(dE, dR, E, R, C, dm):
    # calculate (dm + E*R + dE*dR) for each star
    bracket = dm[:,:] + E[:,None]*R[None,:] + dE[:,None]*dR[None,:]
    # calculate chi_square for each star and return it as an array (not summed)
    cho = np.einsum('ni,nij->nj',bracket,C)
    chi = np.einsum('ni,ni->n',cho,bracket)

    return chi


def calculate_E(R, dE, dR, C, m):
    # calculate the constant term with dm+reddening (or variation)
    dm = m[:,:] + dE[:,None] * dR[None,:]
    # calculate some terms of the chi_square with just E as variable, which can be reddening or var.
    RCdm = np.einsum('i,nij,nj->n',R,C,dm)
    dmCR = np.einsum('ni,nij,j->n',dm,C,R)
    RCR = np.einsum('i,nij,j->n',R,C,R)
    # calculate the root of the chi_sq function of E, which can be reddening or variation
    E = -0.5*(RCdm + dmCR)/RCR

    return E


def constraint_ortho(R, dR, l1 = None, B_inv = None):
    # transform R and dR into the right space where we want them to be orthogonal
    if B_inv is not None:
        R = np.einsum('ij,j->i',B_inv,R)
        dR = np.einsum('ij,j->i',B_inv,dR)
    # constraint that is  f(R,dR) - b
    con = (np.dot(R,dR))

    if l1 is not None:
        con *= l1

    return con


def constraint_unity(R, l2 = None):
    # constraint that is f(R) - b
    con = (np.dot(R,R) - 1)

    if l2 is not None:
        con *= l2

    return con


def lagrangian(R2, l1, l2, n, rho, B_inv, R1, EEC, EmC, ECm, ECR, ERC, RCm, mCR, RCR):
    # calculate the terms that depend on the varibale we minimize for in chi_sq
    rCr = np.einsum('i,ij,j',R2,EEC,R2)
    mCr = np.dot(EmC,R2)
    rCm = np.dot(ECm,R2)
    rCR = np.dot(ERC,R2)
    RCr = np.dot(ECR,R2)

    # define the constraints for the minimization (length 1 and perpendicular in mag space to R1)
    con_1 = constraint_ortho(R2, R1, l1, B_inv)
    con_2 = constraint_unity(R2, l2)
    pen_1 = constraint_ortho(R2, R1, (0.5*rho[0])**0.5, B_inv)**2
    pen_2 = constraint_unity(R2, (0.5*rho[1])**0.5)**2

    # calculate the 3 components of our augmented lagrangian
    chi = rCr + mCr + rCm + rCR + RCr + RCm + mCR + RCR
    con = con_1 + con_2
    pen = pen_1 + pen_2
    # calculate the augmented lagranian
    lag = chi/n + con + pen

    return lag


def calculate_R(R, E, dR, dE, C, m, l1, l2, rho, B_inv):
    # calculate terms of chi_sq if R (reddening or variation) is the variables
    mCm, mC, Cm = calculate_pre_loop(m,C)
    RC = dE[:,None]*np.einsum('i,nij->nj',dR,C)
    CR = dE[:,None]*np.einsum('nij,j->ni',C,dR)
    RCm = dE*np.einsum('i,ni->n',dR,Cm)
    mCR = dE*np.einsum('ni,i->n',mC,dR)
    RCR = dE*np.einsum('i,ni->n',dR,CR)
    n = len(E)

    # sum each term over all data points so the scipy method doesn't do the calculations
    EEC = np.sum(E[:,None,None]**2*C, axis=0)
    EmC = np.sum(E[:,None]*mC, axis=0)
    ECm = np.sum(E[:,None]*Cm, axis=0)
    ECR = np.sum(E[:,None]*CR, axis=0)
    ERC = np.sum(E[:,None]*RC, axis=0)
    RCm = np.sum(RCm, axis=0)
    mCR = np.sum(mCR, axis=0)
    RCR = np.sum(RCR, axis=0)

    partial = (EEC, EmC, ECm, ECR, ERC, RCm, mCR, RCR)
    arg = (l1, l2, n, rho, B_inv, dR) + partial

    co1 = lagrangian(R, 0, 0, n, [0,0], B_inv, dR, *partial) + mCm/n
    co2 = np.sum(chi_sq(E, R, dE, dR, C, m))/n
    if not np.allclose(co1,co2):
        print("Values differ by ",co1-co2)

    # minimizing the augmented lagranian with respect to R and append the solution
    with np.errstate(divide='ignore', invalid='ignore'):
        res = minimize(lagrangian, R, arg)


    #print(constraint_unity(res.x)+1, constraint_ortho(res.x,dR,B_inv=B_inv))
    return res.x


def predict_R(theta, nn):
    # create the partial Neural Network to predict R depending on normed theta
    inputs = nn.get_layer('theta').input
    outputs = nn.get_layer(name='R').output
    R_model = tf.keras.Model(inputs, outputs)

    # return the prediction for R
    return R_model(theta).numpy()


def normalize_theta(theta):
    # load the median and std needed to use the data set for the neural network
    with open('green2020_theta_normalizer.json', 'r') as f:
        d = json.load(f)

    # norm the data
    theta_med = np.array(d['theta_med'])
    theta_std = np.array(d['theta_std'])
    x = (theta - theta[None,:]) / theta_std[None,:]

    return x


def prepare_data(d, nn):
    """
    Predicts the data we need from the neural network and calculates the covariance matrix and it's
    inverse, which is needed for chi_sq,

    Parameters
    ----------
    d: np.ndarray
        Data-array which contains the theta, mag, parallax and their errors
    nn: keras.model
        The neural network from Greg, which we use to predict the magnitudes

    Returns
    -------
    dm: np.array
        The difference between observed magnitude in mag/color space and predicited unreddened mag
    C: np.array
        Contains the inverse covariance matrix for each star as per Gregs paper with reddening contr.
    R_mean: np.array
        The mean general reddening vector from our dataset, as predicted by the Neural Network
    B: np.array
        The matrix to transform from mag -> mag/color space
    B_inv: np.array
        The maxtrix to transform from mag/color -> mag space
    """
    n_stars, n_bands = d['mag'].shape
    large_err = 999.
    # define the partial model to predict BM
    inputs = nn.get_layer('theta').input
    outputs = nn.get_layer(name='BM').output
    B_M_model = tf.keras.Model(inputs, outputs)

    m = d['mag'].copy()
    cov_m = np.zeros((n_stars, n_bands, n_bands))

    # fill the cov. matrix with the observed mag errors, replace nans with median and large errors
    for b in range(n_bands):
        cov_m[:,b,b] = d['mag_err'][:,b]**2

        idx = ~np.isfinite(m[:,b]) | ~np.isfinite(cov_m[:,b,b])
        m[idx,b] = np.nanmedian(m[:,b])
        cov_m[idx,b,b] = large_err**2.

    # create B to transform between mag and mag/color space and B's inverse
    B = np.identity(n_bands, dtype='f4')
    B_inv = B.copy()
    B[1:,0] = -1.
    B_inv[1:,0] = 1.

    # transform m and cov_m to mag/color space
    m = np.einsum('ij,nj->ni',B,m)
    cov_m = np.einsum('nij,kj->nik',cov_m,B)
    cov_m = np.einsum('ij,njk->nik',B,cov_m)

    # subtract distance modulus from mags (Gaia's g-band is the only mag)
    m[:,0] -= (10. - 5.*np.log10(d['parallax']))
    err_over_plx = d['parallax_err']/d['parallax']
    cov_m[:,0,0] += (5./np.log(10.) * err_over_plx)**2

    # define a mask to identify bad distance measurements and flag them in cov. matrix with large err
    idx = (
          (err_over_plx > 0.2)
        | (d['parallax'] < 1.e-8)
        | ~np.isfinite(d['parallax'])
        | ~np.isfinite(d['parallax_err'])
    )

    cov_m[idx,0,0] = large_err**2
    m[idx,0] = np.nanmedian(m[:,0])

    # predict R, transform to proper space and take the mean
    R = predict_R(d['atm_param_p'], nn)
    BR = np.einsum('ij,nj->ni',B,R)
    R_mean = np.mean(BR, axis=0)

    # calculate the gradient of predicting BM for the covariance matrix
    with tf.GradientTape() as g:
        x_p = tf.constant(d['atm_param_p'])
        g.watch(x_p)
        mag_color = B_M_model([x_p])
    J = g.batch_jacobian(mag_color, x_p).numpy()

    cov_m += np.einsum('nik,nkl,njl->nij',J,d['atm_param_cov_p'],J)

    # invert the covariance matrix
    C = cov_m.copy()
    for i, cov in enumerate(cov_m):
        C[i] = np.linalg.inv(cov)

    # subtract the observed reddened BM from predicted unreddened BM
    dm = B_M_model(d['atm_param_p']).numpy()
    dm -= m

    return dm, C, R_mean, B, B_inv


def probe_chi_E(E, R, dm, C, dR, dE, i, label):
    probe = np.linspace(0.7,1.3,100)*E
    dm_c = np.full(probe.shape+dm.shape, dm)
    dE_c = np.full(probe.shape, dE)
    C_c = np.full(probe.shape+C.shape, C)
    chi = chi_sq(probe, R, dE_c, dR, C_c, dm_c)

    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,13), facecolor= 'white')
    ax = fig.subplots(1,1)
    ax.plot(probe, chi, color='black')
    ax.axvline(E,-10,100,color='red',alpha=0.6)
    ax.axhline(min(chi),-1,10,color='red',alpha=0.6)
    ax.set_ylim([min(chi)-0.2*max(chi), 1.1*max(chi)])
    ax.set_xlim([min(probe)*0.95, max(probe)*1.05])
    ax.set_xlabel(label)
    ax.set_ylabel('$\chi^2$')

    pic = label.replace(' ', '_')
    picp = 'pictures/' + pic + '_iteration_' + str(i)
    plt.savefig(picp, dpi = 150, bbox_inches='tight')


def probe_R(R, R_real, c, name):
    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,13), facecolor= 'white')
    ax = fig.subplots(1,1)
    ax.plot(range(len(c)), c, color='black')
    ax.axhline(1,-1,int(1.1*len(c)))
    ax.set_xlim(-1,int(1.1*len(c)))
    ax.set_ylabel('Iteration')
    ax.set_xlabel('Dotproduct of fitted ' + name + ' and real ' + name)

    pic = name.replace(' ', '_')
    picp = 'pictures/' + pic + '_Dot'
    plt.savefig(picp, dpi = 150, bbox_inches='tight')

    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,13), facecolor= 'white')
    ax = fig.subplots(1,1)
    ax.plot(np.linspace(1,len(R)-1,len(R)-1), R[1:]-R_real)
    ax.axhline(0,1,int(1.05*len(R)),color='black')
    ax.set_xlim(1,int(1.05*len(R)))
    ax.set_ylabel('Difference of fitted ' + name + ' and real ' + name)
    ax.set_xlabel('Iteration')

    pic = name.replace(' ', '_')
    picp = 'pictures/' + pic + '_Difference'
    plt.savefig(picp, dpi = 150, bbox_inches='tight')


def result_hist(x, iter, label, xlim = None):
    std = np.std(x)

    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,14), facecolor = 'white')
    ax = fig.subplots(1,1)
    ax.hist(x, bins='auto', label='$\sigma$ = {}'.format(std))
    ax.legend()
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_xlabel(label)
    """
    for n, l in enumerate(ax.xaxis.get_ticklabels()):
        if n % 3 != 0:
            l.set_visible(False)
    """
    if xlim is not None:
        if hasattr(xlim,'__len__'):
            if len(xlim) == 2:
                ax.set_xlim(xlim)
            else:
                raise ValueError('More than 2 values for xlim')
        else:
            ax.set_xlim(0,xlim)

    pic = label.replace(' ', '_')
    picp = 'pictures/' + pic + '_iteration_' + str(iter)
    plt.savefig(picp, dpi = 150, bbox_inches='tight')


def main():
    # set a max loop limit if we don't converge
    convergence_limit = 1000
    test_len = 101
    sample = 900

    # the path where the data is saved
    dir = 'data/'
    mock = dir + 'mock_data.h5'
    model_path = dir + 'green2020_nn_model.h5'
    saving = dir + 'mock_result.h5'
    tests = ['dE','E','dR','R','dEdR']
    test = ''

    if len(sys.argv) == 2:
        if sys.argv[1] in tests:
            test = sys.argv[1]
        else:
            print("Only arguments accepted are: ", tests)
    elif len(sys.argv) > 2:
        print("Only 1 of ", tests, " can be passed")

    #TODO Maybe lower rho as iterations go on?
    rho = [4e-1,4e-1,4e-1,4e-1]

    # read out the data, Neural Network and prepare the data with the Neural Network
    data, r_fit, R_mock, dr_mock, dR_mock = read_out(mock)
    nn_model = tf.keras.models.load_model(model_path)
    dm, C, BR_m, B, B_inv = prepare_data(data, nn_model)
    r_fit *= np.dot(BR_m, BR_m)**0.5
    BR_m /= np.dot(BR_m, BR_m)**0.5

    # prepare the list that saved the values for each iteration (!!we fit for BR/BdR!!)
    BR = [BR_m]
    E = [r_fit]
    BdR = [guess_dR(BR_m, B, B_inv)]
    dE = []
    chi = []

    # set an inital value for lambdas
    l1, l2, l3, l4 = [0], [0], [0], [0]

    if test == 'dE':
        BdR.append(dR_mock)
        BR.append(R_mock)
        dE_temp = calculate_E(BdR[-1], E[-1], BR[-1], C, dm)
        dE.append(dE_temp)
        diff = dE[-1] - dr_mock
        result_hist(diff, 0, 'Difference fitted dE real dE')
        la = 'Reddening Variation Star ' + str(sample)
        probe_chi_E(dE[-1][sample], BdR[-1], dm[sample], C[sample], BR[-1], E[-1][sample], 0, la)

        return 0

    if test == 'E':
        BdR.append(dR_mock)
        BR.append(R_mock)
        dE.append(dr_mock)
        E_temp = calculate_E(BR[-1], dE[-1], BdR[-1], C, dm)
        E.append(E_temp)
        diff = E[-1] - r_fit
        result_hist(diff, 0, 'Difference fitted E real E')
        la = 'Reddening Star ' + str(sample)
        probe_chi_E(E[-1][sample], BR[-1], dm[sample], C[sample], BdR[-1], dE[-1][sample], 0, la)

        return 0

    if test == 'dR':
        dE.append(dr_mock+np.random.default_rng(14).normal(size=len(dr_mock))*0.1*np.mean(dr_mock))
        BR.append(R_mock)
        te = np.zeros((test_len,))
        for i in range(test_len):
            BdR_temp = calculate_R(BdR[-1], dE[-1], BR[-1], E[-1], C, dm, l1[-1], l2[-1], rho[:2], B_inv)
            BdR.append(BdR_temp)
            l_1 = l1[-1] + constraint_ortho(BdR[-1],BR[-1],rho[0],B_inv)
            l_2 = l2[-1] + constraint_unity(BdR[-1],rho[1])
            l1.append(l_1)
            l2.append(l_2)
            te[i] = np.dot(BdR[-1],dR_mock)
        probe_R(BdR, dR_mock, te, 'dR')

        return 0

    if test == 'R':
        dE.append(dr_mock)
        E[-1] += np.random.default_rng(14).normal(size=len(E[-1]))*0.1*np.mean(E[-1])
        BdR.append(dR_mock)
        te = np.zeros((test_len,))
        for i in range(test_len):
            BR_temp = calculate_R(BR[-1], E[-1], BdR[-1], dE[-1], C, dm, l3[-1], l4[-1], rho[2:], B_inv)
            BR.append(BR_temp)
            l_3 = l3[-1] + constraint_ortho(BR[-1], BdR[-1], rho[2], B_inv)
            l_4 = l4[-1] + constraint_unity(BR[-1], rho[3])
            l3.append(l_3)
            l4.append(l_4)
            te[i] = np.dot(BR[-1], R_mock)
        probe_R(BR, R_mock, te, 'R')
        print(np.dot(R_mock,BR[-1]))

        return 0

    if test == 'dEdR':
        BR.append(R_mock)
        te = np.zeros((test_len,))
        for i in range(test_len):
            dE_temp = calculate_E(BdR[-1], E[-1], BR[-1], C, dm)
            dE.append(dE_temp)
            if i%50==0:
                diff = dE[-1] - dr_mock
                result_hist(diff, i, 'Difference fitted dE real dE')
                la = 'Reddening Variation Star ' + str(sample)
                probe_chi_E(dE[-1][sample], BdR[-1], dm[sample], C[sample], BR[-1], E[-1][sample], i, la)
            BdR_temp = calculate_R(BdR[-1], dE[-1], BR[-1], E[-1], C, dm, l1[-1], l2[-1], rho[:2], B_inv)
            BdR.append(BdR_temp)
            l_1 = l1[-1] + constraint_ortho(BdR[-1],BR[-1],rho[0],B_inv)
            l_2 = l2[-1] + constraint_unity(BdR[-1],rho[1])
            l1.append(l_1)
            l2.append(l_2)
            te[i] = np.dot(BdR[-1],dR_mock)
        probe_R(BdR, dR_mock, te, 'dR with dE')

        return 0


    for i in range(convergence_limit):
        # calculate the root of chi_sq for dE
        dE_temp = calculate_E(BdR[-1], E[-1], BR[-1], C, dm)
        dE.append(dE_temp)
        # calculate the terms of chi_sq when fitting for dR, that are independant of dR
        BdR_temp = calculate_R(BdR[-1], dE[-1], BR[-1], E[-1], C, dm, l1[-1], l2[-1], rho[:2], B_inv)
        BdR.append(BdR_temp)
        # update lambda_1 and _2 according to the constraints and append
        l_1 = l1[-1] + constraint_ortho(BdR[-1],BR[-1],rho[0],B_inv)
        l_2 = l2[-1] + constraint_unity(BdR[-1],rho[1])
        l1.append(l_1)
        l2.append(l_2)

        # calculate the root of chi_sq for E and append
        E_temp = calculate_E(BR[-1], dE[-1], BdR[-1], C, dm)
        E.append(E_temp)
        BR_temp = calculate_R(BR[-1], E[-1], BdR[-1], dE[-1], C, dm, l3[-1], l4[-1], rho[2:], B_inv)
        BR.append(BR_temp)
        # update lambda_3 and _4 according to the constraints and append
        l_3 = l3[-1] + constraint_ortho(BdR[-1],BR[-1],rho[2],B_inv)
        l_4 = l4[-1] + constraint_unity(BR[-1], rho[3])
        l3.append(l_3)
        l4.append(l_4)

        # Progress
        chi_t = chi_sq(E[-1], BR[-1], dE[-1], BdR[-1], C, dm)
        chi.append(chi_t)
        if i%(convergence_limit//20) == 0:
            print(f'Iteration {i} Complete')


    # checking if we actually converged or did max amount of iterations without converging
    if i == (convergence_limit-1):
        print('No convergence within {} iterations possible!'.format(convergence_limit))

    # saving the data
    dtype = [('l1','f4'),('l2','f4'),('l3','f4'),('l4','f4')]
    l = np.empty(len(l1),dtype=dtype)
    l['l1'] = l1
    l['l2'] = l2
    l['l3'] = l3
    l['l4'] = l4

    save(saving, E, BR, dE, BdR, l, dm, C, chi)

    return 0

if __name__ == '__main__':
    main()

