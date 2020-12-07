#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predicting reddening and variation through chi_sq minimization of diff. between obs. and pred. mag
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
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
            R, dr, dR = None, None, None

    return d, r, R, dr, dR


def save(path, r, R, dr, dR, l, dm , C, chi_sq, c, cov_m):
    # save the mock data run
    with h5py.File(path, 'w') as f:
        f.create_dataset('/E', data=r, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dE', data=dr, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/R', data=R, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dR', data=dR, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/l', data=l, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dm', data=dm, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/C', data=C, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/chi', data=chi_sq, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/con', data=c, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/cov', data=cov_m, chunks=True, compression='gzip', compression_opts=3)

def guess_dR(R):
    # read out the mean wavelengths of the 13 filters we use
    with open('mean_wavelengths.json', 'r') as f:
        mean = json.load(f)

    x = np.array(mean)

    # calculate the offset of our linear dR-model to be perpendicular to R
    b = np.sum(x*R)/np.sum(R)
    y = x[:] - b[None]

    # norm it to 1 and transform it to mag/color space
    y /= np.dot(y,y)**0.5
    BdR = y.copy()
    BdR[1:] -= BdR[0]

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


def constraint_ortho(R_o, dR_o, l1 = None, B_inv = True):
    # transform R and dR into the right space where we want them to be orthogonal
    R = R_o.copy()
    dR = dR_o.copy()
    if B_inv:
        R[1:] += R[0]
        dR[1:] += dR[0]
    # constraint that is  f(R,dR) - b
    con = (np.dot(R,dR))

    if l1 is not None:
        con *= l1

    return con


def constraint_unity(R_o, l2 = None, B_inv = True):
    R = R_o.copy()
    if B_inv:
        R[1:] += R[0]
    # constraint that is f(R) - b
    con = (np.dot(R,R)**0.5 - 1)

    if l2 is not None:
        con *= l2

    return con


def lagrangian(R2, l1, l2, n, rho, R1, EEC, EmC, ECm, ECR, ERC, RCm, mCR, RCR):
    # calculate the terms that depend on the varibale we minimize for in chi_sq
    rCr = np.einsum('i,ij,j',R2,EEC,R2)
    mCr = np.dot(EmC,R2)
    rCm = np.dot(ECm,R2)
    rCR = np.dot(ERC,R2)
    RCr = np.dot(ECR,R2)

    # define the constraints for the minimization (length 1 and perpendicular in mag space to R1)
    con_1 = constraint_ortho(R2, R1, l1)
    con_2 = constraint_unity(R2, l2)
    pen_1 = constraint_ortho(R2, R1, (0.5*rho[0])**0.5)**2
    pen_2 = constraint_unity(R2, (0.5*rho[1])**0.5)**2

    # calculate the 3 components of our augmented lagrangian
    chi = rCr + mCr + rCm + rCR + RCr + RCm + mCR + RCR
    con = con_1 + con_2
    pen = pen_1 + pen_2
    # calculate the augmented lagranian
    lag = chi/n + con + pen

    return lag


def calculate_R(R, E, dR, dE, C, m, l1, l2, rho, dof):
    # calculate terms of chi_sq if R (reddening or variation) is the variables
    mCm, mC, Cm = calculate_pre_loop(m,C)
    RC = dE[:,None]*np.einsum('i,nij->nj',dR,C)
    CR = dE[:,None]*np.einsum('nij,j->ni',C,dR)
    RCm = dE*np.einsum('i,ni->n',dR,Cm)
    mCR = dE*np.einsum('ni,i->n',mC,dR)
    RCR = dE*np.einsum('i,ni->n',dR,CR)

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
    arg = (l1, l2, dof, rho, dR) + partial

    co1 = lagrangian(R, 0, 0, dof, [0,0], dR, *partial) + mCm/dof
    co2 = np.sum(chi_sq(E, R, dE, dR, C, m))/dof
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


def prepare_data(d, nn, r_fit):
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
    dof = n_stars*(n_bands-2)
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
        dof -= np.sum(idx)
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

    dof -= np.sum(idx)
    cov_m[idx,0,0] = large_err**2
    m[idx,0] = np.nanmedian(m[:,0])

    # predict R, transform to proper space and take the mean
    R = predict_R(d['atm_param_p'], nn)
    R_mean = np.mean(R, axis=0)
    r_fit *= np.dot(R_mean,R_mean)**0.5
    R_mean /= np.dot(R_mean,R_mean)**0.5
    # R_mean += np.random.default_rng(10).normal(size=R_mean.shape)

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

    return dm, C, R_mean, r_fit, dof, cov_m


def result_hist(x, iter, label, prefix, xlim = None):
    # plot the result histogram and calculate mean and std of the values
    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,14), facecolor = 'white')
    ax = fig.subplots(1,1)
    std = np.std(x)
    mean = np.mean(x)
    ax.hist(x, bins='auto', label='$\sigma$ = {} \n $\mu$ = {}'.format(round(std,3),round(mean,3)))
    ax.legend()
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_xlabel(label)

    if xlim is not None:
        if hasattr(xlim,'__len__'):
            if len(xlim) == 2:
                ax.set_xlim(xlim)
            else:
                raise ValueError('More than 2 values for xlim')
        else:
            ax.set_xlim(0,xlim)
    else:
        lim = 1.1*max(max(x),-min(x))
        ax.set_xlim(-lim,lim)

    pic = label.replace(' ', '_')
    picp = 'pictures/'+ prefix + pic + '_iteration_' + str(iter)
    plt.savefig(picp, dpi = 150, bbox_inches='tight')


def lambda_plot(l, label, prefix):
    # plot the lambda given to the function
    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,14), facecolor = 'white')
    ax = fig.subplots(1,1)
    ax.plot(range(len(l)), l, 'k-')
    ax.set_ylabel(label)
    ax.set_xlabel('Iteration')
    ax.grid(True)
    pic = label.replace(' ', '_')
    picp = 'pictures/' + prefix + pic
    plt.savefig(picp, dpi = 150, bbox_inches='tight')


def probe_chi_E(E, R, dm, C, dR, dE, i, label):
    # create the arrays with only varying E to probe what effect small changes in it make on chi
    probe = np.linspace(0.7,1.3,100)*E
    dm_c = np.full(probe.shape+dm.shape, dm)
    dE_c = np.full(probe.shape, dE)
    C_c = np.full(probe.shape+C.shape, C)
    chi = chi_sq(probe, R, dE_c, dR, C_c, dm_c)

    # a simple plot for the range we probed with 2 lines indidcated the minimum chi and E
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


def probe_R(R, ortho, unity, R_real, c, name, prefix):
    # plot the change in the dot product of fit and real value over iterations 
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

    # plot the difference between fit and real R's component
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

    # plot unity and orthogonal lambda over iterations
    ort = 'Orthogonal Lambda for ' + name
    uni = 'Unity Lambda for ' + name
    lambda_plot(ortho, ort, prefix)
    lambda_plot(unity, uni, prefix)


def unit_tests(test, BR, BdR, E, dE, l1, l2, dm, C, rho, test_len, prefix):
    # the different tests possible that only solves parts on its own to verify that it is working
    sample = 900
    te = np.zeros((test_len,))

    if test == 'dE':
        dE_temp = calculate_E(BdR[-1], E[-1], BR[-1], C, dm)
        dE.append(dE_temp)
        diff = dE[-1] - E[0]
        result_hist(diff, 0, 'Difference fitted dE real dE', prefix)
        la = 'Reddening Variation Star ' + str(sample)
        probe_chi_E(dE[-1][sample], BdR[-1], dm[sample], C[sample], BR[-1], E[-1][sample], 0, la)

    if test == 'E':
        E_temp = calculate_E(BR[-1], dE[-1], BdR[-1], C, dm)
        E.append(E_temp)
        diff = E[-1] - r_fit
        result_hist(diff, 0, 'Difference fitted E real E', prefix)
        la = 'Reddening Star ' + str(sample)
        probe_chi_E(E[-1][sample], BR[-1], dm[sample], C[sample], BdR[-1], dE[-1][sample], 0, la)

    if test == 'dR':
        # dE[-1] += np.random.default_rng(14).normal(size=len(dr_mock))*0.1*np.mean(dr_mock))
        for i in range(test_len):
            BdR_temp = calculate_R(BdR[-1], dE[-1], BR[-1], E[-1], C, dm, l1[-1], l2[-1], rho[:2])
            BdR.append(BdR_temp)
            l_1 = l1[-1] + constraint_ortho(BdR[-1],BR[-1],rho[0],True)
            l_2 = l2[-1] + constraint_unity(BdR[-1],rho[1],True)
            l1.append(l_1)
            l2.append(l_2)
            te[i] = np.dot(BdR[-1],BdR[1])
        probe_R(BdR, l1, l2, BdR[1], te, 'dR', prefix)

    if test == 'R':
        # E[-1] += np.random.default_rng(14).normal(size=len(E[-1]))*0.1*np.mean(E[-1])
        for i in range(test_len):
            BR_temp = calculate_R(BR[-1], E[-1], BdR[-1], dE[-1], C, dm, l1[-1], l2[-1], rho[2:])
            BR.append(BR_temp)
            l_1 = l1[-1] + constraint_ortho(BR[-1], BdR[-1], rho[2], True)
            l_2 = l2[-1] + constraint_unity(BR[-1], rho[3], True)
            l1.append(l_1)
            l2.append(l_2)
            te[i] = np.dot(BR[-1], BR[1])
        probe_R(BR, l1, l2, BR[1], te, 'R', prefix)

    if test == 'dEdR':
        for i in range(test_len):
            dE_temp = calculate_E(BdR[-1], E[-1], BR[-1], C, dm)
            dE.append(dE_temp)
            if i%((test_len-1)//1)==0:
                diff = dE[-1] - dE[0]
                result_hist(diff, i, 'Difference fitted dE real dE', prefix)
                la = 'Reddening Variation Star ' + str(sample)
                probe_chi_E(dE[-1][sample], BdR[-1], dm[sample], C[sample], BR[-1], E[-1][sample], i, la)
            BdR_temp = calculate_R(BdR[-1], dE[-1], BR[-1], E[-1], C, dm, l1[-1], l2[-1], rho[:2])
            BdR.append(BdR_temp)
            l_1 = l1[-1] + constraint_ortho(BdR[-1],BR[-1],rho[0],True)
            l_2 = l2[-1] + constraint_unity(BdR[-1],rho[1],True)
            l1.append(l_1)
            l2.append(l_2)
            te[i] = np.dot(BdR[-1],BdR[1])
        probe_R(BdR, l1, l2, BdR[1], te, 'dR with dE', prefix)

    if test == 'ER':
        for i in range(test_len):
            E_temp = calculate_E(BR[-1], dE[-1], BdR[-1], C, dm)
            E.append(E_temp)
            if i%((test_len-1)//1)==0:
                diff = E[-1] - E[0]
                result_hist(diff, i, 'Difference fitted E real E', prefix)
                la = 'Reddening Star ' + str(sample)
                probe_chi_E(E[-1][sample], BR[-1], dm[sample], C[sample], BdR[-1], dE[-1][sample], i, la)
            BR_temp = calculate_R(BR[-1], E[-1], BdR[-1], dE[-1], C, dm, l1[-1], l2[-1], rho[2:])
            BR.append(BR_temp)
            l_1 = l1[-1] + constraint_ortho(BR[-1],BdR[-1],rho[2],True)
            l_2 = l2[-1] + constraint_unity(BR[-1],rho[3],True)
            l1.append(l_1)
            l2.append(l_2)
            te[i] = np.dot(BR[-1],BR[1])
        probe_R(BR, l1, l2, BR[1], te, 'R with E', prefix)


def main():
    # the path where the data is saved
    dir = 'data/'
    model_path = dir + 'green2020_nn_model.h5'
    prefix = 'green2020_small_'
    path = dir + prefix + 'data.h5'
    saving = dir + prefix +  'result.h5'

    #TODO Maybe lower rho as iterations go on?
    rho = [1e-1,1e-1,1e-1,1e-0]
    l1, l2, l3, l4 = [0], [0], [0], [0]

    # resolve arguments for the unit tests; see tests for possible test
    parser = argparse.ArgumentParser(prog='Variation Fit')
    parser.add_argument('-t', '--test',default=None, choices=[None,'dE','E','dR','R','dEdR','ER'],
                                     help='For testing, default value is None')
    parser.add_argument('limit' ,default=1000, nargs='?', type=int,
                                     help='Max iterations, default is 1000')
    args = parser.parse_args()
    limit = args.limit
    test = args.test

    # read out the data, Neural Network and prepare the data with the Neural Network
    data, r, R_mock, dr_mock, dR_mock = read_out(path)
    nn_model = tf.keras.models.load_model(model_path)
    dm, C, R, r_fit, dof, cov_m = prepare_data(data, nn_model, r)

    # prepare the list that saved the values for each iteration (!!we fit for BR/BdR!!)
    BR_m = R.copy()
    BR_m[1:] -= BR_m[0]
    BR = [BR_m]
    E = [r_fit]
    BdR = [guess_dR(R)]
    dE = []
    chi = []

    if test is not None:
        BdR.append(dR_mock)
        BR.append(R_mock)
        dE.append(dr_mock)
        unit_tests(test, BR, BdR, E, dE, l1, l2, dm, C, rho, limit, prefix)
        return 0

    r_dot = np.empty((limit,))
    r_len = np.empty((limit,))
    dr_dot = np.empty((limit,))
    dr_len = np.empty((limit,))
    rm_dot = np.empty((limit,))
    drm_dot = np.empty((limit,))

    for i in range(limit):
        # calculate dE
        dE_temp = calculate_E(BdR[-1], E[-1], BR[-1], C, dm)
        dE.append(dE_temp)

        # calculate dR
        BdR_temp = calculate_R(BdR[-1], dE[-1], BR[-1], E[-1], C, dm, l1[-1], l2[-1], rho[:2], dof)
        BdR.append(BdR_temp)
        dR = BdR[-1].copy()
        dR[1:] += dR[0]
        dr_dot[i] = np.dot(dR,R)
        dr_len[i] = np.dot(dR,dR)**0.5

        # update lambda_1 and _2 according to the constraints and append
        l_1 = l1[-1] + constraint_ortho(BdR[-1],BR[-1],rho[0])
        l_2 = l2[-1] + constraint_unity(BdR[-1],rho[1])
        l1.append(l_1)
        l2.append(l_2)

        # calculate E
        E_temp = calculate_E(BR[-1], dE[-1], BdR[-1], C, dm)
        E.append(E_temp)

        # calculate R
        BR_temp = calculate_R(BR[-1], E[-1], BdR[-1], dE[-1], C, dm, l3[-1], l4[-1], rho[2:], dof)
        BR.append(BR_temp)
        R = BR[-1].copy()
        R[1:] += R[0]
        r_dot[i] = np.dot(R,dR)
        r_len[i] = np.dot(R,R)**0.5

        # update lambda_3 and _4 according to the constraints and append
        l_3 = l3[-1] + constraint_ortho(BdR[-1],BR[-1],rho[2])
        l_4 = l4[-1] + constraint_unity(BR[-1], rho[3])
        l3.append(l_3)
        l4.append(l_4)

        # Progress
        chi_t = np.sum(chi_sq(E[-1], BR[-1], dE[-1], BdR[-1], C, dm))/dof
        chi.append(chi_t)
        if i%(limit//20) == 0:
            print(f'Iteration {i} Complete')


        # if integration testing
        if R_mock is not None:
            rm_dot[i] = np.dot(BR[-1],R_mock)
            drm_dot[i] = np.dot(BdR[-1],dR_mock)
            if i%(limit//3) == 0:
                diff = dE[-1]*np.dot(BdR[-1],BdR[-1])**0.5 - dr_mock
                result_hist(diff, i, 'Difference dE', prefix)
                diff = E[-1]*np.dot(BR[-1],BR[-1])**0.5 - E[0]
                result_hist(diff, i, 'Difference E', prefix)
                print(f'Iteration {i}: Made a dE and E difference plot')


    # checking if we actually converged or did max amount of iterations without converging
    if i == (limit-0):
        print('No convergence within {} iterations possible!'.format(limit))

    # plots of mock taken over iterations
    if R_mock is not None:
        plots = [(drm_dot, 'Dot dR mock'), (rm_dot, 'Dot R mock')]
        print('Difference in R and mock:')
        print(BR[-1] - R_mock)
        print('Angle between R and mock:')
        print(np.arccos(np.dot(BR[-1],R_mock)/(np.dot(R_mock,R_mock)**0.5*r_len[-1]))*180/np.pi)
        print('Difference in dR and mock:')
        print(BdR[-1] - dR_mock)
        print('Angle between dR and mock:')
        print(np.arccos(np.dot(BdR[-1],dR_mock)/(np.dot(dR_mock,dR_mock)**0.5*dr_len[-1]))*180/np.pi)
        for (arr, text) in plots:
            lambda_plot(arr, text, prefix)



    # saving the data
    l = np.empty((len(l1),4))
    c = np.empty((len(r_dot),4))
    for idx, (lx, cx) in enumerate([(l1,r_dot),(l2,r_len),(l3,dr_dot),(l4,dr_len)]):
        l[:,idx] = lx
        c[:,idx] = cx

    save(saving, E, BR, dE, BdR, l, dm, C, chi, c, cov_m)

    return 0

if __name__ == '__main__':
    main()

