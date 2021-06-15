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


def vary_mock(V_ang, R_ang, dR_ang, V_mock, R_mock, dR_mock, seed=12):
    rng = np.random.default_rng(seed)

    scale = V_ang/13**0.5*np.pi/180
    rando = rng.normal(0,scale,13)
    guess = gram_schmitt(rando, V_mock)
    V_mock1 = V_mock + guess

    scale = R_ang/13**0.5*np.pi/180
    rando = rng.normal(0,scale,13)
    guess1 = gram_schmitt(rando, R_mock)
    R_mock1 = R_mock + guess1
    R_mock1 /= np.dot(R_mock1,R_mock1)**0.5

    scale = dR_ang/13**0.5*np.pi/180
    rando = rng.normal(0,scale,13)
    guess2 = gram_schmitt(rando, dR_mock)
    dR_mock1 = dR_mock + guess2
    dR_mock2 = gram_schmitt(dR_mock1, R_mock1)
    dR_mock2 /= np.dot(dR_mock2,dR_mock2)**0.5

    print(angle(V_mock1,V_mock))
    print(angle(R_mock1,R_mock))
    print(angle(dR_mock2,dR_mock))
    print(np.dot(dR_mock2,R_mock1))
    print(np.dot(dR_mock2,dR_mock2)**0.5)

    return V_mock1, R_mock1, dR_mock2


def read_out(path):
    """
    Read out the test data set located at the specified path, if not mock path we assign Nones
    """
    with h5py.File(path, 'r') as f:
        d = f['data'][:]
        r = f['r_fit'][:]
        if 'mock' in path:
            R = f['R'][:]
            dr = f['dr_fit'][:]
            dR = f['dR'][:]
            V = f['V'][:]
        else:
            R, dr, dR, V = None, None, None, None


    old = ["Gaia_G", "Gaia_BP", "Gaia_RP", "PS1_g", "PS1_r", "PS1_i", "PS1_z", "PS1_y",
     "2MASS_J", "2MASS_H", "2MASS_K", "unWISE_W1", "unWISE_W2"]


    new = ["Gaia_G", "Gaia_BP", "Gaia_RP", "GALEX_FUV", "GALEX_NUV", "SDSS_u", "PS1_g", "PS1_r",
     "PS1_i", "PS1_z", "PS1_y", "DECam_g", "DECam_r", "DECam_i", "DECam_z", "DECam_Y", "2MASS_J",
     "2MASS_H", "2MASS_K", "unWISE_W1", "unWISE_W2", "UKIDSS_GPS_J", "UKIDSS_GPS_H", "UKIDSS_GPS_K",
     "UKIDSS_UHS_J", "Vista_VHS_J", "Vista_VHS_H", "Vista_VHS_KS", "GLIMPSE_3_6", "GLIMPSE_4_5"]

    id = []
    for t in old:
        id.append(new.index(t))

    print(id)

    n_stars, n_bands = d['mag'].shape

    dtype = [('mag','f4',(13,)),('mag_err','f4',(13,))]
    mag = np.ndarray((n_stars,),dtype=dtype)

    if n_bands != 13:
        print('Sorting out unstable mags')
        mag['mag'] = d['mag'][:,id]
        mag['mag_err'] = d['mag_err'][:,id]
    else:
        mag['mag'] = d['mag']
        mag['mag_err'] = d['mag_err']

    return d, mag, r, R, dr, dR, V


def save(path, d):
    # save the mock data run
    with h5py.File(path, 'w') as f:
        for key in d.keys():
            f.create_dataset(key, data=d[key], chunks=True, compression='gzip', compression_opts=9)


def guess_dR(R, file='mean'):
    # read out the mean wavelengths of the 13 filters we use
    with open(file+'_wavelength.json', 'r') as f:
        mean = json.load(f)

    # inverted parabula peaking around 1 mikrometer
    x = np.array(mean)
    l = 1/(x*1e-4)
    p = -np.abs(x-1.2)+4

    # Gram-Schmidt process
    y = gram_schmitt(p, R)

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


def chi_sq(dE, dR, E, R, C, dm, TE = np.array(0), TR = np.array(0), alpha = 0):
    # calculate (dm + E*R + dE*dR) for each star
    bracket = dm[:,:] + E[:,None]*R[None,:] + dE[:,None]*dR[None,:] + TE[:,None]*TR[None,:]
    # calculate chi_square for each star and return it as an array (not summed)
    cho = np.einsum('ni,nij->nj',bracket,C)
    chi = np.einsum('ni,ni->n',cho,bracket)
    chi += alpha*dE**2

    return chi


def calculate_E(Ro, dE, dRo, C, m, alpha = 0):
    """
    The function to find the E that minimizes the chi_sq function, finding the origin of the
    parabula.
    ----------------------------------------------------------------------------------------
    
    """
    R = Ro.copy()
    dR = dRo.copy()
    if len(R) == 13:
        R = [R]
    if len(dR) == 13:
        dR = [dR]
    # calculate the constant term with dm+reddening (or variation)
    dm = m[:,:] + dE[:,None] * dR
    # calculate some terms of the chi_square with just E as variable, which can be reddening or var.
    RCdm = np.einsum('ni,nij,nj->n',R,C,dm)
    dmCR = np.einsum('ni,nij,nj->n',dm,C,R)
    RCR = np.einsum('ni,nij,nj->n',R,C,R)
    # calculate the root of the chi_sq function of E, which can be reddening or variation
    E = -0.5*(RCdm + dmCR)/(RCR+alpha)

    return E


def constraint_ortho(R_o, dR_o, l1 = None, B_inv = True):
    """
    The orthogonal constraint function we use to calculate the penalty/diviation term
    """
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
    """
    The unity constraint function we use to calculate the penalty/diviation term
    """
    R = R_o.copy()
    if B_inv:
        R[1:] += R[0]
    # constraint that is f(R) - b
    con = (np.dot(R,R)**0.5 - 1)

    if l2 is not None:
        con *= l2

    return con


def lagrangian(R2, l1, l2, n, rho, R1, d):
    """
    The function that we let scipy.optimise fit with.
    ---------------------------------------------------------------------------------------------
    R2: the vector we fit
    l1/l2: the penalty terms from previous iterations
    n: the degrees of freedom we have
    rho: rate of growth - it controls how much of the penalty we take into account
    R1: the perpendicular vector we need for one of the constraints
    d: dic that contains and the precalculated terms to shorten computation time
    --------------------------------------------------------------------------------------------
    Returns the calculated augmented lagranian
    """
    # calculate the terms that depend on the varibale we minimize for in chi_sq
    rCr = np.einsum('i,ij,j',R2,d['EEC'],R2)
    mCr = np.dot(d['EmC'],R2)
    rCm = np.dot(d['ECm'],R2)
    rCR = np.dot(d['ERC'],R2)
    RCr = np.dot(d['ECR'],R2)
    TCr = np.dot(d['ETC'],R2)
    rCT = np.dot(d['ECT'],R2)

    # define the constraints for the minimization (length 1 and perpendicular in mag space to R1)
    con_1 = constraint_ortho(R2, R1, l1)
    con_2 = constraint_unity(R2, l2)
    pen_1 = constraint_ortho(R2, R1, (0.5*rho[0])**0.5)**2
    pen_2 = constraint_unity(R2, (0.5*rho[1])**0.5)**2

    # calculate the 3 components of our augmented lagrangian
    dic = d['TCm'] + d['mCT'] + d['TCR'] + d['RCT'] + d['RCm'] + d['mCR'] + d['RCR'] + d['TCT']
    chi = rCr + mCr + rCm + rCR + RCr + TCr + rCT + dic
    con = con_1 + con_2
    pen = pen_1 + pen_2
    # calculate the augmented lagranian
    lag = chi/n + con + pen

    return lag


def calculate_R(R, E, dR, dE, C, m, l1, l2, rho, dof, TR = None, TE = None):
    """
    Method to calculate R in our model of E*R+dE*dR+TE*TR+m. Due to the interchangibility of the
    terms it can also be used to calcualte dR and TR by assigning values correctly.
    ---------------------------------------------------------------------------------------------
    rho: The 2 values that determine growth in connection with constraint l1 and l2
    l1/l2: The 2 penalty terms from the last iteration
    dof: degree of freedom of our fit model to compare chi_sq and singular terms
    ---------------------------------------------------------------------------------------------
    We expand the bracket expression to complete the sums before the fit to safe computation time
    within the scipy.optimize. We check calculate were done correctly with np.allclose and chi_sq
    function.
    ---------------------------------------------------------------------------------------------
    Returns the fitted R vector
    """
    if TR is None or TE is None:
        TR = np.zeros_like(R)
        TE = np.zeros_like(E)
    # calculate terms of chi_sq if R (reddening or variation) is the variables
    mCm, mC, Cm = calculate_pre_loop(m,C)
    RC  = dE[:,None]*np.einsum('i,nij->nj',dR,C)
    CR  = dE[:,None]*np.einsum('nij,j->ni',C,dR)
    TC  = TE[:,None]*np.einsum('i,nij->nj',TR,C)
    CT  = TE[:,None]*np.einsum('nij,j->ni',C,TR)
    RCm = dE*np.einsum('i,ni->n',dR,Cm)
    mCR = dE*np.einsum('ni,i->n',mC,dR)
    TCm = TE*np.einsum('i,ni->n',TR,Cm)
    mCT = TE*np.einsum('ni,i->n',mC,TR)
    TCR = TE*np.einsum('i,ni->n',TR,CR)
    RCT = dE*np.einsum('i,ni->n',dR,CT)
    RCR = dE*np.einsum('i,ni->n',dR,CR)
    TCT = TE*np.einsum('i,ni->n',TR,CT)

    # sum each term over all data points so the scipy method doesn't do the calculations
    EEC = np.sum(E[:,None,None]**2*C, axis=0)
    EmC = np.sum(E[:,None]*mC, axis=0)
    ECm = np.sum(E[:,None]*Cm, axis=0)
    ECR = np.sum(E[:,None]*CR, axis=0)
    ERC = np.sum(E[:,None]*RC, axis=0)
    ETC = np.sum(E[:,None]*TC, axis=0)
    ECT = np.sum(E[:,None]*CT, axis=0)
    TCR = np.sum(TCR, axis=0)
    RCT = np.sum(RCT, axis=0)
    RCm = np.sum(RCm, axis=0)
    mCR = np.sum(mCR, axis=0)
    TCm = np.sum(TCm, axis=0)
    mCT = np.sum(mCT, axis=0)
    TCT = np.sum(TCT, axis=0)
    RCR = np.sum(RCR, axis=0)

    partial = {}
    partials = ((EEC,'EEC'), (EmC,'EmC'), (ECm,'ECm'), (ECR,'ECR'), (ERC,'ERC'),
                (ETC,'ETC'), (ECT,'ECT'), (TCR,'TCR'), (RCT,'RCT'), (RCm,'RCm'),
                (mCR,'mCR'), (TCm,'TCm'), (mCT,'mCT'), (TCT,'TCT'), (RCR,'RCR'))

    for i, key in partials:
        partial[key] = i

    co1 = lagrangian(R, 0, 0, dof, [0,0], dR, partial) + mCm/dof
    co2 = np.sum(chi_sq(E, R, dE, dR, C, m, TE, TR))/dof
    if not np.allclose(co1,co2):
        print("Values differ by ",co1-co2)

    arg = (l1, l2, dof, rho, dR, partial)
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


def normalize_theta(theta, theta_cov):
    # load the median and std needed to use the data set for the neural network
    with open('green2020_theta_normalization.json', 'r') as f:
        d = json.load(f)

    # norm the data
    theta_med = np.array(d['theta_med'])
    theta_std = np.array(d['theta_std'])

    x = (theta - theta_med[None,:]) / theta_std[None,:]

    x_cov = theta_cov.copy()
    for i in range(3):
        x_cov[:,i,:] /= theta_std[i]
        x_cov[:,:,i] /= theta_std[i]

    return x, x_cov


def prepare_data(d, mag, nn):
    """
    Predicts the data we need from the neural network and calculates the covariance matrix and it's
    inverse, which is needed for chi_sq,

    Parameters
    ----------
    d: np.ndarray
        Data-array which contains the theta, parallax, sources and their errors
    nn: keras.model
        The neural network from Greg, which we use to predict the magnitudes
    mag: np.ndarray
        Contains the magnitudes and their errors

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
    n_stars, n_bands = mag['mag'].shape
    dof = n_stars*(n_bands-2)
    large_err = 999.
    # define the partial model to predict BM
    inputs = nn.get_layer('theta').input
    outputs = nn.get_layer(name='BM').output
    B_M_model = tf.keras.Model(inputs, outputs)

    m = mag['mag'].copy()
    cov_m = np.zeros((n_stars, n_bands, n_bands))

    # fill the cov. matrix with the observed mag errors, replace nans with median and large errors
    for b in range(n_bands):
        cov_m[:,b,b] = mag['mag_err'][:,b]**2
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
    try:
        atm = d['atm_param_p']
        atm_cov = d['atm_param_cov_p']
    except:
        atm, atm_cov = normalize_theta(d['atm_param'],d['atm_param_cov'])

    R = predict_R(atm, nn)
    R_mean = np.mean(R, axis=0)
    R_mean /= np.dot(R_mean,R_mean)**0.5

    # calculate the gradient of predicting BM for the covariance matrix
    with tf.GradientTape() as g:
        x_p = tf.constant(atm)
        g.watch(x_p)
        mag_color = B_M_model([x_p])
    J = g.batch_jacobian(mag_color, x_p).numpy()

    cov_m += np.einsum('nik,nkl,njl->nij',J,atm_cov,J)

    # invert the covariance matrix
    C = cov_m.copy()
    for i, cov in enumerate(cov_m):
        C[i] = np.linalg.inv(cov)

    # subtract the observed reddened BM from predicted unreddened BM
    dm = B_M_model(atm).numpy()
    dm -= m

    return dm, C, R_mean, dof, cov_m


def angle(v1, v2):
    angle = np.arccos(np.dot(v1,v2)/(np.dot(v1,v1)**0.5*np.dot(v2,v2)**0.5))*180/np.pi
    return angle


def gram_schmitt(v, u):
    w = v - np.dot(v,u)/np.dot(u,u)*u
    return w


def stability_check(array, size, length):

    if len(array) < length:
        length = len(array)

    check = array[-length:]
    vary = np.std(check, axis=0)

    stab = (vary < size)
    return stab


def main():
    # the path where the data is saved
    dir = 'data/'
    model_path = dir + 'green2020_nn_model.h5'
    prefix = 'manybands_big_'  #'mock_seed20_small_temp_' #'green2020_small_'
    path = dir + prefix + 'data.h5'

    #TODO Maybe lower rho as iterations go on?
    rho = np.array([1e-1,1e-1,1e-1])
    l1, l2, l3 = [0], [0], [0]

    # resolve arguments for the unit tests; see tests for possible test
    parser = argparse.ArgumentParser(prog='Variation Fit')
    parser.add_argument('limit' ,default=1000, nargs='?', type=int,
                                     help='Max iterations, default is 1000')

    args = parser.parse_args()
    limit = args.limit

    # read out the data, Neural Network and prepare the data with the Neural Network
    data, mags ,r_mock, R_mock, dr_mock, dR_mock, V_mock = read_out(path)
    nn_model = tf.keras.models.load_model(model_path)
    dm, C, R, dof, cov_m = prepare_data(data, mags, nn_model)

    # prepare the list that saved the values for each iteration (!!we fit for BR/BdR!!)
    T = (data['atm_param'][:,0].copy()-6000)/1000
    BR_m = R.copy()
    BR_m[1:] -= BR_m[0]

    V = [np.zeros(13)]
    BR = [BR_m]
    BdR = [guess_dR(R)]
    dE = np.zeros(len(T))
    chi = []
    E_s = []
    dE_s = []
    twenty = np.zeros((limit,))

    if R_mock is not None:
        V_mock1, R_mock1, dR_mock1 = vary_mock(5,5,5,V_mock,R_mock,dR_mock)
        V = [V_mock1]
        BR = [R_mock1]
        BdR = [dR_mock1]
        [drm_dot, rm_dot, vm_dot, r_angle, dr_angle, v_angle] = (np.empty((limit,)) for i in range(6))


    dR = BdR[-1].copy()
    dR[1:] += dR[0]
    RR = BR[-1][None,:] + T[:,None]*V[-1][None,:]

    # initialize empty arrays
    [r_dot, r_len, dr_dot, dr_len, v_len] = (np.empty((limit,)) for i in range(5))

    for i in range(limit):
        # calculate E
        E = calculate_E(RR, dE, BdR[-1], C, dm)
        TE = T*E

        # calculate R
        BR_temp = calculate_R(BR[-1], E, BdR[-1], dE, C, dm, l1[-1], l3[-1], rho[::2], dof, V[-1], TE)
        BR.append(BR_temp)
        R = BR[-1].copy()
        R[1:] += R[0]
        r_dot[i] = np.dot(R,dR)
        r_len[i] = np.dot(R,R)**0.5

        # update lambda_3 and _4 according to the constraints and append
        l_1 = l3[-1] + constraint_ortho(BdR[-1],BR[-1],rho[0])
        l_3 = l3[-1] + constraint_unity(BR[-1], rho[2])
        l1.append(l_1)
        l3.append(l_3)

        # calculate V
        V_temp = calculate_R(V[-1], TE, BR[-1], E, C, dm, 0, 0, [0,0], dof, BdR[-1], dE)
        V.append(V_temp)
        RR = BR[-1][None,:] + T[:,None]*V[-1][None,:]

        # calculate dE
        dE = calculate_E(BdR[-1], E, RR, C, dm)

        # calculate dR
        BdR_temp = calculate_R(BdR[-1], dE, BR[-1], E, C, dm, l1[-1], l2[-1], rho[:2], dof, V[-1], TE)
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

        # Progress
        chi_t = np.sum(chi_sq(E, BR[-1], dE, BdR[-1], C, dm, TE, V[-1]))/dof
        chi.append(chi_t)
        if (i%(limit//20) == 0) or (i == (limit-1)):
            print(f'Iteration {i} Complete')
            E_s.append(E)
            dE_s.append(dE)
            twenty[i] = 1

        # if integration testing
        if R_mock is not None:
            if i == 0:
                R_mo = R_mock.copy()
                R_mo[1:] += R_mo[0]
                dR_mo = dR_mock.copy()
                dR_mo[1:] += dR_mo[0]
            rm_dot[i] = np.dot(R,R_mo)
            vm_dot[i] = np.dot(V[-1],V_mock)
            drm_dot[i] = np.dot(dR,dR_mo)
            v_len[i] = np.dot(V[-1],V_mock)/(np.dot(V_mock,V_mock)**0.5*np.dot(V[-1],V[-1]))
            r_angle[i] = angle(R,R_mo)
            dr_angle[i] = angle(dR, dR_mo)
            v_angle[i] = angle(V[-1], V_mock)


    # saving the data
    c = np.empty((len(r_dot),4))
    for idx, cx in enumerate([r_dot,r_len,dr_dot,dr_len]):
        c[:,idx] = cx

    saves = {'rho':rho, 'E':E_s, 'dE':dE_s, 'R':BR, 'dR':BdR, 'l1':l1, 'l2':l2, 'l3':l3, 'con':c,
            'C':C, 'chi':chi, 'cov':cov_m, 'V':V, 'dm':dm, 'save':twenty}


    if R_mock is not None:
        saves.update({'drm_dot':drm_dot, 'rm_dot':rm_dot, 'vm_dot':vm_dot,'v_len':v_len,
            'r_angle':r_angle, 'dr_angle':dr_angle, 'v_angle':v_angle})

    saving = dir + prefix + str(limit) +'its_result.h5'
    save(saving, saves)

    return 0

if __name__ == '__main__':
    main()

