#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a mock data catalogue so we can test the method and see if we can recover these values.
"""

import h5py
import json
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf


def read_out(path):
    # read out the existing Test file supplied by Greg's Paper
    with h5py.File(path, 'r') as f:
        d = f['data'][:]
        r = f['r_fit'][:]

    return d, r


def save(path, d):
    # save the mock data set with the used R, E, dR, dE
    with h5py.File(path, 'w') as f:
        for key in d.keys():
            f.create_dataset(key, data=d[key], chunks=True, compression='gzip', compression_opts=3)


def guess_dR(R, file='mean'):
    # read out the mean wavelengths of the 13 filters we use
    with open(file+'_wavelength.json', 'r') as f:
        mean = json.load(f)

    # inverted parabula peaking around 1 mikrometer
    x = np.array(mean)
    l = 1/(x*1e-4)
    print(l)
    p = -np.abs(l-1.3)+1
    #p = -(l-1.15)**2+2

    # Gram-Schmidt process
    y = p - (np.dot(R,p)/np.dot(R,R))*R

    # norm it to 1 and transform it to mag/color space
    y /= np.dot(y,y)**0.5

    return y


def predict_M(theta, nn):
    # define the partial Neural Network that predicts only BM
    inputs = nn.get_layer('theta').input
    outputs = nn.get_layer('BM').output
    BM_model = tf.keras.Model(inputs, outputs)

    # return the prediction of BM
    return BM_model(theta).numpy()


def predict_R(theta, nn):
    # define the partial Neural Network that predicts only R
    inputs = nn.get_layer('theta').input
    outputs = nn.get_layer('R').output
    R_model = tf.keras.Model(inputs, outputs)

    # return the prediction of R
    return R_model(theta).numpy()


def mock_m(d, r, nn, seed = 14, max = 0.1):
    """
    Calculates a mock magnitude by predicting absolute magnitude and add Reddening as well as
    Variation manually to create a mock magnitude.

    Parameters
    ----------
    d: np.ndarray
        Contains the readout data from the Greg's test data with parallax, theta and mag
    r: np.array
        Contains the reddening E for each star from Greg's test data
    nn: keras.model
        The neural Network supplied with Greg's test data
    seed: int
        RNG-Seed to reproduce dE values if needed
    max: float
        Max value for dE

    Returns
    -------
    BR: np.array
        The general reddening vector we obtained from the Neural Network
    dE: np.array
        The reddening of each star of the test data set, which we randomly generated
    BdR: np.array
        The general reddening variation vector we obtained from guess_dR
    """
    # set rng seed to be able to reproduce results
    rng = np.random.default_rng(seed)

    # determine the amount of bands and stars measured, should be 13 bands and create the B-matrix
    n_stars, n_bands = d['mag'].shape
    B = np.identity(n_bands, dtype='f4')
    B[1:,0] = -1.
    B_inv = B.copy()
    B_inv[1:,0] = 1

    # predict R and take the mean to get a general reddening vector and transform it into mag/color
    R_all = predict_R(d['atm_param_p'], nn)
    R = np.mean(R_all, axis=0)
    r *= np.dot(R,R)**0.5
    R /= np.dot(R,R)**0.5
    BR = np.einsum('ij,j->i',B,R)

    # predict m from theta and take parallax and reddening into account for m
    m = predict_M(d['atm_param_p'], nn)
    m = np.einsum('ij,nj->ni',B_inv,m)
    m[:,:] += (10. - 5.*np.log10(d['parallax']))[:,None]
    m[:,:] += r[:,None] * R[None,:]

    # adding the temperature variance
    T = (d['atm_param'][:,0].copy()-6000)/1000
    V = rng.normal(0,0.1,13)  #np.var(R_all, axis=0)
    BV = np.einsum('ij,j->i',B,V)
    print(np.dot(BV,BV)**0.5)
    m[:,:] += r[:,None] * T[:,None] * V[None,:]

    # guess BdR and generate random dE to add them to the mock m
    dR = guess_dR(R)
    #dR[1:] = rng.normal(0,0.5,12)
    #dR[12] = 0.01 
    #dR[0] = -np.dot(dR[1:],R[1:])/R[0]
    #dR /= np.dot(dR,dR)**0.5
    print(dR)
    print(R)
    print(np.dot(dR,R))
    BdR = np.einsum('ij,j->i',B,dR)
    dE = rng.normal(0,max,(n_stars,))
    m[:,:] += dE[:,None] * dR[None,:]
    d['mag'] = m

    # make sure we have an error for every magnitude
    m_err = d['mag_err'].copy()

    for b in range(n_bands):
        idx = ~np.isfinite(m_err[:,b])
        idx2 = (m_err[:,b] < 0.08)
        sample = m_err[idx2]
        replace = rng.choice(sample, np.sum(idx))
        m_err[idx] = replace

    d['mag_err'] = m_err

    return BR, dE, BdR, BV


def noise(d, par_gauss = False, seed = 14):
    """
    Takes a mock dataset with artificial magnitudes and adds noise to all observed qunatities.
    This includes Stellar Parameter, Magnitudes and distances. For the last one we include the
    option to add gaussian noise on the distance or the parallax (which doesn't transform 1 to 1).

    Parameters
    ----------
    d: np.ndarray
        Data-array which contains the the theta, mag, parllax and their errors
    par_gauss: Boolean
        False by default, means we add the noise to the distance and not the parllax
    seed: int
        The rng seed, set to 14 as default.

    Returns:
    --------
    d1: np.ndarray
        The noisy data
    d0: np.ndarray
        The true value for the noisy data
    """
    #
    rng = np.random.default_rng(seed)
    d1 = d.copy()

    # create array where the noiseless values are saved
    dtype = [
        ('mag','13f4'),
        ('atm_param_p','3f4'),
        ('parallax','f4')
    ]

    d0 = np.empty(d.size, dtype=dtype)

    for type, _ in dtype:
        d0[type] = d1[type]
        norm = rng.normal(size=d1[type].shape)
        err = type + '_err'
        if type == 'atm_param_p':
            for k in range(3):
                d1[type][:,k] += norm[:,k]*d1['atm_param_cov_p'][:,k,k]
        elif (not par_gauss) & (type =='parallax'):
            dm = (10. - 5.*np.log10(d1[type]))
            dm += norm*(5./np.log(10))*(d1[err]/d1[type])
            d1[type] = 10**(2-0.2*dm)
            print('Adding gaussian error in the distance')
        else:
            d1[type] += norm*d[err]

    check = (d1['parallax'] < 0)
    print(np.sum(check), ' parallaxes went negative')

    return d0, d1


def main():
    # the folder location where the datasets are saved
    dir = 'data/'
    big = 1
    seed = np.random.default_rng().integers(1,1000)
    seed = 20
    # the actual file names
    base = dir + 'green2020_small_data.h5'
    mock = dir + 'mock_seed' + str(seed) + '_small_temp_data.h5'
    file = dir + 'green2020_nn_model.h5'

    # load the Neural Network and read out the values from the test dataset
    nn_model = tf.keras.models.load_model(file)
    d, r = read_out(base)

    #set up a mask to filter bad parallaxes for our mock data
    err_over_plx = d['parallax_err']/d['parallax']
    idx = ~(
          (err_over_plx > 0.2)
        | (d['parallax'] < 1.e-8)
        | ~np.isfinite(d['parallax'])
        | ~np.isfinite(d['parallax_err'])
            )

    print('We filter out ', sum(~idx), ' and use a total of ', sum(idx), ' Stars')
    d = d[idx]
    r = r[idx]

    # calculate R, dR and dE and replace mag and mag_err in the dataset
    R, dr, dR, V = mock_m(d, r, nn_model, seed = seed)
    d = np.tile(d, big)
    r = np.tile(r, big)
    dr = np.tile(dr, big)
    d0, d1 = noise(d)
    print('The mock catalogue contains ', len(dr), ' Stars')

    saving = {'data':d1,'r_fit':r,'R':R,'dR':dR, 'dr_fit':dr, 'V':V, 'data_no_noise':d0}
    save(mock, saving)

    return 0


if __name__ == '__main__':
    main()
