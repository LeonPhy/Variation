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


def save(path, d, d0, r, R, dr, dR):
    # save the mock data set with the used R, E, dR, dE
    with h5py.File(path, 'w') as f:
        f.create_dataset('/data', data=d, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/r_fit', data=r, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/R', data=R, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dr_fit', data=dr, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/dR', data=dR, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('/data_no_noise', data=d0, chunks=True, compression='gzip', compression_opts=3)


def guess_dR(R):
    # read out the mean wavelengths of the 13 filters
    with open('mean_wavelengths.json', 'r') as f:
        mean = json.load(f)

    x = np.array(mean)

    # calculate the offset of the linear scaling to garnatuee it is perpendicular to R
    b = np.sum(x*R)/np.sum(R)
    y = x[:] - b[None]

    # add a factor to scale the reddening vector so our initial guess is not the same
    y *= 0.4
    # norm it to 1
    y[1:] -= y[0]
    y /= np.dot(y,y)**0.5
    y[1:] += y[0]

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
    BR = np.einsum('ij,j->i',B,R)

    # predict m from theta and take parallax and reddening into account for m
    m = predict_M(d['atm_param_p'], nn)
    m = np.einsum('ij,nj->ni',B_inv,m)
    m[:,:] += (10. - 5.*np.log10(d['parallax']))[:,None]
    m[:,:] += r[:,None] * R[None,:]

    # guess BdR and generate random dE to add them to the mock m
    dR = guess_dR(R)
    BdR = np.einsum('ij,j->i',B,dR)
    dE = max*rng.random((n_stars,))
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

    return BR, dE, BdR


def noise(d, seed = 14):
    """

    """
    #
    rng = np.random.default_rng(seed)

    # create array where the noiseless values are saved
    dtype = [
        ('mag','13f4'),
        ('atm_param_p','3f4'),
        ('parallax','f4')
    ]

    d0 = np.empty(d.size, dtype=dtype)

    for type, _ in dtype:
        d0[type] = d[type]
        norm = rng.normal(size=d[type].shape)
        if type == 'atm_param_p':
            for k in range(3):
                d[type][:,k] += norm[:,k]*d['atm_param_cov_p'][:,k,k]
        else:
            err = type + '_err'
            d[type] += norm*d[err]

    check = (d['parallax'] < 0)
    print(np.sum(check), ' parallaxes went negative')

    return d0


def main():
    # the folder location where the datasets are saved
    dir = 'data/'
    # the actual file names
    base = dir + 'green2020_test_data_small.h5'
    mock = dir + 'mock_data.h5'
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

    d = d[idx]
    r = r[idx]

    # calculate R, dR and dE and replace mag and mag_err in the dataset
    R, dr, dR = mock_m(d, r, nn_model)
    d0 = noise(d)
    save(mock, d, d0, r, R, dr, dR)

    return 0


if __name__ == '__main__':
    main()
