#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting the results from the Reddening Variation Script
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py


def read_out(path):
    # read out the data set
    path1 = path + 'data.h5'
    path2 = path + 'result.h5'
    with h5py.File(path1, 'r') as f:
        d = f['data'][:]

    with h5py.File(path2, 'r') as f:
        E = f['E'][:]
        dE = f['dE'][:]
        R = f['R'][:]
        dR = f['dR'][:]
        l = f['l'][:]
        dm = f['dm'][:]
        C = f['C'][:]
        chi = f['chi'][:]
        con = f['con'][:]

    return d, E, dE, R, dR, l, dm, C, chi, con


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
    picp = 'pictures/'+ prefix + pic +'_iteration_' + str(iter)
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
    picp = 'pictures/' + prefix +pic
    plt.savefig(picp, dpi = 150, bbox_inches='tight')


def main():
    dir = 'data/'
    prefix = 'green2020_small_'
    path = dir + prefix

    d, E, dE, R, dR, l, dm, C, chi, c = read_out(path)


    plots = [(l[:,0], 'Lambda 1 (Ortho dR)'), (l[:,2], 'Lambda 3 (Ortho R)'),
                (l[:,1], 'Lambda 2 (Unity dR)'), (l[:,3], 'Lambda 4 (Unity R)'),
                (c[:,2], 'Ortho after dR'), (c[:,3], 'Unity dR'),
                (c[:,0], 'Ortho after R'), (c[:,1], 'Unity R'), (chi, 'Chi Sqaure')]

    for (arr, text) in plots:
        lambda_plot(arr, text, prefix)

    return 0


if __name__ == '__main__':
    main()

