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
        cov = f['cov'][:]

    return d, E, dE, R, dR, l, dm, C, chi, con, cov


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


def correlation_plot(ax, x, y, xbins = 25, ybins = 20,
                 y_abs = None, y_pct = 99., pct = [15.87, 50., 84.13]):
    """
    Makes a Hogg-style correlation plot between x  and y, returns bins and their weights.

    Parameter
    ---------
    ax: matplotlib.pyplot.axis
        The axis on which we draw the correlation plot
    x: np.array
        The x array must be same dimension as y
    y: np.array
        The y array, must be same dimension as x
    xbins: int or np.array
        If int, it's the amount of xbins, if given an array, it's the bin borders
    ybins: int or np.array
        If int, it's the amount of ybins, if given an array, it's the bin borders
    y_abs: float
        The absolute max value, if this is set to None, y_pct instead to determine max y
    y_pct: float
        The perecentile max value of y that we still use, only used if y_abs is None
    pct: list of floats
        The percentiles on which lines are drawn, corresponds to mean, and +/- 1 sigma
    """

    #Determine limits on y
    if y_abs is None:
        ymax = 1.1 * np.percentile(np.abs(y), y_pct)
    else:
        ymax = y_abs

    #bins
    if not hasattr(xbins, '__len__'):
        xbins = np.linspace(np.min(x), np.max(x), xbins+1)
    if not hasattr(ybins, '__len__'):
        ybins = np.linspace(-ymax, ymax, ybins+1)

    n_bins = (len(xbins)-1, len(ybins)-1)
    density = np.zeros(n_bins, dtype='f4')
    thresholds = np.zeros((n_bins[0],3), dtype='f4')

    #Density
    for n,(x0,x1) in enumerate(zip(xbins[:-1],xbins[1:])):
        idx = (x >= x0) & (x <x1)

        if np.any(idx):
            y_sel = y[idx]
            thresholds[n,:] = np.percentile(y_sel, pct)
            density[n,:], _ = np.histogram(
                y_sel,
                bins=ybins,
                density=False,
                range=[-ymax, ymax])

    a = np.max(density, axis=1)
    a[a == 0] = 1
    hi_des = density.copy()
    density /= a[:,None]

    idx = ~np.isfinite(density)
    density[idx] = 0

    #Plot density
    extent = (xbins[0], xbins[-1], ybins[0], ybins[-1])

    ax.imshow(
        density.T,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='gray_r',
        interpolation='nearest',
        vmin=0.,
        vmax=1.)

    ax.axhline(y=0, c='red', ls='-', alpha=0.8)

    #Plot pct
    xrange = np.linspace(xbins[0], xbins[-1], n_bins[0]+1)
    for i in range(3):
        y_env = np.hstack([[thresholds[0,i]], thresholds[:,i]])
        ax.step(xrange, y_env, where='pre', c='blue', linewidth=2, alpha=1.)

    #Set limits
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(axis='y', right=True, labelright=False)
    ax.tick_params(axis='x', top=True, labeltop=False)

    return xbins, np.sum(hi_des, axis=1), ybins,  np.sum(hi_des, axis=0)


def temp_res(d, E, dE, R, dR, dm, cov, Bands = [(4,0), (4,), (8,), (11,)]):
    bands = ['G', 'BP', 'RP'] + list('grizyJHK') + ['W1', 'W2']
    xfields = [
                (
                    d['atm_param'][:,0],
                    r'temp',
                    (3500., 8500.)
                ),
                (
                    d['atm_param'][:,1],
                    r'logg',
                    (0.6,6.4)
                ),
                    (d['atm_param'][:,2],
                    r'feh',
                    (-2.5,0.5)
                ),
                (
                    E[:],
                    r'E',
                    (0,3)
                ),
                (
                    dE[:],
                    r'dE',
                    (-0.5,0.5)
                )
            ]
    n_stars, n_bands = d['mag'].shape
    diff = dm + E[:,None]*R[None,:] + dE[:,None]*dR[None,:]

    for i, b in enumerate(Bands):
        if len(b) == 1:
            Bands[i] += (b[0] + 1,)

    for (i,j) in Bands:
        if j == 0:
            obs = cov[:,i,i] < 100
            c = cov[:,i,i]
            dy = diff[:,i]/np.sqrt(c)
        else:
            obs = (cov[:,i,i] < 100) & (cov[:,j,j] < 100)
            c = cov[:,i,i] + cov[:,j,j] - 2*cov[:,i,j]
            dy = (diff[:,i] - diff[:,j])/np.sqrt(c)

        ylabel = f'{bands[i]}-{bands[j]}'
        y_bins = np.arange(-5,5.001, 0.2)
        for x,xlabel,xlim in xfields:
            idx = np.isfinite(x) & np.isfinite(dy) & obs
            if xlabel == xfields[0][1]:
                outside = np.around(np.sum(((dy[idx] <= -5) | (dy[idx] >= 5.001)))/len(dy[idx]),4)
                suggest = np.percentile(dy[idx], [1,99])
                print('                ')
                print(f'outside of  y-limits -5 to 5: {outside}')
                print(f'1%: {suggest}')

            print('--------------------------')
            x_bins = np.linspace(xlim[0], xlim[1], 26)
            outside = np.around(np.sum(((x[idx] <= xlim[0]) | (x[idx] >= xlim[1])))/len(x[idx]),4)
            suggest = np.percentile(x[idx], [1,99])
            print(f'outside of {xlabel}-limits {xlim[0]} to {xlim[1]}: {outside}')
            print(f'1% : {suggest}')

            plt.rcParams.update({'font.size':24})
            fig = plt.figure(figsize=(20,14), facecolor = 'white')
            ax = fig.subplots(1,1)
            correlation_plot(ax,x[idx],dy[idx], xbins=x_bins, ybins=y_bins)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            picp = f'pictures/{bands[i]}-{bands[j]}_{xlabel}'
            plt.savefig(picp, dpi=150, bbox_inches='tight')
            plt.close(fig)


def red_teff(E,R,dE,dR,teff):
    a = E[:,None]*R[None,:]
    da = dE[:,None]*dR[None,:]
    A = a+da
    dE_E = dE/E
    idx = (E > 0.1)
    x = [
            (dE_E[idx],teff[idx],'dE over E'),
            (A[:,0],teff,'dA+A'),
            (a[:,0],teff,'A'),
            (da[:,0],teff,'dA')
        ]


    for x_value, y_value, x_label in x:
        plt.rcParams.update({'font.size':24})
        fig = plt.figure(figsize=(20,14), facecolor = 'white')
        ax = fig.subplots(1,1)
        correlation_plot(ax,y_value,x_value)
        ax.set_xlabel('effective temperature')
        ax.set_ylabel(x_label+' in G-band')
        picp = 'pictures/'+ x_label.replace(' ','_') + '_over_eff_temp'
        plt.savefig(picp, dpi=150, bbox_inches='tight')
        plt.close(fig)


def component_plot(R, label):
    it, n = R.shape
    B_i = np.identity(n, dtype='f4')
    B_i[1:,0] = 1
    r = np.einsum('ij,nj->ni',B_i,R)
    bands = ['G', 'BP', 'RP'] + list('grizyJHK') + ['W1', 'W2']
    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,14), facecolor = 'white')
    ax = fig.subplots(1,1)

    for i, band in enumerate(bands):
        ax.plot(range(it),r[:,i])
        ax.text(1.05*it, r[-1,i],f'{band}')


    ax.set_ylabel(label)
    ax.set_xlabel('Iterations')
    picp = 'pictures/'+ label.replace(' ', '_') + '_over_iterations'
    plt.savefig(picp, dpi=150, bbox_inches='tight')
    plt.close(fig)



def main():
    dir = 'data/'
    prefix = 'green2020_small_'
    path = dir + prefix

    d, E, dE, R, dR, l, dm, C, chi, c, cov = read_out(path)

    plots = [(l[:,0], 'Lambda 1 (Ortho dR)'), (l[:,2], 'Lambda 3 (Ortho R)'),
                (l[:,1], 'Lambda 2 (Unity dR)'), (l[:,3], 'Lambda 4 (Unity R)'),
                (c[:,2], 'Ortho after dR'), (c[:,3], 'Unity dR'),
                (c[:,0], 'Ortho after R'), (c[:,1], 'Unity R'), (chi, 'Chi Sqaure')]

    for (arr, text) in plots:
        pass #lambda_plot(arr, text, prefix)

    result_hist(E[-1,:],0,'E',prefix,(0,4))
    result_hist(dE[-1,:],0,'dE',prefix,(-1,1))

    temp_res(d, E[-1,:], dE[-1,:], R[-2,:], dR[-2,:], dm, cov)
    red_teff(E[-1,:],R[-1,:],dE[-1,:],dR[-1,:],d['atm_param'][:,0])
    component_plot(R,'reddening')
    component_plot(dR,'reddening variation')


    return 0


if __name__ == '__main__':
    main()

