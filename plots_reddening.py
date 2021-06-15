#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting the results from the Reddening Variation Script
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py


def read_out(path, result = 'result.h5'):
    # read out the data set
    path1 = path + 'data.h5'
    path2 = path + result
    data = {}
    keys = ['save','E', 'dE', 'R', 'dR', 'l1', 'l2', 'l3', 'con', 'C', 'chi',
                'rho', 'cov', 'V', 'dm']
    mock = ['drm_dot','rm_dot','vm_dot','v_len','r_angle','dr_angle','v_angle']

    with h5py.File(path2, 'r') as f:
        for key in keys:
            data[key] = f[key][:]
        if "mock" in path:
            for key in mock:
                data[key] = f[key][:]

    with h5py.File(path1, 'r') as f:
        data['d'] = f['data'][:]

    return data


def result_hist(x, iter, label, prefix, xlim = None):
    """
    x: np.array
        The value that are displayed in the histogram

    """
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
    plt.close(fig)

def lambda_plot(l, label, prefix):
    """

    """
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
    plt.close(fig)

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


def red_teff(E,R,V,dE,dR,teff, band):
    a = E[:,None]*(R[None,:]+V[None,:])
    da = dE[:,None]*dR[None,:]
    A = a+da
    dE_E = dE/E
    idx = (E > 0.1)
    bands = ['G', 'BP', 'RP'] + list('grizyJHK') + ['W1', 'W2']
    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,14), facecolor = 'white')
    [[ax1, ax2], [ax3, ax4]] = fig.subplots(2,2,sharex=True,gridspec_kw={'hspace':0,'wspace':0.3})

    x = [
            (ax1,dE_E[idx],teff[idx],'dE over E'),
            (ax2,A[:,band],teff,'dA+A in band '+str(bands[band])),
            (ax3,a[:,band],teff,'A in band '+str(bands[band])),
            (ax4,da[:,band],teff,'dA in band '+str(bands[band]))
        ]


    for ax, x_value, y_value, x_label in x:
        correlation_plot(ax,y_value,x_value)
        ax.set_ylabel(x_label)

    ax3.set_xlabel('effective temperature')
    ax4.set_xlabel('effective temperature')
    picp = 'pictures/extinction_over_eff_temp_band'+str(bands[band])
    plt.savefig(picp, dpi=150, bbox_inches='tight')
    plt.close(fig)


def component_plot(r, label):
    it, n = r.shape
    bands = ['G', 'BP', 'RP'] + list('grizyJHK') + ['W1', 'W2']
    if len(bands) != n:
        bands = np.linspace(1,n,n,dtype=int)
    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,14), facecolor = 'white')
    ax = fig.subplots(1,1)

    for i, band in enumerate(bands):
        ax.plot(range(it),r[:,i])
        ax.text(1.05*it, r[-1,i],f'{band}')


    ax.set_ylabel(label)
    ax.set_xlabel('Iterations')
    ax.grid(True)
    picp = 'pictures/'+ label.replace(' ', '_') + '_over_iterations'
    plt.savefig(picp, dpi=150, bbox_inches='tight')
    plt.close(fig)


def obj(chi, l1, l2, l3, rho, plots):
    it = len(chi)
    pen1 = 0.5*(np.diff(l1[::2]))**2/rho[0]
    pen2 = 0.5*(np.diff(l2))**2/rho[1]
    pen3 = 0.5*(np.diff(l3))**2/rho[2]

    con1 = l1[:-2:2]*np.diff(l1[::2])/rho[0]
    con2 = l2[:-1]*np.diff(l2)/rho[1]
    con3 = l3[:-1]*np.diff(l3)/rho[2]

    obj = chi + con1 + con2 + con3 + pen1 + pen2 + pen3

    if plots:
        parts = [pen1, pen2, pen3, con1 , con2, con3]
        all = parts.copy()
        all.append(chi)
        all.append(obj)
        parts = np.array(parts).T
        all = np.array(all).T
        component_plot(all, 'Objective function and parts')
        component_plot(parts, 'Parts of objective function')

    """
    plt.rcParams.update({'font.size':24})
    fig = plt.figure(figsize=(20,14), facecolor = 'white')
    ax1, ax2 = fig.subplots(1,2)
    ax1.set_ylabel('Objective functions')
    ax1.set_xlabel('Iteration')
    ax.grid(True)
    pic = label.replace(' ', '_')
    picp = 'pictures/' + prefix +pic
    plt.savefig(picp, dpi = 150, bbox_inches='tight')
    plt.close(fig)
    """
    return obj


def stability_check(og_array, size = 0.05, length=100):

    if len(og_array)-1 < length:
        return False


    array = np.diff(og_array)

    check1 = array[-length:-length//2]
    check2 = array[-length//2:]

    mean1 = np.mean(check1)
    mean2 = np.mean(check2)

    vary = np.std(check1+check2, axis=0)

    combine = np.abs(mean1 - mean2)/vary
    #print(combine)

    stab = (size > combine)

    return stab


def criterion(x):
    length = x.size
    x0 = np.mean(x[:length//2])
    x1 = np.mean(x[-length//2:])
    sigma = np.std(x)
    norm = 4*sigma**2/length
    return (x1-x0)**2/norm


def main():
    dir = 'data/'
    prefix = 'mock_seed20_small_temp_' #'green2020_small_'
    path = dir + prefix
    plots = []

    d = read_out(path, '500000its_result.h5')

    R = d['R']
    V = d['V']
    dR = d['dR']
    E = d['E'][-1]
    dE = d['dE'][-1]


    for dd in (R,V,dR):
        for i in range(len(dd[:,0])):
            dd[i,1:] += dd[i,0]


    Obj = False
    Stop = False
    Lam = False
    Con = False
    Chi = True
    Diff = False
    Comp = False
    Mock = False

    E_hist = False
    temp_res = False
    ex_bands = []


    #entangle obj calc and making the plot
    obje = obj(d['chi'],d['l1'],d['l2'],d['l3'],d['rho'], Obj)

    num = [str(s) for s in range(1,14)]
    R_num = ['R'+s for s in num]
    dR_num = ['dR'+s for s in num]
    V_num = ['V'+s for s in num]
    names = ['obj','l1','l2','l3'] + R_num + dR_num + V_num

    R_ch = [R[:,s] for s in range(13)]
    dR_ch = [dR[:,s] for s in range(13)]
    V_ch = [V[:,s] for s in range(13)]
    checks = [obje, d['l1'][::2], d['l2'], d['l3']] + R_ch + dR_ch + V_ch

    if Stop:
        for name, ar in zip(names,checks):
            length = 500
            stop = []
            for ob in range(2*length,(len(d['l2'])),length):
                stop.append(criterion(ar[(ob-length):ob]))

            lambda_plot(stop,name+' criterion',prefix)

    plot = []

    #maybe make dedicated methods for drawing these plots?
    if Lam:
        Lam = [
            (d['l1'], 'Lambda 1 (Ortho)'),
            (d['l2'], 'Lambda 2 (Unity dR)'),
            (d['l3'], 'Lambda 3 (Unity R)')]

        plot = plot + Lam
        component_plot(np.array((d['l1'][1::2],d['l1'][2::2])).T,'lambda 1')


    if Con:
        Con = [
            (d['con'][:,2], 'Ortho after dR'),
            (d['con'][:,3], 'Unity dR'),
            (d['con'][:,0], 'Ortho after R'),
            (d['con'][:,1], 'Unity R')]

        plot = plot + Con


    if Chi:
        Chi = [
            (d['chi'][-10000:], 'Chi Square')]
            #(d['chi'][200:], 'Zoom 200 Chi Square')]

        plot = plot + Chi


    if Diff:
        Diff = [
            (np.diff(d['l2'][1500:]),'Lambda 2 diff'),
            (np.diff(d['l3'][1000:]),'Lambda 3 diff'),
            (np.diff(d['chi'][1500:]),'Chi diff'),
            (np.diff(obje[1500:]),'Objective diff')]

        plot = plot + Diff
        component_plot(np.diff(R[-5000:],axis=0),'reddening diff')
        component_plot(np.diff(dR[-5000:],axis=0),'reddening variation diff')
        component_plot(np.diff(V[-5000:],axis=0),'temperature diff')



    if Mock:
        Mock = [
            (d['r_angle'],'Angle between R and R mock'),
            (d['dr_angle'],'Angle between dR and dR mock'),
            (d['v_angle'],'Angle between V and V mock'),
            (d['drm_dot'],'Dot between dR and Mock dR'),
            (d['rm_dot'],'Dot between R and Mock R')]

        plot = plot + Mock


    if E_hist:
        result_hist(E,0,'E',prefix,(-2,6))
        result_hist(dE,0,'dE',prefix,(-1,1))

    if temp_res:
        temp_res(d['d'], E, dE, d['R'][-1,:], d['dR'][-1,:], d['dm'], d['cov'])

    for i in ex_bands:
        red_teff(E,R[-1],V[-1],dE,dR[-1],d['d']['atm_param'][:,0],i)

    if Comp:
        #component_plot(R[-200000::10],'reddening')
        #component_plot(dR[-200000::10],'reddening variation')
        #component_plot(V[-200000::10],'temperature dependency')
        for ar, name in zip(checks, names):
            lambda_plot(ar[-200000::10],name+' component',prefix)

    for arr, n in plot:
        lambda_plot(arr,n,prefix)

    return 0


if __name__ == '__main__':
    main()

