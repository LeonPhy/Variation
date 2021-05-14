#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculates the average wavelengths of the passbands.dat (data/passbands) in Angstrom
"""


from astropy.io import ascii
import numpy as np
import json
from glob import glob

def main():
    colors = glob('data/passbands/*.dat')
    #Sorting them by alphabet
    colors.sort()
    print(colors)
    #Manually sorted to fit the order of the data
    idx = [3,4,5,6,8,7,10,9,1,0,2,11,12]
    sort = colors.copy()

    for i,j in enumerate(idx):
        colors[i] = sort[j]

    mean = []
    width = []
    div = []

    #Read out passband wavelengths and strength and calculate mean
    print('Band               Mean Wavelength')
    print('----------------------------------')
    for i, b in enumerate(colors):
        x, y = [], []
        data = ascii.read(b)
        for row in range(len(data)):
            xt, yt = data[row]
            x.append(xt)
            y.append(yt)

        x = np.array(x)
        y = np.array(y)
        max = np.max(y)
        x_cut = x[np.where(y > max/2)]
        width.append(float(x_cut[-1] - x_cut[0]))
        mean.append(np.sum(x*y)/np.sum(y))
        div.append((mean[i]/width[i])**2)
        print(b[24:-4], mean[i], width[i], div[i])

    files = [('mean_wavelength.json', mean),('width_wavelength.json', width),
        ('div_wavelength.json', div)]

    for file, a in files:
        with open(file, 'w') as f:
            json.dump(a, f)


    return 0

if __name__ == '__main__':
    main()
