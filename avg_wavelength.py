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
    #Manually sorted to fit the order of the data
    idx = [3,4,5,6,8,7,10,9,0,1,2,11,12]
    sort = colors.copy()

    for i,j in enumerate(idx):
        colors[i] = sort[j]

    mean = []

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
        mean.append(np.sum(x*y)/np.sum(y))
        print(b[24:-4], mean[i])

    file = 'mean_wavelengths.json'
    with open(file, 'w') as f:
        json.dump(mean, f)

    return 0

if __name__ == '__main__':
    main()
