#!/usr/bin/env python
import numpy


if __name__ == "__main__":
    NDIM = 4 # number of dimensions

    # read points into array
    a = numpy.fromfile('./results/million_3D_points.txt', sep=' ')
    a.shape = int(a.size / NDIM), NDIM

    point =  [ 69.06310224,   2.23409409,  50.41979143, 50.41979143] # use the same point as above
    print('point:', point)
    d = ((a-point)**2).sum(axis=1)  # compute distances
    ndx = d.argsort() # indirect sort 

    # print 10 nearest points to the chosen one
    print(a[ndx[:10]], d[ndx[:10]])
