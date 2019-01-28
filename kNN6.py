#!/usr/bin/env python
import numpy

if __name__ == "__main__":

    NDIM = 4 # number of dimensions

    # read points into array
    a = numpy.fromfile('./results/million_3D_points.txt', sep=' ')
    a.shape = int(a.size / NDIM), NDIM

    point =  [ 69.06310224,   2.23409409,  50.41979143, 50.41979143] # use the same point as above
    print('point:', point)


    from scipy.spatial import KDTree

    # find 10 nearest points
    tree = KDTree(a, leafsize=a.shape[0]+1)
    distances, ndx = tree.query([point], k=10)

    # print 10 nearest points to the chosen one
    print(a[ndx])
