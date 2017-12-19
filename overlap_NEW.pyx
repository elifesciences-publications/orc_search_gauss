#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np

def overlap(double[:] vec_norm, double[:,:] setA, double[:,:] setB, double[:] q0, double zrange, double dz):

    cdef int i, k
    cdef int nA = setA.shape[0]
    cdef int nB = setB.shape[0]
    cdef double overlap_n = 0
    cdef double z0 = np.dot(q0, vec_norm)
    cdef double zmin = z0 - zrange
    cdef double zmax = z0 + zrange
    cdef int znum = np.int((zmax - zmin)/dz + 1)
    cdef double[:] z = np.linspace(zmin, zmax, num=znum)
    cdef double[:] NzA = np.zeros(znum)
    cdef double[:] NzB = np.zeros(znum)

    cdef double[:] setA_norm = np.zeros(nA)
    cdef double[:] setB_norm = np.zeros(nB)
    for k in xrange(nA):
        setA_norm[k] = np.dot(setA[k], vec_norm)
    for k in xrange(nB):
        setB_norm[k] = np.dot(setB[k], vec_norm)

    for i in xrange(znum-1):
        for k in xrange(nA):
            if setA_norm[k] >= z[i] and setA_norm[k] < z[i+1]:
                NzA[i] += 1
        for k in xrange(nB):
            if setB_norm[k] >= z[i] and setB_norm[k] < z[i+1]:
                NzB[i] += 1

    NzA = NzA/np.sqrt(np.dot(NzA, NzA))
    NzB = NzB/np.sqrt(np.dot(NzB, NzB))

    for i in xrange(znum):
        overlap_n += dz*NzA[i]*NzB[i]

    return overlap_n

def z_projection(double[:] vec_norm, double[:,:] setA, double[:,:] setB, double[:] q0, double zrange, double dz):

    cdef int i, k
    cdef int nA = setA.shape[0]
    cdef int nB = setB.shape[0]
    cdef double z0 = np.dot(q0, vec_norm)
    cdef double zmin = z0 - zrange
    cdef double zmax = z0 + zrange
    cdef int znum = np.int((zmax - zmin)/dz + 1)
    cdef double[:] z = np.linspace(zmin, zmax, num=znum)
    cdef double[:] NzA = np.zeros(znum)
    cdef double[:] NzB = np.zeros(znum)

    cdef double[:] setA_norm = np.zeros(nA)
    cdef double[:] setB_norm = np.zeros(nB)
    for k in xrange(nA):
        setA_norm[k] = np.dot(setA[k], vec_norm)
    for k in xrange(nB):
        setB_norm[k] = np.dot(setB[k], vec_norm)

    for i in xrange(znum-1):
        for k in xrange(nA):
            if setA_norm[k] >= z[i] and setA_norm[k] < z[i+1]:
                NzA[i] += 1
        for k in xrange(nB):
            if setB_norm[k] >= z[i] and setB_norm[k] < z[i+1]:
                NzB[i] += 1

    NzA = NzA/np.sqrt(np.dot(NzA, NzA))
    NzB = NzB/np.sqrt(np.dot(NzB, NzB))

    return (z, NzA, NzB)

def variance1(double[:] vec_norm, double[:,:] setP):

    cdef int k
    cdef int nP = setP.shape[0]

    cdef double[:] setP_norm = np.zeros(nP)
    cdef double var
    for k in xrange(nP):
        setP_norm[k] = np.dot(setP[k], vec_norm)

    var = np.var(setP_norm)

    return var

def variance2(double[:] vec_norm, double[:,:] setA, double[:,:] setB):

    cdef int k
    cdef int nA = setA.shape[0]
    cdef int nB = setB.shape[0]

    cdef double[:] setA_norm = np.zeros(nA)
    cdef double[:] setB_norm = np.zeros(nB)
    cdef double[:] setAB_norm = np.zeros(nA + nB)
    cdef double var
    for k in xrange(nA):
        setA_norm[k] = np.dot(setA[k], vec_norm)
    for k in xrange(nB):
        setB_norm[k] = np.dot(setB[k], vec_norm)

    setAB_norm = np.concatenate((setA_norm, setB_norm))
    var = np.var(setAB_norm)

    return var
