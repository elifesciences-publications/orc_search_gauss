#!/home/migaev/.linuxbrew/bin/python

import numpy as np
from scipy.optimize import fmin
from scipy.spatial.distance import pdist
from scipy.linalg import qr
from random import gauss
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from overlap_NEW import overlap
from overlap_NEW import z_projection
from overlap_NEW import variance1
from overlap_NEW import variance2

from optparse import OptionParser


####################
# PARAMETER PARSING
####################
parser = OptionParser()
parser.add_option("--locpca", "--input_locpca", action="store", dest="locpca_switch", help="", default="no")
parser.add_option("--orc", "--input_orc", action="store", dest="orc_switch", help="", default="no")
parser.add_option("--pdb", "--input_pdb", action="store", dest="pdb_switch", help="", default="no")
parser.add_option("--dim", "--input_dim", action="store", dest="dim", help="", default="3")
parser.add_option("--fit", "--input_fit", action="store", dest="fit_switch", help="", default="no")
parser.add_option("--proj", "--input_proj", action="store", dest="proj_switch", help="", default="yes")
parser.add_option("--proj_orc", "--input_proj_orc", action="store", dest="proj_orc_switch", help="", default="no")
parser.add_option("--sel_orc", "--input_sel_orc", action="store", dest="sel_orc_switch", help="", default="no")
parser.add_option("--plot", "--input_plot", action="store", dest="plot_switch", help="", default="yes")
parser.add_option("--trajx", "--input_trajx", action="store", dest="trajx_switch", help="", default="no")
opt,args = parser.parse_args()


#################
# IMPORTING DATA
#################
fileP = 'PDB_projT.xvg'
fileA = '3JAT_projT.xvg'
fileB = '5JQG_projT.xvg'

m = 20
N = int(opt.dim)

fh = open(fileP)
cont = fh.readlines()
fh.close()

cont_new = []
for line in cont:
    if line.isspace(): continue
    if line[0] == '&': continue
    if line[0] == '#': continue
    if line[0] == '@': continue
    cont_new.append(line)
cont = cont_new

nP = int(len(cont)/m)
tP = np.zeros((nP, 1))
pcP = np.zeros((nP, N))

for j in xrange(N):
    for i in xrange(nP):
        tP[i] = cont[i].split()[0]
        pcP[i, j] = cont[i + nP*j].split()[1]


fh = open(fileA)
cont = fh.readlines()
fh.close()

cont_new = []
for line in cont:
    if line.isspace(): continue
    if line[0] == '&': continue
    if line[0] == '#': continue
    if line[0] == '@': continue
    cont_new.append(line)
cont = cont_new

nA = int(len(cont)/m)
tA = np.zeros((nA, 1))
pcA = np.zeros((nA, N))

for j in xrange(N):
    for i in xrange(nA):
        tA[i] = cont[i].split()[0]
        pcA[i, j] = cont[i + nA*j].split()[1]


fh = open(fileB)
cont = fh.readlines()
fh.close()

cont_new = []
for line in cont:
    if line.isspace(): continue
    if line[0] == '&': continue
    if line[0] == '#': continue
    if line[0] == '@': continue
    cont_new.append(line)
cont = cont_new

nB = int(len(cont)/m)
tB = np.zeros((nB, 1))
pcB = np.zeros((nB, N))

for j in xrange(N):
    for i in xrange(nB):
        tB[i] = cont[i].split()[0]
        pcB[i, j] = cont[i + nB*j].split()[1]


if opt.trajx_switch == 'yes':
    fileX = 'TEST_projT.xvg'
    
    fh = open(fileX)
    cont = fh.readlines()
    fh.close()

    cont_new = []
    for line in cont:
        if line.isspace(): continue
        if line[0] == '&': continue
        if line[0] == '#': continue
        if line[0] == '@': continue
        cont_new.append(line)
    cont = cont_new

    nX = int(len(cont)/m)
    tX = np.zeros((nX, 1))
    pcX = np.zeros((nX, N))

    for j in xrange(N):
        for i in xrange(nX):
            #if not cont[i].startswith(('#', '@', '&')):
            tX[i] = cont[i].split()[0]
            pcX[i, j] = cont[i + nX*j].split()[1]


#########################################
# GENERATE SYNTHETIC GAUSSIANS IF NEEDED
#########################################
#pcA = np.random.multivariate_normal([1, 2, 3], [[2, 10, 0], [0, 1, 0], [0, 0, 5]], n)
#pcB = np.random.multivariate_normal([10, 2, 3], [[2, 20, 0], [0, 1, 0], [0, 0, 5]], n)
#print 'Data set P:\n', pcP
#print 'Data set A:\n', pcA
#print 'Data set B:\n', pcB


###############################################
# COMPUTING N-DIMENSIONAL MEAN VECTORS AND COM
###############################################
mean_vectorA = np.zeros(N)
mean_vectorB = np.zeros(N)
for i in xrange(N):
    mean_vectorA[i] = np.mean(pcA[:, i])
    mean_vectorB[i] = np.mean(pcB[:, i])

q0 = 0.5*(mean_vectorA + mean_vectorB)

print '\nMean vector A:\n', mean_vectorA
print '\nMean vector B:\n', mean_vectorB
print '\nCOM vector A-B:\n', q0


##############################################
# COMPUTING COVARIANCE MATRICES FOR LOCAL PCA
##############################################
if opt.locpca_switch == 'yes':
    l_covA = []
    l_covB = []
    for i in xrange(N):
        l_covA.append(pcA[:, i])
        l_covB.append(pcB[:, i])
    cov_matA = np.cov(l_covA)
    cov_matB = np.cov(l_covB)
    #print '\nCovariance matrix A:\n', cov_matA
    #print '\nCovariance matrix B:\n', cov_matB


#########################################
# COMPUTING EIGENVECTORS AND EIGENVALUES
#########################################
if opt.locpca_switch == 'yes':
    eig_valA, eig_vecA = np.linalg.eig(cov_matA)
    eig_valB, eig_vecB = np.linalg.eig(cov_matB)
    idxA = eig_valA.argsort()[::-1]
    idxB = eig_valB.argsort()[::-1]
    eig_valA = eig_valA[idxA]
    eig_valB = eig_valB[idxB]
    eig_vecA = eig_vecA[:, idxA]
    eig_vecB = eig_vecB[:, idxB]
    eig_vecA = eig_vecA.T
    eig_vecB = eig_vecB.T
else:
    eig_valA = np.exp(np.linspace(-0.1, -3, N))
    eig_valB = np.exp(np.linspace(-0.1, -3, N))
    eig_vecA = np.zeros((N, N))
    eig_vecB = np.zeros((N, N))
    for k in xrange(N):
        eig_vecA[k] = np.eye(1, N, k)
        eig_vecB[k] = np.eye(1, N, k)

print '\nEigenvalues A:\n', eig_valA
print '\nEigenvectors A:\n', eig_vecA
print '\nEigenvalues B:\n', eig_valB
print '\nEigenvectors B:\n', eig_vecB

print '\nOrthogonality check for the first 3 A eigenvectors:\ns0*s1 =', np.dot(eig_vecA[0], eig_vecA[1])
print 's0*s2 =', np.dot(eig_vecA[0], eig_vecA[2])
print 's1*s2 =', np.dot(eig_vecA[1], eig_vecA[2])
print '\nOrthogonality check for the first 3 B eigenvectors:\ns0*s1 =', np.dot(eig_vecB[0], eig_vecB[1])
print 's0*s2 =', np.dot(eig_vecB[0], eig_vecB[2])
print 's1*s2 =', np.dot(eig_vecB[1], eig_vecB[2])


################################################
# COMPUTING OVERLAP BETWEEN PDB AND A SUBSPACES
################################################
if opt.locpca_switch == 'yes':
    pdb_eig_valA = np.exp(np.linspace(-0.1, -3, N))
    pdb_eig_valB = np.exp(np.linspace(-0.1, -3, N))
    pdb_eig_vecA = np.zeros((N, N))
    pdb_eig_vecB = np.zeros((N, N))
    for k in xrange(N):
        pdb_eig_vecA[k] = np.eye(1, N, k)
        pdb_eig_vecB[k] = np.eye(1, N, k)

    print '\nCalculating scalar products between P and A...\n'
    for i in xrange(N):
        print 'Vector pair i=', i, 'and dot(P, A) =', np.dot(pdb_eig_vecA[i], eig_vecA[i])

    print '\nCalculating scalar products between A and B\n'
    for i in xrange(N):
        print 'Vector pair i =', i, 'and dot(A, B) =', np.dot(eig_vecA[i], eig_vecB[i])


############################################
# CALCULATING PROJECTIONS ONTO NEW SUBSPACE
############################################
if opt.proj_switch == 'yes':
    if N <= 3:
        proj1A, proj2A, proj3A = (0, 1, 2)
        proj1B, proj2B, proj3B = (0, 1, 2)
        eig_pairsA = [(eig_valA[i], eig_vecA[i]) for i in range(len(eig_valA))]
        eig_pairsB = [(eig_valB[i], eig_vecB[i]) for i in range(len(eig_valB))]
        matrix_wA = np.hstack((eig_pairsA[proj1A][1].reshape(N, 1),
                               eig_pairsA[proj2A][1].reshape(N, 1),
                               eig_pairsA[proj3A][1].reshape(N, 1)))
        matrix_wB = np.hstack((eig_pairsB[proj1B][1].reshape(N, 1),
                               eig_pairsB[proj2B][1].reshape(N, 1),
                               eig_pairsB[proj3B][1].reshape(N, 1)))
    else:
        proj1A, proj2A, proj3A, proj4A = (0, 1, 2, 3)
        proj1B, proj2B, proj3B, proj4B = (0, 1, 2, 3)
        eig_pairsA = [(eig_valA[i], eig_vecA[i]) for i in range(len(eig_valA))]
        eig_pairsB = [(eig_valB[i], eig_vecB[i]) for i in range(len(eig_valB))]
        matrix_wA = np.hstack((eig_pairsA[proj1A][1].reshape(N, 1),
                               eig_pairsA[proj2A][1].reshape(N, 1),
                               eig_pairsA[proj3A][1].reshape(N, 1),
                               eig_pairsA[proj4A][1].reshape(N, 1)))
        matrix_wB = np.hstack((eig_pairsB[proj1B][1].reshape(N, 1),
                               eig_pairsB[proj2B][1].reshape(N, 1),
                               eig_pairsB[proj3B][1].reshape(N, 1),
                               eig_pairsB[proj4B][1].reshape(N, 1)))

    #matrix_wA = np.hstack((np.eye(N, 1), np.eye(N, 1, k=-1), np.eye(N, 1, k=-8)))
    # HACK: PROJECT ON PC1,2,3,4 AS IF N = 4
    #matrix_wA = np.zeros((N, 4))
    #matrix_wA[0] = np.array([0.30357415, 0.62912253, 0.69257505, -0.17996493])
    #matrix_wA[1] = np.array([0.85021621, -0.20961075, -0.05768234, 0.47944601])
    #matrix_wA[2] = np.array([-0.32997819, -0.45690914, 0.6811264, 0.46734918])
    #matrix_wA[3] = np.array([-0.2758433, 0.59287622, -0.23038962, 0.72064477])

    print '\nMatrix W A:\n', matrix_wA
    print '\nMatrix W B:\n', matrix_wB

    transfA = matrix_wA.T.dot(pcA.T)
    transfB_A = matrix_wA.T.dot(pcB.T)
    transfA = transfA.T
    transfB_A = transfB_A.T

    if opt.pdb_switch == 'yes':
        transfP_A = matrix_wA.T.dot(pcP.T)
        transfP_A = transfP_A.T

    if opt.trajx_switch == 'yes':
        transfX_A = matrix_wA.T.dot(pcX.T)
        transfX_A = transfX_A.T

    #print '\nA projected on PC subspace A:\n', transfA
    #print '\nB projected on PC subspace A:\n', transfB_A


#################################################
# COMPUTING OPTIMAL PLANE AND OPTIMAL COORDINATE
#################################################
if opt.fit_switch == 'yes':
    # default parameters
    zrange = 8.0
    dz = 0.1
    # use precalculated n_opt IF NEEDED
    #n_opt = np.array([])
    if N == 3:
        n_opt = np.array([ 0.70360405,  0.68669705, -0.1827252])
    elif N == 4:
        n_opt = np.array([ 0.69492178,  0.50391057, -0.21653963, -0.46504672])
    elif N == 5:
        n_opt = np.array([ 0.23798423,  0.36333282,  0.0027416,  -0.1990207,  -0.87848507])
    elif N == 7:
        n_opt = np.array([ 0.24798122,  0.36661836,  0.01969597, -0.22090064, -0.83589377,  0.13057474,
                          -0.19784617])
    elif N == 10:
        n_opt = np.array([ 0.23877958,  0.36408613, -0.00667061, -0.20235328, -0.82876336,  0.04971486,
                          -0.15657809,  0.05933661,  0.21253714,  0.08309279])
    elif N == 15:
        n_opt = np.array([ 0.22469446,  0.36074498, -0.00140852, -0.20018062, -0.84797981, -0.02813882,
                          -0.10130116,  0.06181178,  0.1717496,   0.09225153, -0.00755316, -0.04948826,
                           0.05707602,  0.03834376, -0.01070965])
    elif N == 20:
        n_opt = np.array([  2.07959348e-01,   3.57152553e-01,   2.85630232e-04,  -1.72992805e-01,
                           -8.57154354e-01,   7.06884439e-02,  -1.45157665e-01,   3.42589081e-02,
                            1.43153146e-01,   1.01884567e-01,  -1.95792179e-02,  -3.78663115e-02,
                            2.70173139e-02,   2.90660944e-02,   3.03120684e-02,   4.13688060e-02,
                            9.66662608e-03,   1.11847692e-02,  -1.40001699e-02,   1.86998052e-03])
    n_opt = -n_opt

    if len(n_opt) == 0:
        # wrapped overlap function
        def overlap_n(vec):
            return overlap(vec, pcA, pcB, q0, zrange, dz)

        # random unit vector in N dimensions
        def gen_rand_unit_vector(dim):
            vec = [gauss(0, 1) for i in xrange(dim)]
            mag = sum(x**2 for x in vec)**0.5
            return np.array([x/mag for x in vec])

        # simple stochastic search without acceptance
        niter = 3
        count_iter = 0
        gen_vec_list = np.zeros((niter, N))
        overlap_list = np.zeros(niter)
        print '\nInitializing simple stochastic search for n...'
        print 'Generating', niter, 'random unit vectors...'
        while count_iter < niter:
            n_g = gen_rand_unit_vector(N)
            gen_vec_list[count_iter] = n_g
            overlap_list[count_iter] = overlap_n(n_g)
            if count_iter % 100 == 0 or count_iter == niter-1:
                print 'Step', count_iter, ', direction vector =', n_g
            count_iter += 1

        min_ndx = overlap_list.argmin()
        n_g = gen_vec_list[min_ndx]

        # more precise minimum search with downhill simplex
        print '\nOptimal normalized G-vector n_g =', n_g
        print 'Initializing the downhill simplex vector search...'
        print 'Initial guess vector n_g =', n_g
        print 'Initial overlap =', overlap_n(n_g)

        n_opt = fmin(overlap_n, n_g, disp=True)
        if n_opt[0] < 0:
            n_opt = -n_opt/np.linalg.norm(n_opt)
        else:
            n_opt = n_opt/np.linalg.norm(n_opt)

    print '\nOptimal normalized vector n_opt =', n_opt
    print 'Computing the A, B projections on n_opt...'
    z, NzA, NzB = z_projection(n_opt, pcA, pcB, q0, zrange, dz)
    z0 = np.dot(n_opt, q0)
    print 'The variance of P on n_opt =', variance1(n_opt, pcP)

    #print '\nVECTOR-VECTOR ANGLES:'
    #print (180/np.pi)*pdist(gen_vec_list, lambda u, v: np.arccos(np.dot(u, v)))


#################################################
# PROJECTING A AND B ONTO ORC AND ITS ORTHOGONAL
#################################################
if opt.proj_orc_switch == 'yes' and opt.fit_switch == 'yes':
    # default parameters
    # use precalculated xi_opt IF NEEDED
    #xi_opt = np.array([])
    if N == 3:
        xi_opt = np.array([-0.71043943,  0.674465,   -0.20092977])
    elif N == 4:
        xi_opt = np.array([-0.56081159,  0.79012988, -0.21680274,  0.11908695])
    elif N == 5:
        xi_opt = np.array([-0.42320723,  0.84729236, -0.22347996,  0.02928062,  0.22845275])
    elif N == 7:
        xi_opt = np.array([-0.42935846,  0.85036097, -0.19570451,  0.00293861,  0.22988845, -0.013676,
                            0.03453852])
    elif N == 10:
        xi_opt = np.array([-4.71143104e-01,   8.15373277e-01,  -2.45314751e-01,   7.60607188e-04,
                            2.26185452e-01,  -1.85371564e-02,  -1.78406535e-02,   1.51774118e-02,
                            -1.58831521e-02,   2.65760582e-02])
    elif N == 15:
        xi_opt = np.array([-0.56680093,  0.74630504, -0.2318634,   0.02717206,  0.16240123, -0.03699605,
                            0.04955595, -0.02421254,  0.02907125,  0.00804693,  0.07516919,  0.02333723,
                            0.01216159,  0.08624047,  0.14761905])
    elif N == 20:
        xi_opt = np.array([-0.63652527,  0.64484261, -0.32225153,  0.02126326,  0.09814095,  0.11238414,
                            0.01858004, -0.10702609, -0.05962233, -0.00595065,  0.0642233,   0.02109631,
                            -0.04173309, -0.07077293, -0.02722324,  0.10038189, -0.07797524, -0.00073425,
                            -0.02584429, -0.09042763])

    if len(xi_opt) == 0:
        # wrapped variance function
        def variance1_xi(vec):
            return variance1(vec, pcA)

        def variance2_xi(vec):
            return variance2(vec, pcA, pcB)

        # generate a perpendicular vector to another one
        def gen_orthogonal(vec, dim):
            temp = np.array([gauss(0, 1) for i in xrange(dim-1)])
            add = -np.dot(temp, vec[:-1])/vec[dim-1]
            v = np.append(temp, add)
            return v/np.linalg.norm(v)

        # simple stochastic search without acceptance
        niter = 3
        count_iter = 0
        orth_vec_list = np.zeros((niter, N))
        variance1_list = np.zeros(niter)
        print '\nInitializing simple stochastic search for xi...'
        print 'Generating', niter, 'random unit vectors orthogonal to n_opt...'
        while count_iter < niter:
            xi_g = gen_orthogonal(n_opt, N)
            orth_vec_list[count_iter] = xi_g
            variance1_list[count_iter] = variance1_xi(xi_g)
            if count_iter % 100 == 0 or count_iter == niter-1:
                print 'Step', count_iter, ', orthogonal vector =', xi_g, ', (n_opt, xi_g) =', np.dot(n_opt, xi_g)
            count_iter += 1

        max_ndx = variance1_list.argmax()
        xi_g = orth_vec_list[max_ndx]
        if xi_g[0] > 0:
            xi_opt = -xi_g
        else:
            xi_opt = xi_g

    print '\nOptimal orthogonal vector xi_opt =', xi_opt

    # constructing W_X matrix
    matrix_wORC = np.hstack((n_opt.reshape(N, 1), xi_opt.reshape(N, 1)))

    # and now the orthogonal complement
    # TAKEN FROM: https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix/5889225
    def qr_null(A, tol=None):
        Q, R, P = qr(A.T, mode='full', pivoting=True)
        tol = np.finfo(R.dtype).eps if tol is None else tol
        rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
        return Q[:, rnk:].conj()

    Z = qr_null(matrix_wORC.T)
    matrix_wX = np.hstack((matrix_wORC, Z))
    print '\nMaxtrix W_X (ORC space):\n', matrix_wX

    # projecting everything onto the ORC space
    xA = matrix_wX.T.dot(pcA.T)
    xB = matrix_wX.T.dot(pcB.T)
    xA = xA.T
    xB = xB.T
    if opt.pdb_switch == 'yes':
        xP = matrix_wX.T.dot(pcP.T)
        xP = xP.T
    if opt.trajx_switch == 'yes':
        xX = matrix_wX.T.dot(pcX.T)
        xX = xX.T

    fh = open('orc_GTP_A.xvg', 'w')
    for i in xrange(len(tA)):
        fh.write('%f %f\n' % (tA[i], xA[i, 0]))
    fh.close()
    fh = open('orc_GTP_B.xvg', 'w')
    for i in xrange(len(tB)):
        fh.write('%f %f\n' % (tB[i], xB[i, 0]))
    fh.close()
    fh = open('orc_GTP_P.xvg', 'w')
    for i in xrange(len(tP)):
        fh.write('%f %f\n' % (tP[i], xP[i, 0]))
    fh.close()

    # picking structures according to geormetry criteria
    if opt.sel_orc_switch == 'yes':
        n_min = -8.03
        n_max = -7.97
        xi_min = 4.97
        xi_max =  5.03
        #tA_sel = np.array([tA[i] for i in xrange(len(tA)) if (xA[i, 0] > n_min and xA[i, 0] < n_max) and
        #                                                     (xA[i, 1] > xi_min and xA[i, 1] < xi_max)])
        #xA_sel = np.array([xA[i] for i in xrange(len(tA)) if (xA[i, 0] > n_min and xA[i, 0] < n_max) and
        #                                                     (xA[i, 1] > xi_min and xA[i, 1] < xi_max)])
        #pcA_sel = np.array([pcA[i] for i in xrange(len(tA)) if (xA[i, 0] > n_min and xA[i, 0] < n_max) and
        #                                                       (xA[i, 1] > xi_min and xA[i, 1] < xi_max)])
        tB_sel = np.array([tB[i] for i in xrange(len(tB)) if (xB[i, 0] > n_min and xB[i, 0] < n_max) and
                                                                   (xB[i, 1] > xi_min and xB[i, 1] < xi_max)])
        xB_sel = np.array([xB[i] for i in xrange(len(tB)) if (xB[i, 0] > n_min and xB[i, 0] < n_max) and
                                                                               (xB[i, 1] > xi_min and xB[i, 1] < xi_max)])
        pcB_sel = np.array([pcB[i] for i in xrange(len(tB)) if (xB[i, 0] > n_min and xB[i, 0] < n_max) and
                                                               (xB[i, 1] > xi_min and xB[i, 1] < xi_max)])
        #xA_sel = matrix_wA.T.dot(pcA_sel.T)
        #xA_sel = xA_sel.T
        xB_sel = matrix_wB.T.dot(pcB_sel.T)
        xB_sel = xB_sel.T
        print '\nSelecting structures within n_opt = [%.2f, %.2f] and xi_opt = [%.2f, %.2f]...' % (n_min, n_max, xi_min, xi_max)
        #print 'Have found %d with the time stamps:' % len(tA_sel)
        #print tA_sel
        #print pcA_sel
        print 'Have found %d with the time stamps:' % len(tB_sel)
        print tB_sel
        print pcB_sel


###########################################
# COMPUTING THE LINE BETWEEN RC^A_1 AND OC
###########################################
if opt.fit_switch == 'yes' and opt.orc_switch == 'yes':
    def dist_A1_OC(t):
        q1 = mean_vectorA
        d1 = eig_vecA[0]
        q2 = q0
        d2 = n_opt
        return np.dot(q1 - q2 + d1*t[0] - d2*t[1], q1 - q2 + d1*t[0] - d2*t[1])

    def dist_B1_OC(t):
        q1 = mean_vectorB
        d1 = eig_vecB[0]
        q2 = q0
        d2 = n_opt
        return np.dot(q1 - q2 + d1*t[0] - d2*t[1], q1 - q2 + d1*t[0] - d2*t[1])

    def v12(t, q1, d1, q2, d2):
        return q1 - q2 + d1*t[0] - d2*t[1]

    print '\nInitializing the minimal vector search...'
    print 'Initial guess A =', np.array([0, 0])
    print 'Initial guess B =', np.array([0, 0])
    print 'Initial distance A1-OC =', dist_A1_OC(np.array([0, 0]))
    print 'Initial distance B1-OC =', dist_B1_OC(np.array([0, 0]))

    t_A1_OC_opt = fmin(dist_A1_OC, np.array([0, 0]), disp=True)
    t_B1_OC_opt = fmin(dist_B1_OC, np.array([0, 0]), disp=True)
    print '\nOptimal values for A =', t_A1_OC_opt
    print 'Optimal values for B =', t_B1_OC_opt
    print 'The minimal A1-OC distance =', np.sqrt(dist_A1_OC(t_A1_OC_opt))
    print 'The minimal B1-OC distance =', np.sqrt(dist_A1_OC(t_A1_OC_opt))
    print 'The minimal connecting vector A1-OC =', v12(t_A1_OC_opt, mean_vectorA, eig_vecA[0], q0, n_opt)
    print 'The minimal connecting vector B1-OC =', v12(t_B1_OC_opt, mean_vectorB, eig_vecB[0], q0, n_opt)
    print 'DONE! :)'

    print '\nOrthogonality check for A1-OC, A1, and OC\n'
    print 'A1-OC * A1 =', np.dot(v12(t_A1_OC_opt, mean_vectorA, eig_vecA[0], q0, n_opt), eig_vecA[0])
    print 'A1-OC * OC =', np.dot(v12(t_A1_OC_opt, mean_vectorA, eig_vecA[0], q0, n_opt), n_opt)
    print 'Orthogonality check for B1-OC, B1, and OC\n'
    print 'B1-OC * B1 =', np.dot(v12(t_B1_OC_opt, mean_vectorB, eig_vecB[0], q0, n_opt), eig_vecB[0])
    print 'B1-OC * OC =', np.dot(v12(t_B1_OC_opt, mean_vectorB, eig_vecB[0], q0, n_opt), n_opt)


#################
# PLOTTING STUFF
#################

# PROJECTION ONTO NEW SUBSPACE
if opt.proj_switch == 'yes' and opt.plot_switch == 'yes':
    if N <= 3:
        fig = plt.figure(1, figsize=(30, 10))
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
    else:
        fig = plt.figure(1, figsize=(20, 5))
        ax0 = fig.add_subplot(141)
        ax1 = fig.add_subplot(142)
        ax2 = fig.add_subplot(143)
        ax3 = fig.add_subplot(144)

    # projected A and B clouds
    ax0.plot(transfA[:, 0], transfA[:, 1], 'o', color='green', markeredgewidth=0.0, markersize=3, alpha=0.3)
    ax0.plot(transfB_A[:, 0], transfB_A[:, 1], 'o', color='steelblue', markeredgewidth=0.0, markersize=3, alpha=0.3)
    ax1.plot(transfA[:, 0], transfA[:, 2], 'o', color='green', markeredgewidth=0.0, markersize=3, alpha=0.3)
    ax1.plot(transfB_A[:, 0], transfB_A[:, 2], 'o', color='steelblue', markeredgewidth=0.0, markersize=3, alpha=0.3)
    ax2.plot(transfA[:, 1], transfA[:, 2], 'o', color='green', markeredgewidth=0.0, markersize=3, alpha=0.3)
    ax2.plot(transfB_A[:, 1], transfB_A[:, 2], 'o', color='steelblue', markeredgewidth=0.0, markersize=3, alpha=0.3)
    if N > 3:
        ax3.plot(transfA[:, 0], transfA[:, 3], 'o', color='green', markeredgewidth=0.0, markersize=3, alpha=0.3)
        ax3.plot(transfB_A[:, 0], transfB_A[:, 3], 'o', color='steelblue', markeredgewidth=0.0, markersize=3, alpha=0.3)

    # indicate selected ORC structures
    if opt.proj_orc_switch == 'yes' and opt.sel_orc_switch == 'yes':
        ax0.plot(xB_sel[:, 0], xB_sel[:, 1], 's', color='yellow', markeredgewidth=0.0, markersize=5, alpha=1.0)
        ax1.plot(xB_sel[:, 0], xB_sel[:, 2], 's', color='yellow', markeredgewidth=0.0, markersize=5, alpha=1.0)
        ax2.plot(xB_sel[:, 1], xB_sel[:, 2], 's', color='yellow', markeredgewidth=0.0, markersize=5, alpha=1.0)
        if N > 3:
            ax3.plot(xB_sel[:, 0], xB_sel[:, 3], 's', color='yellow', markeredgewidth=0.0, markersize=5, alpha=1.0)

    # projected mean vectors for A and B
    tA_meanA = matrix_wA.T.dot(mean_vectorA.T)
    tA_meanB = matrix_wA.T.dot(mean_vectorB.T)
    ax0.plot([tA_meanA[0]], [tA_meanA[1]], 'o', markersize=10, color='red', alpha=0.5)
    ax0.plot([tA_meanB[0]], [tA_meanB[1]], 'o', markersize=10, color='red', alpha=0.5)
    ax1.plot([tA_meanA[0]], [tA_meanA[2]], 'o', markersize=10, color='red', alpha=0.5)
    ax1.plot([tA_meanB[0]], [tA_meanB[2]], 'o', markersize=10, color='red', alpha=0.5)
    ax2.plot([tA_meanA[1]], [tA_meanA[2]], 'o', markersize=10, color='red', alpha=0.5)
    ax2.plot([tA_meanB[1]], [tA_meanB[2]], 'o', markersize=10, color='red', alpha=0.5)
    if N > 3:
        ax3.plot([tA_meanA[0]], [tA_meanA[3]], 'o', markersize=10, color='red', alpha=0.5)
        ax3.plot([tA_meanB[0]], [tA_meanB[3]], 'o', markersize=10, color='red', alpha=0.5)

    # projected first eigenvectors of A and B
    tA_eig_vec1A = matrix_wA.T.dot(4*eig_vecA[0].T)
    tA_eig_vec1B = matrix_wA.T.dot(4*eig_vecB[0].T)
    ax0.plot([tA_meanA[0], tA_meanA[0] + tA_eig_vec1A[0]],
             [tA_meanA[1], tA_meanA[1] + tA_eig_vec1A[1]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax1.plot([tA_meanA[0], tA_meanA[0] + tA_eig_vec1A[0]],
             [tA_meanA[2], tA_meanA[2] + tA_eig_vec1A[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax2.plot([tA_meanA[1], tA_meanA[1] + tA_eig_vec1A[1]],
             [tA_meanA[2], tA_meanA[2] + tA_eig_vec1A[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    if N > 3:
        ax3.plot([tA_meanA[0], tA_meanA[0] + tA_eig_vec1A[0]],
                 [tA_meanA[3], tA_meanA[3] + tA_eig_vec1A[3]],
                 linewidth=2, markersize=10, color='red',
                 alpha=0.8)
    ax0.plot([tA_meanB[0], tA_meanB[0] + tA_eig_vec1B[0]],
             [tA_meanB[1], tA_meanB[1] + tA_eig_vec1B[1]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax1.plot([tA_meanB[0], tA_meanB[0] + tA_eig_vec1B[0]],
             [tA_meanB[2], tA_meanB[2] + tA_eig_vec1B[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax2.plot([tA_meanB[1], tA_meanB[1] + tA_eig_vec1B[1]],
             [tA_meanB[2], tA_meanB[2] + tA_eig_vec1B[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    if N > 3:
        ax3.plot([tA_meanB[0], tA_meanB[0] + tA_eig_vec1B[0]],
                 [tA_meanB[3], tA_meanB[3] + tA_eig_vec1B[3]],
                 linewidth=2, markersize=10, color='red',
                 alpha=0.8)
    ax0.plot([tA_meanA[0], tA_meanA[0] - tA_eig_vec1A[0]],
             [tA_meanA[1], tA_meanA[1] - tA_eig_vec1A[1]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax1.plot([tA_meanA[0], tA_meanA[0] - tA_eig_vec1A[0]],
             [tA_meanA[2], tA_meanA[2] - tA_eig_vec1A[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax2.plot([tA_meanA[1], tA_meanA[1] - tA_eig_vec1A[1]],
             [tA_meanA[2], tA_meanA[2] - tA_eig_vec1A[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    if N > 3:
        ax3.plot([tA_meanA[0], tA_meanA[0] - tA_eig_vec1A[0]],
                 [tA_meanA[3], tA_meanA[3] - tA_eig_vec1A[3]],
                 linewidth=2, markersize=10, color='red',
                 alpha=0.8)
    ax0.plot([tA_meanB[0], tA_meanB[0] - tA_eig_vec1B[0]],
             [tA_meanB[1], tA_meanB[1] - tA_eig_vec1B[1]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax1.plot([tA_meanB[0], tA_meanB[0] - tA_eig_vec1B[0]],
             [tA_meanB[2], tA_meanB[2] - tA_eig_vec1B[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    ax2.plot([tA_meanB[1], tA_meanB[1] - tA_eig_vec1B[1]],
             [tA_meanB[2], tA_meanB[2] - tA_eig_vec1B[2]],
             linewidth=2, markersize=10, color='red',
             alpha=0.8)
    if N > 3:
        ax3.plot([tA_meanB[0], tA_meanB[0] - tA_eig_vec1B[0]],
                 [tA_meanB[3], tA_meanB[3] - tA_eig_vec1B[3]],
                 linewidth=2, markersize=10, color='red',
                 alpha=0.8)

    # projected trajectory X data
    if opt.trajx_switch == 'yes':
        ax0.plot(transfX_A[:, 0], transfX_A[:, 1], 'o', color='red', markersize=3, alpha=0.9)
        ax1.plot(transfX_A[:, 0], transfX_A[:, 2], 'o', color='red', markersize=3, alpha=0.9)
        ax2.plot(transfX_A[:, 1], transfX_A[:, 2], 'o', color='red', markersize=3, alpha=0.9)
        if N > 3:
            ax3.plot(transfX_A[:, 0], transfX_A[:, 3], 'o', color='red', markersize=3, alpha=0.9)

    # projected PDB data
    if opt.pdb_switch == 'yes':
        ax0.plot(transfP_A[:, 0], transfP_A[:, 1], 'o', color='black', markersize=5, alpha=0.9)
        #ax0.plot(transfP_A[:2, 0], transfP_A[:2, 1], 'o', color='steelblue', markersize=15, alpha=0.9)
        #ax0.plot(transfP_A[2:4, 0], transfP_A[2:4, 1], 'o', color='green', markersize=15, alpha=0.9)
        ax1.plot(transfP_A[:, 0], transfP_A[:, 2], 'o', color='black', markersize=5, alpha=0.9)
        #ax1.plot(transfP_A[:2, 0], transfP_A[:2, 2], 'o', color='steelblue', markersize=15, alpha=0.9)
        #ax1.plot(transfP_A[2:4, 0], transfP_A[2:4, 2], 'o', color='green', markersize=15, alpha=0.9)
        ax2.plot(transfP_A[:, 1], transfP_A[:, 2], 'o', color='black', markersize=5, alpha=0.9)
        #ax2.plot(transfP_A[:2, 1], transfP_A[:2, 2], 'o', color='steelblue', markersize=15, alpha=0.9)
        #ax2.plot(transfP_A[2:4, 1], transfP_A[2:4, 2], 'o', color='green', markersize=15, alpha=0.9)
        if N > 3:
            ax3.plot(transfP_A[:, 0], transfP_A[:, 3], 'o', color='black', markersize=5, alpha=0.9)

    # projected optimal line
    if opt.fit_switch == 'yes':
        tA_n_opt = matrix_wA.T.dot(n_opt.T)
        tA_q0 = matrix_wA.T.dot(q0.T)
        ax0.plot([tA_q0[0]], [tA_q0[1]], 'o', markersize=10, color='magenta', alpha=0.5)
        ax1.plot([tA_q0[0]], [tA_q0[2]], 'o', markersize=10, color='magenta', alpha=0.5)
        ax2.plot([tA_q0[1]], [tA_q0[2]], 'o', markersize=10, color='magenta', alpha=0.5)
        if N > 3:
            ax3.plot([tA_q0[0]], [tA_q0[3]], 'o', markersize=10, color='magenta', alpha=0.5)
        ax0.plot([tA_q0[0], tA_q0[0] + 8*tA_n_opt[0]],
                 [tA_q0[1], tA_q0[1] + 8*tA_n_opt[1]],
                 linewidth=2, markersize=10, color='magenta',
                 alpha=0.8)
        ax0.plot([tA_q0[0], tA_q0[0] - 8*tA_n_opt[0]],
                 [tA_q0[1], tA_q0[1] - 8*tA_n_opt[1]],
                 linewidth=2, markersize=10, color='magenta',
                 alpha=0.8)
        ax1.plot([tA_q0[0], tA_q0[0] + 8*tA_n_opt[0]],
                 [tA_q0[2], tA_q0[2] + 8*tA_n_opt[2]],
                 linewidth=2, markersize=10, color='magenta',
                 alpha=0.8)
        ax1.plot([tA_q0[0], tA_q0[0] - 8*tA_n_opt[0]],
                 [tA_q0[2], tA_q0[2] - 8*tA_n_opt[2]],
                 linewidth=2, markersize=10, color='magenta',
                 alpha=0.8)
        ax2.plot([tA_q0[1], tA_q0[1] + 8*tA_n_opt[1]],
                 [tA_q0[2], tA_q0[2] + 8*tA_n_opt[2]],
                 linewidth=2, markersize=10, color='magenta',
                 alpha=0.8)
        ax2.plot([tA_q0[1], tA_q0[1] - 8*tA_n_opt[1]],
                 [tA_q0[2], tA_q0[2] - 8*tA_n_opt[2]],
                 linewidth=2, markersize=10, color='magenta',
                 alpha=0.8)
        if N > 3:
            ax3.plot([tA_q0[0], tA_q0[0] + 8*tA_n_opt[0]],
                     [tA_q0[3], tA_q0[3] + 8*tA_n_opt[3]],
                     linewidth=2, markersize=10, color='magenta',
                     alpha=0.8)
            ax3.plot([tA_q0[0], tA_q0[0] - 8*tA_n_opt[0]],
                     [tA_q0[3], tA_q0[3] - 8*tA_n_opt[3]],
                     linewidth=2, markersize=10, color='magenta',
                     alpha=0.8)

        # projected connection lines
        if opt.orc_switch == 'yes':
            q1A = mean_vectorA + eig_vecA[0]*t_A1_OC_opt[0]
            tA_q1A = matrix_wA.T.dot(q1A.T)
            q1B = mean_vectorB + eig_vecB[0]*t_B1_OC_opt[0]
            tA_q1B = matrix_wA.T.dot(q1B.T)
            qAOC = q0 + n_opt*t_A1_OC_opt[1]
            tA_qAOC = matrix_wA.T.dot(qAOC.T)
            qBOC = q0 + n_opt*t_B1_OC_opt[1]
            tA_qBOC = matrix_wA.T.dot(qBOC.T)
            v12_A1_OC_opt = v12(t_A1_OC_opt, mean_vectorA, eig_vecA[0], q0, n_opt)
            tA_v12_A1_OC_opt = matrix_wA.T.dot(v12_A1_OC_opt.T)
            v12_B1_OC_opt = v12(t_B1_OC_opt, mean_vectorB, eig_vecB[0], q0, n_opt)
            tA_v12_B1_OC_opt = matrix_wA.T.dot(v12_B1_OC_opt.T)

            ax0.plot([tA_q1A[0]], [tA_q1A[1]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax1.plot([tA_q1A[0]], [tA_q1A[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax2.plot([tA_q1A[1]], [tA_q1A[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax0.plot([tA_qAOC[0]], [tA_qAOC[1]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax1.plot([tA_qAOC[0]], [tA_qAOC[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax2.plot([tA_qAOC[1]], [tA_qAOC[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax0.plot([tA_q1B[0]], [tA_q1B[1]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax1.plot([tA_q1B[0]], [tA_q1B[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax2.plot([tA_q1B[1]], [tA_q1B[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax0.plot([tA_qBOC[0]], [tA_qBOC[1]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax1.plot([tA_qBOC[0]], [tA_qBOC[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax2.plot([tA_qBOC[1]], [tA_qBOC[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax0.plot([tA_qAOC[0], tA_qAOC[0] + tA_v12_A1_OC_opt[0]],
                     [tA_qAOC[1], tA_qAOC[1] + tA_v12_A1_OC_opt[1]],
                     linewidth=2, markersize=10, color='cyan',
                     alpha=0.8)
            ax0.plot([tA_qBOC[0], tA_qBOC[0] + tA_v12_B1_OC_opt[0]],
                     [tA_qBOC[1], tA_qBOC[1] + tA_v12_B1_OC_opt[1]],
                     linewidth=2, markersize=10, color='cyan',
                     alpha=0.8)
            ax1.plot([tA_qAOC[0], tA_qAOC[0] + tA_v12_A1_OC_opt[0]],
                     [tA_qAOC[2], tA_qAOC[2] + tA_v12_A1_OC_opt[2]],
                     linewidth=2, markersize=10, color='cyan',
                     alpha=0.8)
            ax1.plot([tA_qBOC[0], tA_qBOC[0] + tA_v12_B1_OC_opt[0]],
                     [tA_qBOC[2], tA_qBOC[2] + tA_v12_B1_OC_opt[2]],
                     linewidth=2, markersize=10, color='cyan',
                     alpha=0.8)
            ax2.plot([tA_qAOC[1], tA_qAOC[1] + tA_v12_A1_OC_opt[1]],
                     [tA_qAOC[2], tA_qAOC[2] + tA_v12_A1_OC_opt[2]],
                     linewidth=2, markersize=10, color='cyan',
                     alpha=0.8)
            ax2.plot([tA_qBOC[1], tA_qBOC[1] + tA_v12_B1_OC_opt[1]],
                     [tA_qBOC[2], tA_qBOC[2] + tA_v12_B1_OC_opt[2]],
                     linewidth=2, markersize=10, color='cyan',
                     alpha=0.8)

    if opt.locpca_switch == 'yes':
        ax0.set_title('Projection local A PC12')
        ax0.set_xlabel('new_PC1')
        ax0.set_ylabel('new_PC2')
        ax1.set_title('Projection local A PC13')
        ax1.set_xlabel('new_PC1')
        ax1.set_ylabel('new_PC3')
        ax2.set_title('Projection local A PC23')
        ax2.set_xlabel('new_PC2')
        ax2.set_ylabel('new_PC3')
        if N > 3:
            ax3.set_title('Projection local A PC14')
            ax3.set_xlabel('new_PC1')
            ax3.set_ylabel('new_PC4')
    else:
        ax0.set_title('Projection GTP PC12')
        ax0.set_xlabel('old_PC1')
        ax0.set_ylabel('old_PC2')
        ax1.set_title('Projection GTP PC13')
        ax1.set_xlabel('old_PC1')
        ax1.set_ylabel('old_PC3')
        ax2.set_title('Projection GTP PC23')
        ax2.set_xlabel('old_PC2')
        ax2.set_ylabel('old_PC3')
        if N > 3:
            ax3.set_title('Projection GTP PC14')
            ax3.set_xlabel('old_PC1')
            ax3.set_ylabel('old_PC4')
    ax0.axis('equal')
    ax0.grid(True)
    ax1.axis('equal')
    ax1.grid(True)
    ax2.axis('equal')
    ax2.grid(True)
    if N > 3:
        ax3.axis('equal')
        ax3.grid(True)

# CALCULATED EIGENVECTORS IN OLD SUBSPACE (ONLY FOR N = 3)
if opt.dim == '3' and opt.plot_switch == 'yes':
    fig = plt.figure(2, figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')

    # clouds A and B in 3D with PC123 from PDB PCA
    ax.plot(pcA[:, 0], pcA[:, 1], pcA[:, 2], 'o', color='green', markersize=2, alpha=0.3)
    ax.plot(pcB[:, 0], pcB[:, 1], pcB[:, 2], 'o', color='steelblue', markersize=2, alpha=0.3)
    if opt.trajx_switch == 'yes':
        ax.plot(pcX[:, 0], pcX[:, 1], pcX[:, 2], 'o', color='gray', markersize=2, alpha=0.3)
    if opt.pdb_switch == 'yes':
        ax.plot(pcP[:, 0], pcP[:, 1], pcP[:, 2], 'o', color='black', markersize=5, alpha=0.9)
        #ax.plot(pcP[:4, 0], pcP[:4, 1], pcP[:4, 2], 'o', color='gray', markersize=15, alpha=0.9)
    ax.plot([mean_vectorA[0]], [mean_vectorA[1]], [mean_vectorA[2]], 'o', markersize=10, color='red', alpha=0.5)
    ax.plot([mean_vectorB[0]], [mean_vectorB[1]], [mean_vectorB[2]], 'o', markersize=10, color='red', alpha=0.5)

    if opt.fit_switch == 'yes':
        ax.plot([q0[0]], [q0[1]], [q0[2]], 'o', markersize=10, color='blue', alpha=0.5)
        ax.plot([q0[0], q0[0] + 4*n_opt[0]],
                [q0[1], q0[1] + 4*n_opt[1]],
                [q0[2], q0[2] + 4*n_opt[2]],
                linewidth=2, markersize=10, color='blue',
                alpha=0.8)
        ax.plot([q0[0], q0[0] - 4*n_opt[0]],
                [q0[1], q0[1] - 4*n_opt[1]],
                [q0[2], q0[2] - 4*n_opt[2]],
                linewidth=2, markersize=10, color='blue',
                alpha=0.8)

        if opt.orc_switch == 'yes':
            q1A = mean_vectorA + eig_vecA[0]*t_A1_OC_opt[0]
            q1B = mean_vectorB + eig_vecB[0]*t_B1_OC_opt[0]
            qAOC = q0 + n_opt*t_A1_OC_opt[1]
            qBOC = q0 + n_opt*t_B1_OC_opt[1]
            v12_A1_OC_opt = v12(t_A1_OC_opt, mean_vectorA, eig_vecA[0], q0, n_opt)
            v12_B1_OC_opt = v12(t_B1_OC_opt, mean_vectorB, eig_vecB[0], q0, n_opt)
            ax.plot([q1A[0]], [q1A[1]], [q1A[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax.plot([q1B[0]], [q1B[1]], [q1B[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax.plot([qAOC[0]], [qAOC[1]], [qAOC[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax.plot([qBOC[0]], [qBOC[1]], [qBOC[2]], 'o', markersize=5, color='cyan', alpha=0.5)
            ax.plot([qAOC[0], qAOC[0] + v12_A1_OC_opt[0]],
                    [qAOC[1], qAOC[1] + v12_A1_OC_opt[1]],
                    [qAOC[2], qAOC[2] + v12_A1_OC_opt[2]],
                    linewidth=2, markersize=10, color='cyan',
                    alpha=0.8)
            ax.plot([qBOC[0], qBOC[0] + v12_B1_OC_opt[0]],
                    [qBOC[1], qBOC[1] + v12_B1_OC_opt[1]],
                    [qBOC[2], qBOC[2] + v12_B1_OC_opt[2]],
                    linewidth=2, markersize=10, color='cyan',
                    alpha=0.8)

    for i_v, i_e in zip(xrange(3), xrange(3)):
        vA = 1*eig_valA[i_e]*eig_vecA[i_v]
        vB = 1*eig_valB[i_e]*eig_vecB[i_v]
        ax.plot([mean_vectorA[0], mean_vectorA[0] + vA[0]],
                [mean_vectorA[1], mean_vectorA[1] + vA[1]],
                [mean_vectorA[2], mean_vectorA[2] + vA[2]],
                linewidth=2, markersize=10, color='red',
                alpha=0.8)
        ax.plot([mean_vectorA[0], mean_vectorA[0] - vA[0]],
                [mean_vectorA[1], mean_vectorA[1] - vA[1]],
                [mean_vectorA[2], mean_vectorA[2] - vA[2]],
                linewidth=2, markersize=10, color='red',
                alpha=0.8)
        ax.plot([mean_vectorB[0], mean_vectorB[0] + vB[0]],
                [mean_vectorB[1], mean_vectorB[1] + vB[1]],
                [mean_vectorB[2], mean_vectorB[2] + vB[2]],
                linewidth=2, markersize=10, color='red',
                alpha=0.8)
        ax.plot([mean_vectorB[0], mean_vectorB[0] - vB[0]],
                [mean_vectorB[1], mean_vectorB[1] - vB[1]],
                [mean_vectorB[2], mean_vectorB[2] - vB[2]],
                linewidth=2, markersize=10, color='red',
                alpha=0.8)

    ax.set_title('Old 3D PDB PCA subspace')
    ax.set_xlabel('old_PC1')
    ax.set_ylabel('old_PC2')
    ax.set_zlabel('old_PC3')
    ax.set_xlim([np.amin([mean_vectorA[0], mean_vectorB[0]]) - 7, np.amax([mean_vectorA[0], mean_vectorB[0]]) + 7])
    ax.set_ylim([np.amin([mean_vectorA[1], mean_vectorB[1]]) - 7, np.amax([mean_vectorA[1], mean_vectorB[1]]) + 7])
    ax.set_zlim([np.amin([mean_vectorA[2], mean_vectorB[2]]) - 7, np.amax([mean_vectorA[2], mean_vectorB[2]]) + 7])
    ax.grid(True)

# OPTIMAL PLANE AND OPTIMAL COORDINATE
if opt.fit_switch == 'yes' and opt.plot_switch == 'yes':
    fig = plt.figure(3, figsize=(13, 7))
    ax0 = fig.add_subplot(111)
    ax0.plot(z, NzA, '-', color='green', linewidth=2)
    ax0.plot(z, NzB, '-', color='steelblue', linewidth=2)
    ax0.set_xlabel('z, nm')
    ax0.set_ylabel('N(z)')
    ax0.set_xlim([z0 - zrange, z0 + zrange])
    ax0.set_title('Minimal overlap = %f' % overlap(n_opt, pcA, pcB, (mean_vectorA + mean_vectorB)/2, zrange, dz))
    ax0.grid(True)

# 3D PROJECTION OF THE ORC SPACE ONTO (n_opt, xi_opt, xi_i)
if opt.proj_orc_switch == 'yes' and opt.plot_switch == 'yes':
    fig = plt.figure(4, figsize=(9, 9))
    ax0 = fig.add_subplot(111, projection='3d')

    I = 2

    ax0.plot(xA[:, 0], xA[:, 1], xA[:, I], 'o', color='green', markeredgewidth=0.0, markersize=3, alpha=0.3)
    ax0.plot(xB[:, 0], xB[:, 1], xB[:, I], 'o', color='steelblue', markeredgewidth=0.0, markersize=3, alpha=0.3)

    # projected mean vectors for A and B
    x_meanA = matrix_wX.T.dot(mean_vectorA.T)
    x_meanB = matrix_wX.T.dot(mean_vectorB.T)
    ax0.plot([x_meanA[0]], [x_meanA[1]], [x_meanA[I]], 'o', markersize=10, color='red', alpha=0.5)
    ax0.plot([x_meanB[0]], [x_meanB[1]], [x_meanB[I]], 'o', markersize=10, color='red', alpha=0.5)

    # projected optimal line
    x0 = matrix_wX.T.dot(q0.T)
    ax0.plot([x0[0]], [x0[1]], [x0[I]], 'o', markersize=10, color='magenta', alpha=0.5)

    # projected PDB data
    if opt.pdb_switch == 'yes':
        ax0.plot(xP[:, 0], xP[:, 1], xP[:, I], 'o', color='black', markersize=5, alpha=0.9)

    # projected traj X data
    if opt.trajx_switch == 'yes':
        #ax0.plot(xX[:, 0], xX[:, 1], xX[:, I], 'o', color='red', markersize=3, alpha=0.9)
        ax0.scatter(xX[:, 0], xX[:, 1], xX[:, I], 'o', c=tX[:, 0], alpha=0.9)

    # indicate selected structures
    if opt.sel_orc_switch == 'yes':
        ax0.plot(xB_sel[:, 0], xB_sel[:, 1], xB_sel[:, I], 's', color='yellow', markeredgewidth=0.0, markersize=5, alpha=1.0)

    ax0.set_title('ORC space projection into (n_opt, xi_opt, xi_%d) (N = %d)' % (I, N))
    ax0.set_xlabel('n_opt')
    ax0.set_ylabel('xi_opt')
    ax0.set_zlabel('xi_i')
    ax0.axis('equal')
    ax0.grid(True)

# 2D PROJECTION OF THE ORC SPACE ONTO (n_opt, xi_opt)
if opt.proj_orc_switch == 'yes' and opt.plot_switch == 'yes':
    fig = plt.figure(5, figsize=(9, 9))
    ax0 = fig.add_subplot(111)

    ax0.plot(xA[:, 0], xA[:, 1], 'o', color='green', markeredgewidth=0.0, markersize=3, alpha=0.3)
    ax0.plot(xB[:, 0], xB[:, 1], 'o', color='steelblue', markeredgewidth=0.0, markersize=3, alpha=0.3)

    # projected mean vectors for A and B
    x_meanA = matrix_wX.T.dot(mean_vectorA.T)
    x_meanB = matrix_wX.T.dot(mean_vectorB.T)
    ax0.plot([x_meanA[0]], [x_meanA[1]], 'o', markersize=10, color='red', alpha=0.5)
    ax0.plot([x_meanB[0]], [x_meanB[1]], 'o', markersize=10, color='red', alpha=0.5)

    # projected optimal line
    x0 = matrix_wX.T.dot(q0.T)
    ax0.plot([x0[0]], [x0[1]], 'o', markersize=10, color='magenta', alpha=0.5)

    # projected PDB data
    if opt.pdb_switch == 'yes':
        ax0.plot(xP[:, 0], xP[:, 1], 'o', color='black', markersize=5, alpha=0.9)
        #fh = open('TEST.txt', 'w')
        #for i in xrange(len(xP[:, 0])):
        #    fh.write("%f %f\n" % (xP[i, 0], xP[i, 1]))
        #fh.close()
    ax0.plot(xP[1, 0], xP[1, 1], 'o', color='red', markersize=10, alpha=0.9)

    # projected traj X data
    if opt.trajx_switch == 'yes':
        ax0.plot(xX[:, 0], xX[:, 1], 'o', color='red', markersize=3, alpha=0.9)

    # indicate selected structures
    if opt.sel_orc_switch == 'yes':
        ax0.plot(xB_sel[:, 0], xB_sel[:, 1], 's', color='yellow', markeredgewidth=0.0, markersize=5, alpha=1.0)

    ax0.set_title('ORC plane projection')
    ax0.set_xlabel('n_opt')
    ax0.set_ylabel('xi_opt')
    ax0.axis('equal')
    ax0.grid(True)

# 2D PROJECTION OF THE ORC SPACE ONTO (n_opt, xi_opt) HISTOGRAM
if opt.proj_orc_switch == 'yes':
    fig = plt.figure(6, figsize=(9, 9))
    ax0 = fig.add_subplot(111)

    #distrA, xedges, yedges = np.histogram2d(xA[:, 0], xA[:, 1], bins=100, range=[[-30, 20], [-10, 40]], normed=True)
    #distrB, xedges, yedges = np.histogram2d(xB[:, 0], xB[:, 1], bins=100, range=[[-30, 20], [-10, 40]], normed=True)
    distrA, xedges, yedges = np.histogram2d(xA[:, 0], xA[:, 1], bins=100, range=[[-20, 30], [-10, 40]], normed=True)
    distrB, xedges, yedges = np.histogram2d(xB[:, 0], xB[:, 1], bins=100, range=[[-20, 30], [-10, 40]], normed=True)
    distrA = distrA.T
    distrB = distrB.T
    #np.savetxt('hist_A.txt', distrA, fmt='%1.4e')

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    norm=matplotlib.colors.Normalize(vmin=0, vmax=9, clip=False)
    levels=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    caxA = ax0.contour(X, Y, -np.log(distrA/2),
                       extent=extent,
                       norm=norm,
                       levels=levels,
                       cmap='Greens',
                       alpha=0.8)
    #cbarA = fig.colorbar(caxA)
    #cbarA.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8])

    caxB = ax0.contour(X, Y, -np.log(distrB/2),
                       extent=extent,
                       norm=norm,
                       levels=levels,
                       cmap='Blues',
                       alpha=0.8)
    #cbarB = fig.colorbar(caxB)
    #cbarB.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # projected PDB data
    if opt.pdb_switch == 'yes':
        ax0.plot(xP[:, 0], xP[:, 1], 's', color='black', markersize=5, alpha=0.9)
        ax0.grid(True)

    # projected traj X data
    if opt.trajx_switch == 'yes':
        #ax0.plot(xX[:, 0], xX[:, 1], linewidth=1, color='red', alpha=0.9)
        ax0.scatter(xX[:, 0], xX[:, 1], c=tX[:, 0])

plt.show()
