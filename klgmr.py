import numpy as np
from numpy.linalg import det
import numba
from numba import prange
from sklearn.neighbors import NearestNeighbors

ii32 = np.iinfo(np.int32)
MAXINT = ii32.max

################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])', nopython=True)
def _outer(x, y):
    """
    Computes the outer production between 1d-ndarrays x and y.
    """
    m = x.shape[0]
    n = y.shape[0]
    res = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            res[i, j] = x[i]*y[j]
    return res


@numba.jit('float64 (float64[:], float64[:], float64[:,:])', nopython=True)
def normal(x, mu, cov):
    """
    Normal distribution with parameters mu (mean) and cov (covariance matrix)
    """
    d = mu.shape[0]
    return (1./np.sqrt((2.*np.pi)**d * det(cov))) * np.exp(-0.5*np.dot(x-mu, np.dot(np.linalg.inv(cov), x-mu)))


def normalize(w, mu, cov):
    pass



def ncomp_finder(kl_hist, w_size=10):
    """
    Heuristic: If the actual diff is 1 order of magnitude
    greater than the mean of the 10 last diff values, we 
    consider this points as the estimate of the number of components
    """
    diff = np.diff(kl_hist)
    diff -= diff.min()
    diff /= diff.max()
    reached_flag = False
    
    for i in range(w_size, len(diff)):
        # If actual diff is 1 order of magnitude
        if diff[i] > 10*np.mean(diff[i-w_size:i]):
            reached_flag = True
            break
    if not reached_flag:
        # in case of no high increase is detected
        i += 1
    return len(kl_hist)-i



@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def merge(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computes the moment preserving merge of components (w1,mu1,cov1) and
    (w2,mu2,cov2)
    """
    w_m = w1+w2
    mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
    cov_m = (w1/w_m)*cov1 + (w2/w_m)*cov2 + (w1*w2/w_m**2)*_outer(mu1-mu2, mu1-mu2)
    return (w_m, mu_m, cov_m)



@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isomorphic_merge(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computes the isomorphic moment preserving merge of components (w1,mu1,cov1) and
    (w2,mu2,cov2)
    """
    d = len(mu1)
    w_m = w1+w2
    mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
    cov_m = (w1/w_m)*cov1 + (w2/w_m)*cov2 + (w1*w2/w_m**2) * np.abs(det(_outer(mu1-mu2, mu1-mu2)))**(1./d) * np.identity(d)
    return (w_m, mu_m, cov_m)



@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def kl_diss(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computation of the KL-divergence (dissimilarity) upper bound between components 
    [(w1,mu1,cov1), (w2,mu2,cov2)]) and its moment preserving merge, as proposed in 
    ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    """
    w_m, mu_m, cov_m = merge(w1, mu1, cov1, w2, mu2, cov2)
    return 0.5*((w1+w2)*np.log(det(cov_m)) - w1*np.log(det(cov1)) - w2*np.log(det(cov2)))



@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isd_diss(w1, mu1, cov1, w2, mu2, cov2):
    """
    Computes the ISD (Integral Square Difference between components [(w1,mu1,cov1), (w2,mu2,cov2)])
    and its moment preserving merge. Ref: Cost-Function-Based Gaussian Mixture Reduction for Target Tracking
    """
    w_m, mu_m, cov_m = merge(w1, mu1, cov1, w2, mu2, cov2)
    # ISD analytical computation between merged component and the pair of gaussians
    Jhr = w1*w_m * normal(mu1, mu_m, cov1+cov_m) + w2*w_m * normal(mu2, mu_m, cov2+cov_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * det(2*cov_m)))
    Jhh = (w1**2)*(1./np.sqrt((2*np.pi)**2 * det(2*cov1))) + \
          (w2**2)*(1./np.sqrt((2*np.pi)**2 * det(2*cov2))) + \
          2*w1*w2*normal(mu1, mu2, cov1+cov2)
    return Jhh - 2*Jhr + Jrr



def compute_neighbors(mu_center, maxsig):
    nn = NearestNeighbors(radius=maxsig, algorithm="ball_tree", n_jobs=-1)
    nn.fit(mu_center)
    neigh_indexes_arr = nn.radius_neighbors(mu_center, return_distance=False)
    
    # creating the initial array
    maxlen = 0
    for arr in neigh_indexes_arr:
        if len(arr)>maxlen:
            maxlen = len(arr)
    neigh_indexes = MAXINT*np.ones((len(neigh_indexes_arr),maxlen-1), dtype=np.int32)
    
    # filling it with the correct indexes
    for i,arr in enumerate(neigh_indexes_arr):
        ll = arr.tolist(); ll.remove(i); ll.sort()
        for j,index in enumerate(ll):
            neigh_indexes[i,j] = index      
    return nn,neigh_indexes



@numba.jit(nopython=True)
def build_diss_matrix(w, mu, cov, nn_indexes):
    M,max_neigh = nn_indexes.shape
    diss_matrix = -1.*np.ones((M,max_neigh))
    for i in range(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            diss_matrix[i,j] = kl_diss(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])  
    return diss_matrix



@numba.jit(nopython=True)
def least_dissimilar(diss_matrix, indexes, nn_indexes):
    max_neigh = diss_matrix.shape[1]
    i_min = -1; j_min = -1
    diss_min = np.inf
    for i in indexes:
        for j in range(max_neigh):
            if diss_matrix[i,j]==-1: break
            if diss_matrix[i,j]<diss_min:
                diss_min = diss_matrix[i,j]
                i_min = i
                j_min = nn_indexes[i,j]
    return i_min,j_min


@numba.jit(nopython=True)
def get_index(array, value):
    n = len(array)
    for i in range(n):
        if array[i]==value: return i
    return -1


@numba.jit(nopython=True)
def update_merge_mapping(merge_mapping, nindex, dindex):
    n = len(merge_mapping)
    for i in range(n):
        if merge_mapping[i]==dindex:
            merge_mapping[i] = nindex



def radius_search(nn, mu, max_neigh, merge_mapping, nindex, dindex):
    neigh_arr = nn.radius_neighbors([mu], return_distance=False)[0]
    for i in range(len(neigh_arr)):
        ii = merge_mapping[neigh_arr[i]]
        # avoiding neighbor of itself
        if ii==nindex or ii==dindex:
            neigh_arr[i] = MAXINT
            continue
        neigh_arr[i] = ii
    neigh_arr = np.unique(neigh_arr)
    if len(neigh_arr)>max_neigh:
        neigh_arr = nn.kneighbors([mu], n_neighbors=max_neigh, return_distance=False)[0]
        for i in range(len(neigh_arr)):
            ii = merge_mapping[neigh_arr[i]]
            # avoiding neighbor of itself
            if ii==nindex or ii==dindex:
                neigh_arr[i] = MAXINT
                continue
            neigh_arr[i] = ii
        neigh_arr = np.unique(neigh_arr)
    ret = MAXINT*np.ones(max_neigh, dtype=np.int32)
    ret[0:len(neigh_arr)] = neigh_arr
    return ret



@numba.jit(nopython=True)
def update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex):
    """
    Updates the nn_indexes and diss_matrix structs by removing the items
    corresponding to dindex and updating the ones corresponding to nindex
    """
    max_neigh = nn_indexes.shape[1]
    for i in indexes:
        if i==nindex: continue # this is an special case (see below)
        flag1 = False
        flag2 = False
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            if jj==nindex: 
                diss_matrix[i,j] = kl_diss(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])
                flag1 = True
            elif jj==dindex and flag1:
                nn_indexes[i,j] = MAXINT
                diss_matrix[i,j] = -1
                flag2 = True
            elif jj==dindex and not flag1:
                nn_indexes[i,j] = nindex
                diss_matrix[i,j] = kl_diss(w[i],mu[i],cov[i],w[jj],mu[jj],cov[jj])
                flag2 = True
        if flag2:
            sorted_indexes = np.argsort(nn_indexes[i,:])
            nn_indexes[i,:] = (nn_indexes[i,:])[sorted_indexes]
            diss_matrix[i,:] = (diss_matrix[i,:])[sorted_indexes]

    # the special case...
    for j in range(max_neigh):
        jj = nn_indexes[nindex,j]
        if jj!=MAXINT:
            diss_matrix[nindex,j] = kl_diss(w[nindex],mu[nindex],cov[nindex],w[jj],mu[jj],cov[jj])
        else:
            diss_matrix[nindex,j] = -1


################################################################
# MAIN FUNCTION
################################################################

def mixture_reduction(w, mu, cov, n_comp, isomorphic=False, verbose=True, optimization=True):
    """
    Gaussian Mixture Reduction Through KL-upper bound approach
    """

    # original size of the mixture
    M = len(w) 
    # target size of the mixture
    N = n_comp
    # dimensionality of data
    d = mu.shape[1]

    # we consider neighbors at a radius equivalent to the lenght of 5 pixels
    if cov.ndim==1:
        maxsig = 5*np.max(cov)
        # if cov is 1-dimensional we convert it to its covariance matrix form
        cov = np.asarray( [(val**2)*np.identity(d) for val in cov] )
    else:
        maxsig = 5*max([np.max(np.linalg.eig(_cov)[0])**(1./2) for _cov in cov])

    indexes = np.arange(M, dtype=np.int32)
    nn,nn_indexes = compute_neighbors(mu, maxsig)

    # idea: keep track that the k-th component was merged into the l-th positon
    merge_mapping = np.arange(M, dtype=np.int32)

    # max number of neighbors
    max_neigh = nn_indexes.shape[1]
    
    # computing the initial dissimilarity matrix
    diss_matrix = build_diss_matrix(w, mu, cov, nn_indexes)  
    
    # main loop
    while M>N:
        i_min, j_min = least_dissimilar(diss_matrix, indexes, nn_indexes)
        if verbose: print('Merged components {0} and {1}'.format(i_min, j_min))  
        w_m, mu_m, cov_m = merge(w[i_min], mu[i_min], cov[i_min], 
                                 w[j_min], mu[j_min], cov[j_min])
 
        # updating structures
        nindex = min(i_min,j_min) # index of the new component
        dindex = max(i_min,j_min) # index of the del component
        w[nindex] = w_m; mu[nindex] = mu_m; cov[nindex] = cov_m
        indexes = np.delete(indexes, get_index(indexes,dindex))
        update_merge_mapping(merge_mapping, nindex, dindex)
        nn_indexes[nindex] = radius_search(nn, mu_m, max_neigh, merge_mapping, nindex, dindex)
        update_structs(nn_indexes, diss_matrix, w, mu, cov, indexes, nindex, dindex)
        M -= 1

    # indexes of the "alive" mixture components
    return w[indexes],mu[indexes],cov[indexes]


