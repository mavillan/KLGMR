import copy
import numba
from numba import prange
import numpy as np
from sklearn.neighbors import NearestNeighbors

ii32 = np.iinfo(np.int32)
MAXINT = ii32.max

################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])', nopython=True)
def _outer(x, y):
    m = x.shape[0]
    n = y.shape[0]
    res = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            res[i, j] = x[i]*y[j]
    return res


@numba.jit('float64 (float64[:,:])', nopython=True)
def _det(X):
    """
    Direct computation of determinant for 
    matrices of size 2x2 and 3x3
    """
    n = X.shape[0]
    if n==2:
        return X[0,0]*X[1,1] - X[0,1]*X[1,0]
    else:
        return X[0,0] * (X[1,1] * X[2,2] - X[2,1] * X[1,2]) - \
               X[1,0] * (X[0,1] * X[2,2] - X[2,1] * X[0,2]) + \
               X[2,0] * (X[0,1] * X[1,2] - X[1,1] * X[0,2])


@numba.jit('float64 (float64[:], float64[:], float64[:,:])', nopython=True)
def normal(x, mu, sig):
    d = mu.shape[0]
    return (1./np.sqrt((2.*np.pi)**d * np.linalg.det(sig))) * np.exp(-0.5*np.dot(x-mu, np.dot(np.linalg.inv(sig), x-mu)))


def remove(l, indexes):
    """
    Remove elements from list
    yielding a new list
    """
    return [val for i,val in enumerate(l) if i not in indexes]


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


#################################################################
# MOMENT PRESERVING GAUSSIAN
#################################################################


@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def merge(w1, mu1, sig1, w2, mu2, sig2):
    w_m = w1+w2
    mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
    sig_m = (w1/w_m)*sig1 + (w2/w_m)*sig2 + (w1*w2/w_m**2)*_outer(mu1-mu2, mu1-mu2)
    return (w_m, mu_m, sig_m)


@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], \
            float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isomorphic_merge(w1, mu1, sig1, w2, mu2, sig2):
    d = len(mu1)
    w_m = w1+w2
    mu_m = (w1/w_m)*mu1 + (w2/w_m)*mu2
    sig_m = (w1/w_m)*sig1 + (w2/w_m)*sig2 + (w1*w2/w_m**2) * np.abs(_det(_outer(mu1-mu2, mu1-mu2)))**(1./d) * np.identity(d)
    return (w_m, mu_m, sig_m)


@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64[:], \
            float64[:,:], float64[:,:,:])', nopython=True)
def merge_full(w, mu, sig):
    n = mu.shape[0]
    d = mu.shape[1]
    w_m = np.sum(w)
    mu_m = np.zeros(d)
    sig_m = np.zeros((d,d))

    #mean calculation
    for i in range(n):
        mu_m += w[i]*mu[i]
    mu_m /= w_m

    #covariance calculation
    for i in range(n):
        sig_m += w[i] * ( sig[i] + _outer(mu[i]-mu_m, mu[i]-mu_m) )
    sig_m /= w_m

    return (w_m, mu_m, sig_m)



###########################################################################
# ISD: Integral Square Difference
# ref: Cost-Function-Based Gaussian Mixture Reduction for Target Tracking
###########################################################################
@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isd_diss(w1, mu1, sig1, w2, mu2, sig2):
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge(w1, mu1, sig1, w2, mu2, sig2)
    # ISD analytical computation between merged component and the pair of gaussians
    Jhr = w1*w_m * normal(mu1, mu_m, sig1+sig_m) + w2*w_m * normal(mu2, mu_m, sig2+sig_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = (w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig1))) + \
          (w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig2))) + \
          2*w1*w2*normal(mu1, mu2, sig1+sig2)
    return Jhh - 2*Jhr + Jrr


#normalized version
@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def isd_diss_(w1, mu1, sig1, w2, mu2, sig2):
    _w1 = w1 / (w1 + w2)
    _w2 = w2 / (w1 + w2)
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge(_w1, mu1, sig1, _w2, mu2, sig2)
    # ISD analytical computation between merged component and the pair of gaussians
    Jhr = _w1*w_m * normal(mu1, mu_m, sig1+sig_m) + _w2*w_m * normal(mu2, mu_m, sig2+sig_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = (_w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig1))) + \
          (_w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig2))) + \
          2*_w1*_w2*normal(mu1, mu2, sig1+sig2)
    return Jhh - 2*Jhr + Jrr


@numba.jit('float64 (float64[:], float64[:,:], float64[:,:,:])', nopython=True)
def isd_diss_full(w, mu, sig):
    # number of components
    c = len(w)
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge_full(w, mu, sig)
    # ISD computation between merge and components
    Jhr = 0.
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = 0.

    for i in range(c):  
        Jhr += w[i]*w_m * normal(mu[i], mu_m, sig[i]+sig_m)
    
    for i in range(c):
        for j in range(c):
            Jhh += w[i]*w[j] * normal(mu[i], mu[j], sig[i]+sig[j])

    return Jhh - 2*Jhr + Jrr


@numba.jit('float64 (float64[:], float64[:,:], float64[:,:,:], \
                     float64[:], float64[:,:], float64[:,:,:])', nopython=True)
def isd_diss_full_(w1, mu1, sig1, w2, mu2, sig2):
    # number of components
    h = len(w1)
    r = len(w2)

    # ISD computation between merge and components
    Jhr = 0.
    Jrr = 0.
    Jhh = 0.

    for i in range(h):
        for j in range(r):
            Jhr += w1[i]*w2[j] * normal(mu1[i], mu2[j], sig1[i]+sig2[j])

    for i in range(r):
        for j in range(r):
            Jrr += w2[i]*w2[j] * normal(mu2[i], mu2[j], sig2[i]+sig2[j])
    
    for i in range(h):
        for j in range(h):
            Jhh += w1[i]*w1[j] * normal(mu1[i], mu1[j], sig1[i]+sig1[j])

    return Jhh - 2*Jhr + Jrr



#################################################################
# KL-DIVERGENCE UPPER BOUND
# ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
#################################################################
@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def kl_diss(w1, mu1, sig1, w2, mu2, sig2):
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge(w1, mu1, sig1, w2, mu2, sig2)
    # KL divergence upper bound as proposed in: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    return 0.5*((w1+w2)*np.log(_det(sig_m)) - w1*np.log(_det(sig1)) - w2*np.log(_det(sig2)))


#################################################################
# MAIN GAUSSIAN REDUCTION FUNCTION
#################################################################
# def gaussian_reduction(c, mu, sig, n_comp, metric='KL', verbose=True):
#     if metric=='KL': 
#         _metric = kl_diss
#         isd_hist = list(); kl_hist = list()
#     elif metric=='ISD': 
#         _metric = isd_diss
#         isd_hist = list(); kl_hist = None
#     elif metric=='ISD_':
#         _metric = isd_diss_
#         isd_hist = list(); kl_hist = None
#     else: return None

#     d = mu.shape[1]
#     c = c.tolist()
#     mu = list(map(np.array, mu.tolist()))
#     if d==2: sig = [(s**2)*np.identity(2) for s in sig]
#     elif d==3: sig = [(s**2)*np.identity(3) for s in sig]

#     # indexes of the actual gaussian components
#     components = [i for i in range(len(c))]
#     structs_dict = {i:[i] for i in range(len(c))}
#     htree = {}
#     new_comp = len(c)

#     # main loop
#     while len(components)>n_comp:
#         m = len(components)
#         diss_min = np.inf
#         for i in range(m):
#             ii = components[i]
#             for j in range(i+1,m):
#                 jj = components[j]
#                 diss = _metric(c[ii], mu[ii], sig[ii], c[jj], mu[jj], sig[jj])
#                 if diss < diss_min: 
#                     i_min = i; j_min = j
#                     ii_min = ii; jj_min = jj 
#                     diss_min = diss
#         # compute the moment preserving  merged gaussian
#         w_m, mu_m, sig_m = merge(c[ii_min], mu[ii_min], sig[ii_min], 
#                                  c[jj_min], mu[jj_min], sig[jj_min])
        
#         if (metric=='ISD' or metric=='ISD_') and verbose:
#             print('Merged components {0} and {1} with {2} ISD dist'.format(ii_min, jj_min, diss_min))
#             isd_hist.append(diss_min)    
#         elif metric=='KL' and verbose:
#             ISD_diss = isd_diss(c[ii_min], mu[ii_min], sig[ii_min], c[jj_min], mu[jj_min], sig[jj_min])
#             print('Merged components {0} and {1} with {2} KL dist and {3} ISD dist'.format(ii_min, jj_min, diss_min, ISD_diss))
#             isd_hist.append(ISD_diss), kl_hist.append(diss_min)

#         # updating structures   
#         del components[max(i_min, j_min)]
#         del components[min(i_min, j_min)]
#         components.append(new_comp)
#         c.append(w_m); mu.append(mu_m); sig.append(sig_m)
#         htree[new_comp] = (min(ii_min,jj_min), max(ii_min,jj_min))
#         tmp = structs_dict[min(ii_min,jj_min)] + structs_dict[max(ii_min,jj_min)]
#         tmp.sort()
#         structs_dict[new_comp] = tmp
#         new_comp += 1
    
#     #return structs_dict,htree
#     return c,mu,sig




def gaussian_reduction(c, mu, sig, n_comp, metric=kl_diss, verbose=True):
    """
    Gaussian Mixture Reduction Through KL-upper bound approach
    """
    d = mu.shape[1]
    c = c.tolist()
    mu = list(map(np.array, mu.tolist()))
    if d==2: sig = [(s**2)*np.identity(2) for s in sig]
    elif d==3: sig = [(s**2)*np.identity(3) for s in sig]

    # indexes of the actual gaussian components
    components = [i for i in range(len(c))]
    structs_dict = {i:[i] for i in range(len(c))}
    htree = {}
    new_comp = len(c)

    # main loop
    while len(components)>n_comp:
        m = len(components)
        diss_min = np.inf
        for i in range(m):
            ii = components[i]
            for j in range(i+1,m):
                jj = components[j]
                diss = metric(c[ii], mu[ii], sig[ii], c[jj], mu[jj], sig[jj])
                if diss < diss_min: 
                    i_min = i; j_min = j
                    ii_min = ii; jj_min = jj 
                    diss_min = diss
        # compute the moment preserving  merged gaussian
        w_m, mu_m, sig_m = merge(c[ii_min], mu[ii_min], sig[ii_min], 
                                 c[jj_min], mu[jj_min], sig[jj_min])
          
        if verbose:
            ISD_diss = isd_diss(c[ii_min], mu[ii_min], sig[ii_min], c[jj_min], mu[jj_min], sig[jj_min])
            print('Merged components {0} and {1} with {2} KL dist and {3} ISD dist'.format(ii_min, jj_min, diss_min, ISD_diss))

        # updating structures   
        del components[max(i_min, j_min)]
        del components[min(i_min, j_min)]
        components.append(new_comp)
        c.append(w_m); mu.append(mu_m); sig.append(sig_m)
        htree[new_comp] = (min(ii_min,jj_min), max(ii_min,jj_min))
        tmp = structs_dict[min(ii_min,jj_min)] + structs_dict[max(ii_min,jj_min)]
        tmp.sort()
        structs_dict[new_comp] = tmp
        new_comp += 1
    
    #return structs_dict,htree
    # return c,mu,sig
    

def _compute_neighbors(mu_center, maxsig):
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
def build_diss_matrix(w, mu, sig, nn_indexes):
    M,max_neigh = nn_indexes.shape
    diss_matrix = -1.*np.ones((M,max_neigh))
    for i in range(M):
        for j in range(max_neigh):
            jj = nn_indexes[i,j]
            if jj==MAXINT: break
            diss_matrix[i,j] = kl_diss(w[i],mu[i],sig[i],w[jj],mu[jj],sig[jj])  
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
def update_structs(nn_indexes, diss_matrix, w, mu, sig, indexes, nindex, dindex):
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
                diss_matrix[i,j] = kl_diss(w[i],mu[i],sig[i],w[jj],mu[jj],sig[jj])
                flag1 = True
            elif jj==dindex and flag1:
                nn_indexes[i,j] = MAXINT
                diss_matrix[i,j] = -1
                flag2 = True
            elif jj==dindex and not flag1:
                nn_indexes[i,j] = nindex
                diss_matrix[i,j] = kl_diss(w[i],mu[i],sig[i],w[jj],mu[jj],sig[jj])
                flag2 = True
        if flag2:
            sorted_indexes = np.argsort(nn_indexes[i,:])
            nn_indexes[i,:] = (nn_indexes[i,:])[sorted_indexes]
            diss_matrix[i,:] = (diss_matrix[i,:])[sorted_indexes]

    # the special case...
    for j in range(max_neigh):
        jj = nn_indexes[nindex,j]
        if jj!=MAXINT:
            diss_matrix[nindex,j] = kl_diss(w[nindex],mu[nindex],sig[nindex],w[jj],mu[jj],sig[jj])
        else:
            diss_matrix[nindex,j] = -1


def mixture_reduction(w, mu, sig, n_comp, metric=kl_diss, isomorphic=False, verbose=True):
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
    if sig.ndim==1:
        maxsig = 5*np.max(sig)
        # if sig is 1-dimensional we convert it to its covariance matrix form
        sig = np.asarray( [(val**2)*np.identity(d) for val in sig] )
    else:
        maxsig = 5*max([np.max(np.linalg.eig(cov)[0])**(1./2) for cov in sig])

    indexes = np.arange(M, dtype=np.int32)
    nn,nn_indexes = _compute_neighbors(mu,maxsig)

    # idea: keep track that the k-th component was merged into the l-th positon
    merge_mapping = np.arange(M, dtype=np.int32)

    # max number of neighbors
    max_neigh = nn_indexes.shape[1]
    
    # computing the initial dissimilarity matrix
    diss_matrix = build_diss_matrix(w, mu, sig, nn_indexes)  
    
    # main loop
    while M>N:
        i_min, j_min = least_dissimilar(diss_matrix, indexes, nn_indexes)
        if verbose: print('Merged components {0} and {1}'.format(i_min, j_min))  
        w_m, mu_m, sig_m = merge(w[i_min], mu[i_min], sig[i_min], 
                                 w[j_min], mu[j_min], sig[j_min])
 
        # updating structures
        nindex = min(i_min,j_min) # index of the new component
        dindex = max(i_min,j_min) # index of the del component
        w[nindex] = w_m; mu[nindex] = mu_m; sig[nindex] = sig_m
        indexes = np.delete(indexes, get_index(indexes,dindex))
        update_merge_mapping(merge_mapping, nindex, dindex)
        nn_indexes[nindex] = radius_search(nn, mu_m, max_neigh, merge_mapping, nindex, dindex)
        update_structs(nn_indexes, diss_matrix, w, mu, sig, indexes, nindex, dindex)
        M -= 1

    # indexes of "alive" mixture components
    return w[indexes],mu[indexes],sig[indexes]


