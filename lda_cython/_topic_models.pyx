#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
The heavy-lifting is here in cython.

Draws from Allen Riddell's LDA library https://github.com/ariddell/lda
"""

from datetime import (datetime, timedelta)
import numpy as np
from libc.math cimport fabs
from cython.operator cimport (preincrement, predecrement)
from cython_gsl cimport (gsl_sf_lngamma as lngamma, gsl_sf_exp as exp,
                         gsl_sf_log as ln, gsl_rng, gsl_rng_mt19937,
                         gsl_rng_alloc, gsl_rng_set,
                         gsl_rng_uniform, gsl_rng_uniform_int,
                         gsl_ran_gaussian as gaussian)
from pypolyagamma import PyPolyaGamma


# we choose this number since it is a large prime
cdef unsigned int n_rands = 1000003
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)


cdef double[:] create_rands(unsigned int n_rands, seed=None):
    """
    Create array of uniformly random numbers on the interval [0, 1).
    """

    cdef:
        int i
        double[::1] rands = np.empty(n_rands, dtype=np.float64, order='C')
    if seed is not None:
        gsl_rng_set(r, seed)
    for i in range(n_rands):
        rands[i] = gsl_rng_uniform(r)
    return rands


cdef int[:] create_topic_lookup(unsigned int n_tokens, unsigned int n_topics,
                                seed=None):
    """
    Create array of uniformly random numbers on the interval [0, 1).
    """

    cdef:
        int i
        int[::1] topic_lookup = np.empty(n_tokens, dtype=np.intc, order='C')
    if seed is not None:
        gsl_rng_set(r, seed)
    for i in range(n_tokens):
        topic_lookup[i] = gsl_rng_uniform_int(r, n_topics)
    return topic_lookup


cdef int searchsorted(double[:] a, double v):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array a such that, if the corresponding
    elements in v were inserted before the indices, the order of a would be
    preserved.

    Like numpy.searchsorted
    (http://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html).
    """

    cdef:
        int imin = 0
        int imax = a.shape[0]
        int imid
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if v > a[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


cdef double loglikelihood_lda(int[:, :] nzw, int[:, :] ndz, int[:] nz,
                              double[:] alpha, double[:] beta, double sum_beta,
                              double lBeta_alpha, double lBeta_beta,
                              double lGamma_sum_alpha_nd):
    """
    Log likelihood calculation for LDA.

    This is an exact calculation.
    """

    cdef int k, d
    cdef int n_docs = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int n_terms = nzw.shape[1]
    cdef double ll = 0
    # calculate log p(w|z)
    ll += n_topics * lBeta_beta
    for k in range(n_topics):
        ll -= lngamma(sum_beta + nz[k])
        for w in range(n_terms):
            ll += lngamma(beta[w] + nzw[k, w])
    # calculate log p(z)
    ll += n_docs * lBeta_alpha
    ll -= lGamma_sum_alpha_nd
    for d in range(n_docs):
        for k in range(n_topics):
            ll += lngamma(alpha[k] + ndz[d, k])
    return ll


cdef double loglikelihood_slda(int[:, :] nzw, int[:, :] ndz, int[:] nz,
                               double[:] alpha, double[:] beta, double sum_beta,
                               double mu, double nu2, double sigma2,
                               double[:] eta, double[:] y, double[:, :] Z):
    """
    Log likelihood calculation for supervised LDA.

    This is not an exact calculation (constants not included).
    """

    cdef int k, d
    cdef int n_docs = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int n_terms = nzw.shape[1]
    cdef double ll = 0.
    cdef double eta_z = 0.
    # calculate log p(w|z) and log p(eta)
    for k in range(n_topics):
        ll -= lngamma(sum_beta + nz[k])
        ll -= (eta[k] - mu) * (eta[k] - mu) / 2 / nu2
        for w in range(n_terms):
            ll += lngamma(beta[w] + nzw[k, w])
    # calculate log p(z) and log p(y|eta)
    for d in range(n_docs):
        eta_z = 0.
        for k in range(n_topics):
            eta_z += eta[k] * Z[k, d]
            ll += lngamma(alpha[k] + ndz[d, k])
        ll -= (y[d] - eta_z) * (y[d] - eta_z) / 2 / sigma2
    return ll


cdef double loglikelihood_blslda(int[:, :] nzw, int[:, :] ndz, int[:] nz,
                                 double[:] alpha, double[:] beta, double sum_beta,
                                 double mu, double nu2, double b,
                                 double[:] eta, double[:] y, double[:, :] Z):
    """
    Log likelihood calculation for binary logistic supervised LDA.

    This is not an exact calculation (constants not included).
    """

    cdef int k, d
    cdef int n_docs = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int n_terms = nzw.shape[1]
    cdef double ll = 0.
    cdef double eta_z = 0.
    # calculate log p(w|z) and log p(eta)
    for k in range(n_topics):
        ll -= lngamma(sum_beta + nz[k])
        ll -= (eta[k] - mu) * (eta[k] - mu) / 2 / nu2
        for w in range(n_terms):
            ll += lngamma(beta[w] + nzw[k, w])
    # calculate log p(z) and log p(y|eta, z)
    for d in range(n_docs):
        eta_z = 0.
        for k in range(n_topics):
            eta_z += eta[k] * Z[k, d]
            ll += lngamma(alpha[k] + ndz[d, k])
        ll += b * (y[d] * eta_z - ln(1 + exp(eta_z)))
    return ll


cdef double loglikelihood_grtm(int[:, :] nzw, int[:, :] ndz, int[:] nz,
                               double[:] alpha, double[:] beta, double sum_beta,
                               double mu, double nu2, double b,
                               double[:, :] H, double[:] y, double[:] zeta):
    """
    Log likelihood calculation for generalized relational topic models with
    data augmentation.

    This is not an exact calculation (constants not included).
    """

    cdef:
        int k, _k, d
        int n_docs = ndz.shape[0]
        int n_topics = ndz.shape[1]
        int n_terms = nzw.shape[1]
        int n_edges = y.shape[0]
        double ll = 0.
    # calculate log p(w|z) and log p(H)
    for k in range(n_topics):
        ll -= lngamma(sum_beta + nz[k])
        for _k in range(n_topics):
            ll -= (H[k, _k] - mu) * (H[k, _k] - mu) / 2 / nu2
        for w in range(n_terms):
            ll += lngamma(beta[w] + nzw[k, w])
    # calculate log p(z)
    for d in range(n_docs):
        for k in range(n_topics):
            ll += lngamma(alpha[k] + ndz[d, k])
    # calculate log p(y|H, z)
    for e in range(n_edges):
        ll += b * (y[e] * zeta[e] - ln(1 + exp(zeta[e])))
    return ll


cdef double loglikelihood_rtm(int[:, :] nzw, int[:, :] ndz, int[:] nz,
                              double[:] alpha, double[:] beta, double sum_beta,
                              double mu, double nu, double sigma2,
                              double[:] eta, double[:] bold_z):
    """
    Log likelihood calculation for RTM.

    Only non-constant terms are included in this calculation.
    """

    cdef int k, d
    cdef int n_docs = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int n_terms = nzw.shape[1]
    cdef double ll = 0
    # calculate log p(w|z)
    for k in range(n_topics):
        ll -= lngamma(sum_beta + nz[k])
        for w in range(n_terms):
            ll += lngamma(beta[w] + nzw[k, w])
    # calculate log p(z)
    for d in range(n_docs):
        for k in range(n_topics):
            ll += lngamma(alpha[k] + ndz[d, k])
    # calculate log p(eta) and log p(y|eta,z)
    for k in range(n_topics):
        ll += eta[k] * bold_z[k] / nu - (eta[k] - mu) * (eta[k] - mu) / 2 / sigma2
    return ll

cdef double loglikelihood_blhslda(int[:, :] nzw, int[:, :] ndz, int[:] nz,
                                 double[:] alpha, double[:] beta, double sum_beta,
                                 double mu, double nu2, double b,
                                 double[:,:] eta, int[:,:] y, double[:, :] Z):
    """
    Log likelihood calculation for binary logistic hierarchical supervised LDA.

    This is not an exact calculation (constants not included).
    """

    cdef int k, d, l
    cdef int n_docs = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int n_terms = nzw.shape[1]
    cdef int n_labels = eta.shape[0]
    cdef double ll = 0.
    cdef double eta_z = 0.
    # calculate log p(w|z) and log p(eta)
    for l in range(n_labels):
        for k in range(n_topics):
            ll -= lngamma(sum_beta + nz[k])
            ll -= (eta[l, k] - mu) * (eta[l, k] - mu) / 2 / nu2
            for w in range(n_terms):
                ll += lngamma(beta[w] + nzw[k, w])
    # calculate log p(z) and log p(y|eta, z)
    for d in range(n_docs):
        eta_z = 0.
        for l in range(n_labels):
            for k in range(n_topics):
                eta_z += eta[l, k] * Z[k, d]
                ll += lngamma(alpha[k] + ndz[d, k])
            ll += b * (y[d,l] * eta_z - ln(1 + exp(eta_z)))
    return ll


cdef print_progress(start_time, int n_report_iter, int i,
                    double lL_now, double lL_last):
    """
    Print progress of iterations.
    """

    if i > 0 and i % n_report_iter == 0:
        now_time = datetime.now()
        print('{} {} elapsed, iter {:>4}, LL {:.4f}, {:.2f}% change from last'
            .format(now_time,
                    now_time - start_time,
                    i,
                    lL_now,
                    (lL_now - lL_last) / fabs(lL_last) * 100))


def estimate_matrix(int[:, :] counts, double[:] psuedo_counts, int n_things):
    """
    Create estimates for theta and phi from counts.
    """

    mat = np.asarray(counts) + np.tile(psuedo_counts, (n_things, 1))
    return (mat.T / mat.sum(axis=1)).T


def iterated_pseudo_counts(doc_lookup, term_lookup, int n_docs,
                           double[:] alpha, double[:] beta, double[:, :] phi,
                           int max_iter, double tol):
    """
    Estimate the topic distributions of new documents using the
    iterated pseudo-counts method mentioned in Wallach et al. (2009) and
    derived in Buntine (2009).
    """

    cdef:
        int d, i, k, s, w, n_tokens_d
        int n_topics = phi.shape[0]
        int[::1] term_lookup_d
        double sum_over_k, _tmp_double
        double[::1] q_sum
        double[:, ::1] q, q_new
        double[:, ::1] theta = np.empty((n_docs, n_topics), dtype=np.float64, order='C')
    for d in range(n_docs):
        term_lookup_d = np.ascontiguousarray(term_lookup[doc_lookup == d])
        n_tokens_d = term_lookup_d.shape[0]
        # initialize proposal distribution q
        q = np.empty((n_tokens_d, n_topics), dtype=np.float64, order='C')
        for i in range(n_tokens_d):
            w = term_lookup_d[i]
            sum_over_k = 0.
            for k in range(n_topics):
                _tmp_double = phi[k, w] * alpha[k]
                q[i, k] = _tmp_double
                sum_over_k += _tmp_double
            for k in range(n_topics):
                q[i, k] /= sum_over_k
        # do fixed point iteration
        q_new = np.empty_like(q, dtype=np.float64, order='C')
        for s in range(max_iter):
            # q_sum is
            q_sum = np.zeros(n_topics, dtype=np.float64, order='C')
            for i in range(n_tokens_d):
                for k in range(n_topics):
                    q_sum[k] += q[i, k]
            for i in range(n_tokens_d):
                w = term_lookup_d[i]
                sum_over_k = 0.
                for k in range(n_topics):
                    _tmp_double = phi[k, w] * (alpha[k] + q_sum[k] - q[i, k])
                    q_new[i, k] = _tmp_double
                    sum_over_k += _tmp_double
                for k in range(n_topics):
                    q_new[i, k] /= sum_over_k
            # return if difference between iterations is small
            sum_over_k = 0.
            for i in range(n_tokens_d):
                for k in range(n_topics):
                    sum_over_k += fabs(q_new[i, k] - q[i, k])
            # set q here in case we break
            q = q_new
            if sum_over_k < tol:
                break
        # calculate topic distributions
        q_sum = np.zeros(n_topics, dtype=np.float64, order='C')
        sum_over_k = 0.
        for i in range(n_tokens_d):
            for k in range(n_topics):
                q_sum[k] += q[i, k]
                sum_over_k += q[i, k]
        for k in range(n_topics):
            theta[d, k] = q_sum[k] / sum_over_k
    return np.array(theta)


def gibbs_sampler_lda(int n_iter, int n_report_iter,
                      int n_topics, int n_docs,
                      int n_terms, int n_tokens,
                      double[:] alpha, double[:] beta,
                      int[:] doc_lookup, int[:] term_lookup,
                      seed=None):
    """
    Perform (collapsed) Gibbs sampling inference for LDA.
    """

    cdef:
        int i, j, k, d, w, z, new_z
        double p_sum, uval
        double sum_alpha = 0.
        double sum_beta = 0.
        double lBeta_alpha = 0.
        double lBeta_beta = 0.
        double lGamma_sum_alpha_nd = 0.
        int[:] topic_lookup = create_topic_lookup(n_tokens, n_topics, seed)
        # log likelihoods
        double[::1] lL = np.empty(n_iter, dtype=np.float64, order='C')
        # number of tokens in document d assigned to topic z, shape = (n_docs, n_topics)
        int[:, ::1] ndz = np.zeros((n_docs, n_topics), dtype=np.intc, order='C')
        # number of tokens assigned to topic z equal to term w, shape = (n_topics, n_terms)
        int[:, ::1] nzw = np.zeros((n_topics, n_terms), dtype=np.intc, order='C')
        # number of tokens assigned to topic k, shape = (n_topics,)
        int[::1] nz = np.zeros(n_topics, dtype=np.intc, order='C')
        # number of tokens in doc d, shape = (n_docs,)
        int[::1] nd = np.zeros(n_docs, dtype=np.intc, order='C')
        # (weighted) probabilities for the discrete distribution
        double[::1] p_cumsum = np.empty(n_topics, dtype=np.float64, order='C')
        # preallocate uniformly random numbers on the interval [0, 1)
        double[:] rands = create_rands(n_rands=n_rands, seed=seed)
        int u = 0
    # initialize counts
    for j in range(n_tokens):
        preincrement(ndz[doc_lookup[j], topic_lookup[j]])
        preincrement(nzw[topic_lookup[j], term_lookup[j]])
        preincrement(nz[topic_lookup[j]])
        preincrement(nd[doc_lookup[j]])
    # initialize sum_alpha, lBeta_alpha
    for k in range(n_topics):
        sum_alpha += alpha[k]
        lBeta_alpha += lngamma(alpha[k])
    lBeta_alpha -= lngamma(sum_alpha)
    # initialize sum_beta, lBeta_beta
    for w in range(n_terms):
        sum_beta += beta[w]
        lBeta_beta += lngamma(beta[w])
    lBeta_beta -= lngamma(sum_beta)
    # initialize lGamma_sum_alpha_nd
    for d in range(n_docs):
        lGamma_sum_alpha_nd += lngamma(sum_alpha + nd[d])
    # iterate
    start_time = datetime.now()
    print('{} start iterations'.format(start_time))
    for i in range(n_iter):
        for j in range(n_tokens):
            d = doc_lookup[j]
            w = term_lookup[j]
            z = topic_lookup[j]
            predecrement(ndz[d, z])
            predecrement(nzw[z, w])
            predecrement(nz[z])
            p_sum = 0.
            for k in range(n_topics):
                p_sum += (nzw[k, w] + beta[w]) / (nz[k] + sum_beta) * (ndz[d, k] + alpha[k])
                p_cumsum[k] = p_sum
            preincrement(u)
            if u == n_rands:
                u = 0
            uval = rands[u] * p_sum
            new_z = topic_lookup[j] = searchsorted(p_cumsum, uval)
            preincrement(ndz[d, new_z])
            preincrement(nzw[new_z, w])
            preincrement(nz[new_z])
        lL[i] = loglikelihood_lda(nzw, ndz, nz, alpha, beta, sum_beta,
                                  lBeta_alpha, lBeta_beta, lGamma_sum_alpha_nd)
        # print progress
        print_progress(start_time, n_report_iter, i, lL[i], lL[i - n_report_iter])
    # populate the topic and word distributions
    theta = estimate_matrix(ndz, alpha, n_docs)
    phi = estimate_matrix(nzw, beta, n_topics)
    return theta, phi, np.asarray(lL)


def gibbs_sampler_slda(int n_iter, int n_report_iter,
                       int n_topics, int n_docs,
                       int n_terms, int n_tokens,
                       double[:] alpha, double[:] beta,
                       double mu, double nu2, double sigma2,
                       int[:] doc_lookup, int[:] term_lookup,
                       double[:] y, seed=None):
    """
    Perform (collapsed) Gibbs sampling inference for supervised LDA.
    """

    cdef:
        int i, j, k, d, w, z, new_z
        double p_sum, uval, y_sum
        double sum_alpha = 0.
        double sum_beta = 0.
        int[:] topic_lookup = create_topic_lookup(n_tokens, n_topics, seed)
        # log likelihoods
        double[::1] lL = np.empty(n_iter, dtype=np.float64, order='C')
        # number of tokens in document d assigned to topic z, shape = (n_docs, n_topics)
        int[:, ::1] ndz = np.zeros((n_docs, n_topics), dtype=np.intc, order='C')
        # number of tokens assigned to topic z equal to term w, shape = (n_topics, n_terms)
        int[:, ::1] nzw = np.zeros((n_topics, n_terms), dtype=np.intc, order='C')
        # number of tokens assigned to topic k, shape = (n_topics,)
        int[::1] nz = np.zeros(n_topics, dtype=np.intc, order='C')
        # number of tokens in doc d, shape = (n_docs,)
        int[::1] nd = np.zeros(n_docs, dtype=np.intc, order='C')
        # (weighted) probabilities for the discrete distribution
        double[::1] p_cumsum = np.empty(n_topics, dtype=np.float64, order='C')
        # preallocate uniformly random numbers on the interval [0, 1)
        double[:] rands = create_rands(n_rands=n_rands, seed=seed)
        int u = 0
        # regression coefficients
        double[:, ::1] eta = np.ascontiguousarray(
            np.tile(mu, (n_iter + 1, n_topics)), dtype=np.float64)
        double[:, ::1] etand = np.empty((n_docs, n_topics), dtype=np.float64, order='C')
        double[::1] eta_tmp = np.empty(n_topics, dtype=np.float64, order='C')
    # initialize counts
    for j in range(n_tokens):
        preincrement(ndz[doc_lookup[j], topic_lookup[j]])
        preincrement(nzw[topic_lookup[j], term_lookup[j]])
        preincrement(nz[topic_lookup[j]])
        preincrement(nd[doc_lookup[j]])
    # initialize sum_alpha, lBeta_alpha
    for k in range(n_topics):
        sum_alpha += alpha[k]
    # initialize sum_beta, lBeta_beta
    for w in range(n_terms):
        sum_beta += beta[w]
    # define numpy variables
    Inu2 = np.identity(n_topics) / nu2
    # iterate
    start_time = datetime.now()
    print('{} start iterations'.format(start_time))
    for i in range(n_iter):
        # initialize etand for iteration i
        for d in range(n_docs):
            for k in range(n_topics):
                etand[d, k] = eta[i, k] / nd[d]
        # sample z
        for j in range(n_tokens):
            d = doc_lookup[j]
            w = term_lookup[j]
            z = topic_lookup[j]
            predecrement(ndz[d, z])
            predecrement(nzw[z, w])
            predecrement(nz[z])
            p_sum = 0.
            y_sum = y[d]
            for k in range(n_topics):
                y_sum -= etand[d, k] * ndz[d, k]
            y_sum = 2 * y_sum
            for k in range(n_topics):
                p_sum += (nzw[k, w] + beta[w]) \
                    / (nz[k] + sum_beta) \
                    * (ndz[d, k] + alpha[k]) \
                    * exp(etand[d, k] / 2 / sigma2 * (y_sum - etand[d, k]))
                p_cumsum[k] = p_sum
            preincrement(u)
            if u == n_rands:
                u = 0
            uval = rands[u] * p_sum
            new_z = topic_lookup[j] = searchsorted(p_cumsum, uval)
            preincrement(ndz[d, new_z])
            preincrement(nzw[new_z, w])
            preincrement(nz[new_z])
        # sample eta
        # (actually, we are setting eta to mean)
        Z = (np.asarray(ndz) / np.asarray(nd)[:, np.newaxis]).T
        eta_tmp = np.linalg.solve(Inu2 + np.dot(Z, Z.T) / sigma2,
                                  np.dot(Z, np.asarray(y) / sigma2))
        for k in range(n_topics):
            eta[i + 1, k] = eta_tmp[k]
        lL[i] = loglikelihood_slda(nzw, ndz, nz, alpha, beta, sum_beta,
                                   mu, nu2, sigma2, eta[i + 1], y, Z)
        # print progress
        print_progress(start_time, n_report_iter, i, lL[i], lL[i - n_report_iter])
    # populate the topic and word distributions
    theta = estimate_matrix(ndz, alpha, n_docs)
    phi = estimate_matrix(nzw, beta, n_topics)
    return theta, phi, np.asarray(eta), np.asarray(lL)


def gibbs_sampler_blslda(int n_iter, int n_report_iter,
                         int n_topics, int n_docs,
                         int n_terms, int n_tokens,
                         double[:] alpha, double[:] beta,
                         double mu, double nu2, double b,
                         int[:] doc_lookup, int[:] term_lookup,
                         double[:] y, seed):
    """
    Perform collapsed Gibbs sampling inference for binary logistic supervised
    LDA using Polson et al.'s data augmentation strategy[1] and Zhu et al.'s
    regularization strategy[2].

    1. Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian Inference
    for Logistic Models Using Pólya–Gamma Latent Variables. Journal of the
    American Statistical Association, 108(504), 1339–1349.
    http://doi.org/10.1080/01621459.2013.829001

    2. Zhu, J., Zheng, X., & Zhang, B. (2013, October). Improved Bayesian
    Logistic Supervised Topic Models with Data Augmentation. arXiv.org.
    """

    cdef:
        int i, j, k, d, w, z, new_z
        double p_sum, uval, eta_sum, kappa_sum
        double sum_alpha = 0.
        double sum_beta = 0.
        int[:] topic_lookup = create_topic_lookup(n_tokens, n_topics, seed)
        # log likelihoods
        double[::1] lL = np.empty(n_iter, dtype=np.float64, order='C')
        # number of tokens in document d assigned to topic z, shape = (n_docs, n_topics)
        int[:, ::1] ndz = np.zeros((n_docs, n_topics), dtype=np.intc, order='C')
        # number of tokens assigned to topic z equal to term w, shape = (n_topics, n_terms)
        int[:, ::1] nzw = np.zeros((n_topics, n_terms), dtype=np.intc, order='C')
        # number of tokens assigned to topic k, shape = (n_topics,)
        int[::1] nz = np.zeros(n_topics, dtype=np.intc, order='C')
        # number of tokens in doc d, shape = (n_docs,)
        int[::1] nd = np.zeros(n_docs, dtype=np.intc, order='C')
        # (weighted) probabilities for the discrete distribution
        double[::1] p_cumsum = np.empty(n_topics, dtype=np.float64, order='C')
        # preallocate uniformly random numbers on the interval [0, 1)
        double[:] rands = create_rands(n_rands=n_rands, seed=seed)
        int u = 0
        # regression coefficients
        double[:, ::1] eta = np.ascontiguousarray(
            np.tile(mu, (n_iter + 1, n_topics)), dtype=np.float64)
        double[:, ::1] etand = np.empty((n_docs, n_topics), dtype=np.float64, order='C')
        double[::1] eta_mean = np.empty(n_topics, dtype=np.float64, order='C')
        # omega: notice I'm initializing omega here
        double[::1] omega = np.ascontiguousarray(np.repeat(1., n_docs))
        # kappa: a transformation of y
        double[::1] kappa = b * (np.asarray(y) - 0.5)
    # initialize counts
    for j in range(n_tokens):
        preincrement(ndz[doc_lookup[j], topic_lookup[j]])
        preincrement(nzw[topic_lookup[j], term_lookup[j]])
        preincrement(nz[topic_lookup[j]])
        preincrement(nd[doc_lookup[j]])
    # initialize sum_alpha, lBeta_alpha
    for k in range(n_topics):
        sum_alpha += alpha[k]
    # initialize sum_beta, lBeta_beta
    for w in range(n_terms):
        sum_beta += beta[w]
    # define numpy variables
    Inu2 = np.identity(n_topics) / nu2
    munu2 = np.repeat(mu / nu2, n_topics)
    # define PolyaGamma sampler
    pg_rng = PyPolyaGamma(seed=seed or 42)
    # iterate
    start_time = datetime.now()
    print('{} start iterations'.format(start_time))
    for i in range(n_iter):
        # sample omega
        for d in range(n_docs):
            eta_sum = 0.
            for k in range(n_topics):
                # initialize etand for iteration i
                etand[d, k] = eta[i, k] / nd[d]
                eta_sum += etand[d, k] * ndz[d, k]
            omega[d] = pg_rng.pgdraw(b, eta_sum)
        # sample z
        for j in range(n_tokens):
            d = doc_lookup[j]
            w = term_lookup[j]
            z = topic_lookup[j]
            predecrement(ndz[d, z])
            predecrement(nzw[z, w])
            predecrement(nz[z])
            p_sum = 0.
            kappa_sum = kappa[d]
            for k in range(n_topics):
                kappa_sum -= omega[d] * etand[d, k] * ndz[d, k]
            for k in range(n_topics):
                p_sum += (nzw[k, w] + beta[w]) \
                    / (nz[k] + sum_beta) \
                    * (ndz[d, k] + alpha[k]) \
                    * exp(etand[d, k] * (kappa_sum - omega[d] / 2 * etand[d, k]))
                p_cumsum[k] = p_sum
            preincrement(u)
            if u == n_rands:
                u = 0
            uval = rands[u] * p_sum
            new_z = topic_lookup[j] = searchsorted(p_cumsum, uval)
            preincrement(ndz[d, new_z])
            preincrement(nzw[new_z, w])
            preincrement(nz[new_z])
        # sample eta
        Z = (np.asarray(ndz) / np.asarray(nd)[:, np.newaxis]).T
        Omega = np.asarray(omega)[np.newaxis, :]
        eta_mean = np.linalg.solve(Inu2 + np.dot(Z * Omega, Z.T),
                                   munu2 + np.dot(Z, kappa))
        # TODO currently setting eta to mean, but need to sample
        for k in range(n_topics):
            eta[i + 1, k] = eta_mean[k]
        # compute log-likelihood
        lL[i] = loglikelihood_blslda(nzw, ndz, nz, alpha, beta, sum_beta,
                                     mu, nu2, b, eta[i + 1], y, Z)
        # print progress
        print_progress(start_time, n_report_iter, i, lL[i], lL[i - n_report_iter])
    # populate the topic and word distributions
    theta = estimate_matrix(ndz, alpha, n_docs)
    phi = estimate_matrix(nzw, beta, n_topics)
    return theta, phi, np.asarray(eta), np.asarray(lL)


def gibbs_sampler_grtm(int n_iter, int n_report_iter,
                       int n_topics, int n_docs,
                       int n_terms, int n_tokens, int n_edges,
                       double[:] alpha, double[:] beta,
                       double mu, double nu2, double b,
                       int[:] doc_lookup, int[:] term_lookup,
                       int[:] out_docs, int[:] out_edges,
                       int[:] in_docs, int[:] in_edges,
                       int[:] edge_tail, int[:] edge_head,
                       double[:] y, seed):
    """
    Perform collapsed Gibbs sampling inference for relational topic models
    using Polson et al.'s data augmentation strategy[1] and Zhu et al.'s
    regularization strategy[2].

    1. Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian Inference
    for Logistic Models Using Pólya–Gamma Latent Variables. Journal of the
    American Statistical Association, 108(504), 1339–1349.
    http://doi.org/10.1080/01621459.2013.829001

    2. Chen, N., Zhu, J., Xia, F., & Zhang, B. (2013). Generalized relational topic
    models with data augmentation. Presented at the IJCAI'13: Proceedings of the
    Twenty-Third international joint conference on Artificial Intelligence,  AAAI
    Press.
    """

    cdef:
        int i, j, k, k1, k2, d, d1, d2, e, w, z, new_z, n, Nd
        double p_sum, uval, H_col_sum, H_row_sum, kappa_sum, zeta_sum
        double sum_alpha = 0.
        double sum_beta = 0.
        int[:] topic_lookup = create_topic_lookup(n_tokens, n_topics, seed)
        # log likelihoods
        double[::1] lL = np.empty(n_iter, dtype=np.float64, order='C')
        # number of tokens in document d assigned to topic z, shape = (n_docs, n_topics)
        int[:, ::1] ndz = np.zeros((n_docs, n_topics), dtype=np.intc, order='C')
        # number of tokens assigned to topic z equal to term w, shape = (n_topics, n_terms)
        int[:, ::1] nzw = np.zeros((n_topics, n_terms), dtype=np.intc, order='C')
        # number of tokens assigned to topic k, shape = (n_topics,)
        int[::1] nz = np.zeros(n_topics, dtype=np.intc, order='C')
        # number of tokens in doc d, shape = (n_docs,)
        int[::1] nd = np.zeros(n_docs, dtype=np.intc, order='C')
        # (weighted) probabilities for the discrete distribution
        double[::1] p_cumsum = np.empty(n_topics, dtype=np.float64, order='C')
        # preallocate uniformly random numbers on the interval [0, 1)
        double[:] rands = create_rands(n_rands=n_rands, seed=seed)
        int u = 0
        # zeta: TODO
        double[::1] zeta = np.empty(n_edges, dtype=np.float64, order='C')
        # regression coefficients
        double[:, :, ::1] H = np.ascontiguousarray(
            np.tile(mu, (n_iter + 1, n_topics, n_topics)), dtype=np.float64)
        # 0 = row, 1 = column
        double[:, ::1] Hznd = np.empty((n_edges, n_topics), dtype=np.float64, order='C')
        double[:, ::1] HTznd = np.empty((n_edges, n_topics), dtype=np.float64, order='C')
        double[::1] eta_mean = np.empty(n_topics * n_topics, dtype=np.float64, order='C')
        # omega: notice I'm initializing omega here
        double[::1] omega = np.ascontiguousarray(np.repeat(1., n_edges))
        # kappa: a transformation of y
        double[::1] kappa = b * (np.asarray(y) - 0.5)
    # initialize counts
    for j in range(n_tokens):
        preincrement(ndz[doc_lookup[j], topic_lookup[j]])
        preincrement(nzw[topic_lookup[j], term_lookup[j]])
        preincrement(nz[topic_lookup[j]])
        preincrement(nd[doc_lookup[j]])
    # initialize sum_alpha, lBeta_alpha
    for k in range(n_topics):
        sum_alpha += alpha[k]
    # initialize sum_beta, lBeta_beta
    for w in range(n_terms):
        sum_beta += beta[w]
    # define numpy variables
    Inu2 = np.identity(n_topics * n_topics) / nu2
    munu2 = np.repeat(mu / nu2, n_topics * n_topics)
    # define PolyaGamma sampler
    pg_rng = PyPolyaGamma(seed=seed or 42)
    # iterate
    start_time = datetime.now()
    print('{} start iterations'.format(start_time))
    for i in range(n_iter):
        # sample omega, and initialize Hznd and zeta for iteration i
        for e in range(n_edges):
            d1 = edge_tail[e]
            d2 = edge_head[e]
            Nd = nd[d1] * nd[d2]
            zeta_sum = 0.
            for k1 in range(n_topics):
                H_row_sum = 0.
                H_col_sum = 0.
                for k2 in range(n_topics):
                    zeta_sum += ndz[d1, k1] * ndz[d2, k2] * H[i, k1, k2]
                    H_row_sum += H[i, k1, k2] * ndz[d2, k2]
                    H_col_sum += H[i, k2, k1] * ndz[d1, k2]
                Hznd[e, k1] = H_row_sum / Nd
                HTznd[e, k1] = H_col_sum / Nd
            zeta[e] = zeta_sum / Nd
            omega[e] = pg_rng.pgdraw(b, zeta[e])
        # sample z
        for j in range(n_tokens):
            d = doc_lookup[j]
            w = term_lookup[j]
            z = topic_lookup[j]
            predecrement(ndz[d, z])
            predecrement(nzw[z, w])
            predecrement(nz[z])
            p_sum = 0.
            for k in range(n_topics):
                kappa_sum = 0.
                for n in range(in_docs[d], in_docs[d + 1]):
                    e = in_edges[n]
                    d1 = edge_tail[e]
                    Nd = nd[d1] * nd[d]
                    zeta_sum = zeta[e]
                    for k1 in range(n_topics):
                        zeta_sum -= ndz[d1, k1] * H[i, k1, z] / Nd
                    kappa_sum += (kappa[e] - omega[e] * zeta_sum
                                  - omega[e] / 2 * HTznd[e, k]) * HTznd[e, k]
                for n in range(out_docs[d], out_docs[d + 1]):
                    e = out_edges[n]
                    d2 = edge_head[e]
                    Nd = nd[d] * nd[d2]
                    zeta_sum = zeta[e]
                    for k2 in range(n_topics):
                        zeta_sum -= ndz[d2, k2] * H[i, z, k2] / Nd
                    kappa_sum += (kappa[e] - omega[e] * zeta_sum
                                  - omega[e] / 2 * Hznd[e, k]) * Hznd[e, k]
                p_sum += (nzw[k, w] + beta[w]) \
                    / (nz[k] + sum_beta) \
                    * (ndz[d, k] + alpha[k]) \
                    * exp(kappa_sum)
                p_cumsum[k] = p_sum
            preincrement(u)
            if u == n_rands:
                u = 0
            uval = rands[u] * p_sum
            new_z = topic_lookup[j] = searchsorted(p_cumsum, uval)
            preincrement(ndz[d, new_z])
            preincrement(nzw[new_z, w])
            preincrement(nz[new_z])
            # TODO update zeta, Hznd, HTznd
        # sample eta
        _Z = np.asarray(ndz) / np.asarray(nd)[:, np.newaxis]
        Z = np.empty((n_topics * n_topics, n_edges), dtype=np.float64, order='C')
        for e in range(n_edges):
            Z[:, e] = np.kron(_Z[edge_head[e]], _Z[edge_tail[e]])
        Omega = np.asarray(omega)[np.newaxis, :]
        eta_mean = np.linalg.solve(Inu2 + np.dot(Z * Omega, Z.T),
                                   munu2 + np.dot(Z, kappa))
        # TODO currently setting eta to mean, but need to sample
        for k1 in range(n_topics):
            for k2 in range(n_topics):
                H[i + 1, k1, k2] = eta_mean[k1 + (k2 * n_topics)]
        # compute log-likelihood
        lL[i] = loglikelihood_grtm(nzw, ndz, nz, alpha, beta, sum_beta,
                                   mu, nu2, b, H[i + 1], y, zeta)
        # print progress
        print_progress(start_time, n_report_iter, i, lL[i], lL[i - n_report_iter])
    # populate the topic and word distributions
    theta = estimate_matrix(ndz, alpha, n_docs)
    phi = estimate_matrix(nzw, beta, n_topics)
    return theta, phi, np.asarray(H), np.asarray(lL)


def gibbs_sampler_rtm(int n_iter, int n_report_iter,
                      int n_topics, int n_docs,
                      int n_terms, int n_tokens,
                      double[:] alpha, double[:] beta,
                      double mu, double sigma2, double nu,
                      int[:] doc_lookup, int[:] term_lookup,
                      adj_mat, seed=None):
    """
    Perform (collapsed) Gibbs sampling inference for RTM.
    """

    cdef:
        int i, j, k, d, w, z, new_z, _d, n
        double p_sum, uval, d_sum, nd_d, nd__d
        double sum_alpha = 0.
        double sum_beta = 0.
        int[:] topic_lookup = create_topic_lookup(n_tokens, n_topics, seed)
        # log likelihoods
        double[::1] lL = np.empty(n_iter, dtype=np.float64, order='C')
        # number of tokens in document d assigned to topic z, shape = (n_docs, n_topics)
        int[:, ::1] ndz = np.zeros((n_docs, n_topics), dtype=np.intc, order='C')
        # number of tokens assigned to topic z equal to term w, shape = (n_topics, n_terms)
        int[:, ::1] nzw = np.zeros((n_topics, n_terms), dtype=np.intc, order='C')
        # number of tokens assigned to topic k, shape = (n_topics,)
        int[::1] nz = np.zeros(n_topics, dtype=np.intc, order='C')
        # number of tokens in doc d, shape = (n_docs,)
        int[::1] nd = np.zeros(n_docs, dtype=np.intc, order='C')
        # (weighted) probabilities for the discrete distribution
        double[::1] p_cumsum = np.empty(n_topics, dtype=np.float64, order='C')
        # preallocate uniformly random numbers on the interval [0, 1)
        double[:] rands = create_rands(n_rands=n_rands, seed=seed)
        int u = 0
        # document network variables
        int[::1] n_neighbors = np.ascontiguousarray(adj_mat.getnnz(axis=1))
        int[::1] neighbor_index = np.zeros(n_docs, dtype=np.intc, order='C')
        int[:, ::1] neighbors = np.zeros((n_docs, np.asarray(n_neighbors).max()),
                                         dtype=np.intc, order='C')
        # bold_z is nu times the bold_z in the write-up
        double[::1] bold_z = np.zeros(n_topics, dtype=np.float64, order='C')
        double[:, ::1] ndznd = np.zeros((n_docs, n_topics), dtype=np.float64,
                                        order='C')
        double[:, ::1] eta = np.ascontiguousarray(
            np.tile(mu, (n_iter + 1, n_topics)), dtype=np.float64)
    # initialize counts
    for j in range(n_tokens):
        preincrement(ndz[doc_lookup[j], topic_lookup[j]])
        preincrement(nzw[topic_lookup[j], term_lookup[j]])
        preincrement(nz[topic_lookup[j]])
        preincrement(nd[doc_lookup[j]])
    # initialize sum_alpha, lBeta_alpha
    for k in range(n_topics):
        sum_alpha += alpha[k]
    # initialize sum_beta, lBeta_beta
    for w in range(n_terms):
        sum_beta += beta[w]
    # initialize neighbors
    for d, _d in zip(*adj_mat.nonzero()):
        neighbors[d, neighbor_index[d]] = _d
        neighbor_index[d] += 1
    # initialize ndznd
    for d in range(n_docs):
        for n in range(n_neighbors[d]):
            _d = neighbors[d, n]
            nd__d = <double>nd[_d]
            for k in range(n_topics):
                ndznd[d, k] += ndz[_d, k] / nd__d
    # initialize bold_z
    for d in range(n_docs):
        nd_d = <double>nd[d]
        for n in range(n_neighbors[d]):
            _d = neighbors[d, n]
            nd__d = <double>nd[_d]
            for k in range(n_topics):
                bold_z[k] += ndz[d, k] / nd_d * ndz[_d, k] / nd__d
    # divide by 2 because we double-counted the edges
    for k in range(n_topics):
        bold_z[k] = bold_z[k] / 2.
    # iterate
    start_time = datetime.now()
    print('{} start iterations'.format(start_time))
    for i in range(n_iter):
        for j in range(n_tokens):
            d = doc_lookup[j]
            w = term_lookup[j]
            z = topic_lookup[j]
            predecrement(ndz[d, z])
            predecrement(nzw[z, w])
            predecrement(nz[z])
            p_sum = 0.
            for k in range(n_topics):
                p_sum += (nzw[k, w] + beta[w]) / (nz[k] + sum_beta) * (ndz[d, k] + alpha[k]) * \
                    exp(eta[i, k] / nd[d] / nu * ndznd[d, k])
                p_cumsum[k] = p_sum
            preincrement(u)
            if u == n_rands:
                u = 0
            uval = rands[u] * p_sum
            new_z = topic_lookup[j] = searchsorted(p_cumsum, uval)
            preincrement(ndz[d, new_z])
            preincrement(nzw[new_z, w])
            preincrement(nz[new_z])
            for n in range(n_neighbors[d]):
                _d = neighbors[d, n]
                nd_d = <double>nd[d]
                nd__d = <double>nd[_d]
                ndznd[_d, z] -= 1. / nd_d
                ndznd[_d, new_z] += 1. / nd_d
                bold_z[z] -= ndz[_d, z] / nd__d / nd_d
                bold_z[new_z] += ndz[_d, new_z] / nd__d / nd_d
        for k in range(n_topics):
            eta[i + 1, k] = gaussian(r, sigma2) + mu + sigma2 * bold_z[k] / nu
        lL[i] = loglikelihood_rtm(nzw, ndz, nz, alpha, beta, sum_beta,
                                  mu, nu, sigma2, eta[i + 1], bold_z)
        # print progress
        print_progress(start_time, n_report_iter, i, lL[i], lL[i - n_report_iter])
    # populate the topic and word distributions
    theta = estimate_matrix(ndz, alpha, n_docs)
    phi = estimate_matrix(nzw, beta, n_topics)
    return theta, phi, np.asarray(eta), np.asarray(lL)

def gibbs_sampler_blhslda(int n_iter, int n_report_iter,
                         int n_topics, int n_docs,
                         int n_terms, int n_tokens,
                         double[:] alpha, double[:] beta,
                         double mu, double nu2, double b,
                         int[:] doc_lookup, int[:] term_lookup,
                         int[:,:] y, int[:] hier, seed):
    """
    Perform (collapsed) Gibbs sampling inference for Binary Logistic HSLDA.
    """

    cdef:
        int i, j, k, d, w, z, new_z, n_labels = y.shape[1], l, pa_l
        double p_sum, uval, eta_sum, eta_l, l_sum, exp_eta_l
        double sum_alpha = 0.
        double sum_beta = 0.
        int[:] topic_lookup = create_topic_lookup(n_tokens, n_topics, seed)
        # log likelihoods
        double[::1] lL = np.empty(n_iter, dtype=np.float64, order='C')
        # number of tokens in document d assigned to topic z, shape = (n_docs, n_topics)
        int[:, ::1] ndz = np.zeros((n_docs, n_topics), dtype=np.intc, order='C')
        # number of tokens assigned to topic z equal to term w, shape = (n_topics, n_terms)
        int[:, ::1] nzw = np.zeros((n_topics, n_terms), dtype=np.intc, order='C')
        # number of tokens assigned to topic k, shape = (n_topics,)
        int[::1] nz = np.zeros(n_topics, dtype=np.intc, order='C')
        # number of tokens in doc d, shape = (n_docs,)
        int[::1] nd = np.zeros(n_docs, dtype=np.intc, order='C')
        # (weighted) probabilities for the discrete distribution
        double[::1] p_cumsum = np.empty(n_topics, dtype=np.float64, order='C')
        # preallocate uniformly random numbers on the interval [0, 1)
        double[:] rands = create_rands(n_rands=n_rands, seed=seed)
        int u = 0
        # regression coefficients

        double [:, :, ::1] eta = np.zeros((n_iter + 1, n_labels, n_topics), dtype=np.float64, order='C')
        double [:, :, ::1] etand = np.empty((n_docs, n_labels, n_topics), dtype=np.float64, order='C')
        double [::1] eta_mean = np.empty(n_topics, dtype=np.float64, order='C')

        # omega: notice I'm initializing omega here
        double[:, ::1] omega = np.ones((n_docs,n_labels), dtype=np.float64, order='C')

        # kappa: a transformation of y
        double[:, ::1] kappa = b * (np.asarray(y) - 0.5)

        double [::1] kappa_sum = np.zeros(n_labels, dtype=np.float64, order='C')

    # initialize counts
    for j in range(n_tokens):
        preincrement(ndz[doc_lookup[j], topic_lookup[j]])
        preincrement(nzw[topic_lookup[j], term_lookup[j]])
        preincrement(nz[topic_lookup[j]])
        preincrement(nd[doc_lookup[j]])
    # initialize sum_alpha, lBeta_alpha
    for k in range(n_topics):
        sum_alpha += alpha[k]
    # initialize sum_beta, lBeta_beta
    for w in range(n_terms):
        sum_beta += beta[w]
    # define numpy variables
    Inu2 = np.identity(n_topics) / nu2
    munu2 = np.repeat(mu / nu2, n_topics)
    # define PolyaGamma sampler
    pg_rng = PyPolyaGamma(seed=seed or 42)
    # iterate
    start_time = datetime.now()
    print('{} start iterations'.format(start_time))
    for i in range(n_iter):
        # sample omega
        for d in range(n_docs):
            for l in range(n_labels):
                eta_sum = 0.
                for k in range(n_topics):
                # initialize etand for iteration i
                    etand[d, l, k] = eta[i, l, k] / nd[d]
                    eta_sum += etand[d, l, k] * ndz[d, k]
                omega[d,l] = pg_rng.pgdraw(b, eta_sum)

        # sample z
        for j in range(n_tokens):
            d = doc_lookup[j]
            w = term_lookup[j]
            z = topic_lookup[j]
            predecrement(ndz[d, z])
            predecrement(nzw[z, w])
            predecrement(nz[z])

            for l in range(n_labels):
                pa_l = hier[l]
                if pa_l == -1:
                    pa_l = l
                if y[d, pa_l]  == 1:
                    kappa_sum[l] = kappa[d,l]
                    for k in range(n_topics):
                        kappa_sum[l] -= omega[d,l] * etand[d, l, k] * ndz[d, k]

            eta_l = 0.
            for l in range(n_labels):
                pa_l = hier[l]
                if pa_l == -1:
                    pa_l = l
                if y[d, pa_l]  == 1:
                    l_sum = 0.
                    for k in range(n_topics):
                        l_sum += etand[d, l, k] * \
                                    (kappa_sum[l] - omega[d,l] / 2 * etand[d, l, k])
                    eta_l += l_sum

            p_sum = 0.
            for k in range(n_topics):
                p_sum += (nzw[k, w] + beta[w]) \
                        / (nz[k] + sum_beta) \
                        * (ndz[d, k] + alpha[k]) \
                        * exp(eta_l)
                p_cumsum[k] = p_sum
            preincrement(u)
            if u == n_rands:
                u = 0
            uval = rands[u] * p_sum
            new_z = topic_lookup[j] = searchsorted(p_cumsum, uval)
            preincrement(ndz[d, new_z])
            preincrement(nzw[new_z, w])
            preincrement(nz[new_z])

        #sample eta
        Z = (np.asarray(ndz) / np.asarray(nd)[:, np.newaxis]).T
        for l in range(n_labels):
            Omega = np.asarray(omega[::,l])[np.newaxis, :]
            Kappa = np.asarray(kappa[::,l])
            eta_mean = np.linalg.solve(Inu2 + np.dot(Z * Omega, Z.T),
                                       munu2 + np.dot(Z, Kappa))
            # TODO currently setting eta to mean, but need to sample
            for k in range(n_topics):
                eta[i + 1, l, k] = eta_mean[k]

        #compute log-likelihood
        lL[i] = loglikelihood_blhslda(nzw, ndz, nz, alpha, beta, sum_beta,
                                      mu, nu2, b, eta[i + 1], y, Z)

        # print progress
        print_progress(start_time, n_report_iter, i, lL[i], lL[i - n_report_iter])
    # populate the topic and word distributions
    theta = estimate_matrix(ndz, alpha, n_docs)
    phi = estimate_matrix(nzw, beta, n_topics)
    return theta, phi, np.asarray(eta), np.asarray(lL)
