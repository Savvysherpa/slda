"""
Topic models using Gibbs sampling.

Draws from Allen Riddell's LDA library https://github.com/ariddell/lda
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse
from ._topic_models import (gibbs_sampler_lda, gibbs_sampler_slda,
                            gibbs_sampler_blslda, gibbs_sampler_grtm,
                            gibbs_sampler_rtm, gibbs_sampler_blhslda,
                            iterated_pseudo_counts)


class TopicModelBase(BaseEstimator, TransformerMixin):
    """
    Base class for topic models.
    """
    n_topics = None
    alpha = None
    beta = None
    theta = None
    phi = None
    loglikelihoods = None

    def __init__(self):
        raise NotImplementedError

    def _create_lookups(self, X):
        """
        Create document and term lookups for all tokens.
        """
        docs, terms = np.nonzero(X)
        if issparse(X):
            x = np.array(X[docs, terms])[0]
        else:
            x = X[docs, terms]
        doc_lookup = np.ascontiguousarray(np.repeat(docs, x), dtype=np.intc)
        term_lookup = np.ascontiguousarray(np.repeat(terms, x), dtype=np.intc)
        return doc_lookup, term_lookup

    def fit(self):
        """
        Estimate the topic distributions per document (theta) and term
        distributions per topic (phi).

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix
        """

        raise NotImplementedError

    def fit_transform(self, X):
        """
        Estimate the topic distributions per document (theta) and term
        distributions per topic (phi), then return theta.

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix

        Returns
        _______
        theta : numpy array, shape = (n_docs, n_topics)
            The topic distribution of each document
        """

        self.fit(X)
        return self.theta

    def transform(self, X, max_iter=20, tol=1e-16):
        """
        Estimate the topic distributions of new documents given the fit model.
        """

        if self.phi is None:
            raise RuntimeError('self.phi is None, which means the model has ' +
                               'not been fit yet. Please fit the model first.')
        n_docs, n_topics = X.shape
        doc_lookup, term_lookup = self._create_lookups(X)
        return iterated_pseudo_counts(doc_lookup, term_lookup, n_docs,
                                      self.alpha, self.beta, self.phi,
                                      max_iter, tol)


class LDA(TopicModelBase):
    """
    Latent Dirichlet allocation, using collapsed Gibbs sampling implemented in
    Cython.

    Parameters
    ----------
    n_topics : int
        Number of topics

    alpha : array-like, shape = (n_topics,)
        Dirichlet distribution parameter for each document's topic
        distribution.

    beta : array-like, shape = (n_terms,)
        Dirichlet distribution parameter for each topic's term distribution.

    n_iter : int, default=500
        Number of iterations of Gibbs sampler

    n_report_iter : int, default=10
        Number of iterations of Gibbs sampler between progress reports.

    random_state : int, optional
        Seed for random number generator
    """

    def __init__(self, n_topics, alpha, beta, n_iter=500, n_report_iter=10,
                 seed=None):
        self.n_topics = n_topics
        self.alpha = np.ascontiguousarray(alpha, dtype=np.float64)
        self.beta = np.ascontiguousarray(beta, dtype=np.float64)
        self.n_iter = n_iter
        self.n_report_iter = n_report_iter
        self.seed = seed

    def fit(self, X):
        """
        Estimate the topic distributions per document (theta) and term
        distributions per topic (phi).

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix
        """

        self.doc_term_matrix = X
        self.n_docs, self.n_terms = X.shape
        self.n_tokens = X.sum()
        doc_lookup, term_lookup = self._create_lookups(X)
        # iterate
        self.theta, self.phi, self.loglikelihoods = gibbs_sampler_lda(
            self.n_iter, self.n_report_iter,
            self.n_topics, self.n_docs, self.n_terms, self.n_tokens,
            self.alpha, self.beta, doc_lookup, term_lookup, self.seed)


class SLDA(TopicModelBase):
    """
    Supervised (regression) latent Dirichlet allocation, using collapsed Gibbs
    sampling implemented in Cython.

    Parameters
    ----------
    n_topics : int
        Number of topics

    alpha : array-like, shape = (n_topics,)
        Dirichlet distribution parameter for each document's topic
        distribution.

    beta : array-like, shape = (n_terms,)
        Dirichlet distribution parameter for each topic's term distribution.

    mu : float
        Mean of regression coefficients (eta).

    nu2 : float
        Variance of regression coefficients (eta).

    sigma2 : float
        Variance of response (y).

    n_iter : int, default=500
        Number of iterations of Gibbs sampler

    n_report_iter : int, default=10
        Number of iterations of Gibbs sampler between progress reports.

    random_state : int, optional
        Seed for random number generator
    """

    def __init__(self, n_topics, alpha, beta, mu, nu2, sigma2, n_iter=500,
                 n_report_iter=10, seed=None):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu2 = nu2
        self.sigma2 = sigma2
        self.n_iter = n_iter
        self.n_report_iter = n_report_iter
        self.seed = seed

    def fit(self, X, y):
        """
        Estimate the topic distributions per document (theta), term
        distributions per topic (phi), and regression coefficients (eta).

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix.

        y : array-like, shape = (n_docs,)
            Response values for each document.
        """

        self.doc_term_matrix = X
        self.n_docs, self.n_terms = X.shape
        self.n_tokens = X.sum()
        doc_lookup, term_lookup = self._create_lookups(X)
        # iterate
        self.theta, self.phi, self.eta, self.loglikelihoods = gibbs_sampler_slda(
            self.n_iter, self.n_report_iter,
            self.n_topics, self.n_docs, self.n_terms, self.n_tokens,
            self.alpha, self.beta, self.mu, self.nu2, self.sigma2,
            doc_lookup, term_lookup,
            np.ascontiguousarray(y, dtype=np.float64), self.seed)


class BLSLDA(TopicModelBase):
    """
    Binary logistic supervised latent Dirichlet allocation, using collapsed
    Gibbs sampling implemented in Cython.

    Parameters
    ----------
    n_topics : int
        Number of topics

    alpha : array-like, shape = (n_topics,)
        Dirichlet distribution parameter for each document's topic
        distribution.

    beta : array-like, shape = (n_terms,)
        Dirichlet distribution parameter for each topic's term distribution.

    mu : float
        Mean of regression coefficients (eta).

    nu2 : float
        Variance of regression coefficients (eta).

    b : float
        Regularization parameter.

    n_iter : int, default=500
        Number of iterations of Gibbs sampler

    n_report_iter : int, default=10
        Number of iterations of Gibbs sampler between progress reports.

    random_state : int, optional
        Seed for random number generator
    """

    def __init__(self, n_topics, alpha, beta, mu, nu2, b, n_iter=500,
                 n_report_iter=10, seed=None):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu2 = nu2
        self.b = b
        self.n_iter = n_iter
        self.n_report_iter = n_report_iter
        self.seed = seed

    def fit(self, X, y):
        """
        Estimate the topic distributions per document (theta), term
        distributions per topic (phi), and regression coefficients (eta).

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix.

        y : array-like, shape = (n_docs,)
            Response values for each document.
        """

        self.doc_term_matrix = X
        self.n_docs, self.n_terms = X.shape
        self.n_tokens = X.sum()
        doc_lookup, term_lookup = self._create_lookups(X)
        # iterate
        self.theta, self.phi, self.eta, self.loglikelihoods = gibbs_sampler_blslda(
            self.n_iter, self.n_report_iter,
            self.n_topics, self.n_docs, self.n_terms, self.n_tokens,
            self.alpha, self.beta, self.mu, self.nu2, self.b,
            doc_lookup, term_lookup,
            np.ascontiguousarray(y, dtype=np.float64), self.seed)


class GRTM(TopicModelBase):
    """
    Generalized relational topic models, using collapsed Gibbs sampling
    implemented in Cython.

    Parameters
    ----------
    n_topics : int
        Number of topics

    alpha : array-like, shape = (n_topics,)
        Dirichlet distribution parameter for each document's topic
        distribution.

    beta : array-like, shape = (n_terms,)
        Dirichlet distribution parameter for each topic's term distribution.

    mu : float
        Mean of regression coefficients (eta).

    nu2 : float
        Variance of regression coefficients (eta).

    b : float
        Regularization parameter.

    n_iter : int, default=500
        Number of iterations of Gibbs sampler

    n_report_iter : int, default=10
        Number of iterations of Gibbs sampler between progress reports.

    random_state : int, optional
        Seed for random number generator
    """

    def __init__(self, n_topics, alpha, beta, mu, nu2, b, n_iter=500,
                 n_report_iter=10, seed=None):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu2 = nu2
        self.b = b
        self.n_iter = n_iter
        self.n_report_iter = n_report_iter
        self.seed = seed

    def _create_edges(self, y, order='tail'):
        y.sort(order=order)
        _docs, _counts = np.unique(y[order], return_counts=True)
        counts = np.zeros(self.n_docs)
        counts[_docs] = _counts
        docs = np.ascontiguousarray(
            np.concatenate(([0], np.cumsum(counts))), dtype=np.intc)
        edges = np.ascontiguousarray(y['index'].flatten(), dtype=np.intc)
        return docs, edges

    def fit(self, X, y):
        """
        Estimate the topic distributions per document (theta), term
        distributions per topic (phi), and regression coefficients (eta).

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix.

        y : array-like, shape = (n_edges, 3)
            Each entry of y is an ordered triple (d_1, d_2, y_(d_1, d_2)),
            where d_1 and d_2 are documents and y_(d_1, d_2) is an indicator of
            a directed edge from d_1 to d_2.
        """

        self.doc_term_matrix = X
        self.n_docs, self.n_terms = X.shape
        self.n_tokens = X.sum()
        self.n_edges = y.shape[0]
        doc_lookup, term_lookup = self._create_lookups(X)
        # edge info
        y = np.ascontiguousarray(np.column_stack((range(self.n_edges), y)))
        # we use a view here so that we can sort in-place using named columns
        y_rec = y.view(dtype=list(zip(('index', 'tail', 'head', 'data'),
                                      4 * [y.dtype])))
        edge_tail = np.ascontiguousarray(y_rec['tail'].flatten(),
                                         dtype=np.intc)
        edge_head = np.ascontiguousarray(y_rec['head'].flatten(),
                                         dtype=np.intc)
        edge_data = np.ascontiguousarray(y_rec['data'].flatten(),
                                         dtype=np.float64)
        out_docs, out_edges = self._create_edges(y_rec, order='tail')
        in_docs, in_edges = self._create_edges(y_rec, order='head')
        # iterate
        self.theta, self.phi, self.H, self.loglikelihoods = gibbs_sampler_grtm(
            self.n_iter, self.n_report_iter, self.n_topics, self.n_docs,
            self.n_terms, self.n_tokens, self.n_edges, self.alpha, self.beta,
            self.mu, self.nu2, self.b, doc_lookup, term_lookup, out_docs,
            out_edges, in_docs, in_edges, edge_tail, edge_head, edge_data,
            self.seed)


class RTM(TopicModelBase):
    """
    Relational topic models, using collapsed Gibbs sampling implemented in
    Cython.

    Parameters
    ----------
    n_topics : int
        Number of topics

    alpha : array-like, shape = (n_topics,)
        Dirichlet distribution parameter for each document's topic
        distribution.

    beta : array-like, shape = (n_terms,)
        Dirichlet distribution parameter for each topic's term distribution.

    mu :

    sigma2:

    nu :

    n_iter : int, default=500
        Number of iterations of Gibbs sampler

    n_report_iter : int, default=10
        Number of iterations of Gibbs sampler between progress reports.

    random_state : int, optional
        Seed for random number generator
    """

    def __init__(self, n_topics, alpha, beta, mu, sigma2, nu, n_iter=500,
                 n_report_iter=10, seed=None):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma2 = sigma2
        self.nu = nu
        self.n_iter = n_iter
        self.n_report_iter = n_report_iter
        self.seed = seed

    def fit(self, X, y=None):
        """
        Estimate the topic distributions per document (theta), the term
        distributions per topic (phi), and the regression coefficients (eta).

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix

        y : array-like, shape = (n_docs, n_docs)
            The adjacency matrix of the document network
        """

        self.doc_term_matrix = X
        self.n_docs, self.n_terms = X.shape
        self.n_tokens = X.sum()
        doc_lookup, term_lookup = self._create_lookups(X)
        self.adjacency_matrix = y
        # iterate
        self.theta, self.phi, self.eta, self.loglikelihoods = gibbs_sampler_rtm(
            self.n_iter, self.n_report_iter,
            self.n_topics, self.n_docs, self.n_terms, self.n_tokens,
            self.alpha, self.beta, self.mu, self.sigma2, self.nu,
            doc_lookup, term_lookup, self.adjacency_matrix, self.seed)


class BLHSLDA(TopicModelBase):
    """
    Binary Logistic Heirarchical supervised latent Dirichlet allocation, using
    collapsed Gibbs sampling implemented in Cython.

    Parameters
    ----------
    n_topics : int
        Number of topics

    alpha : array-like, shape = (n_topics,)
        Dirichlet distribution parameter for each document's topic
        distribution.

    beta : array-like, shape = (n_terms,)
        Dirichlet distribution parameter for each topic's term distribution.

    mu : float
        Mean of regression coefficients (eta).

    nu2 : float
        Variance of regression coefficients (eta).

    b : float
        Regularization parameter.

    n_iter : int, default=500
        Number of iterations of Gibbs sampler

    n_report_iter : int, default=10
        Number of iterations of Gibbs sampler between progress reports.

    random_state : int, optional
        Seed for random number generator
    """

    def __init__(self, n_topics, alpha, beta, mu, nu2, b, n_iter=500,
                 n_report_iter=10, seed=None):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu2 = nu2
        self.b = b
        self.n_iter = n_iter
        self.n_report_iter = n_report_iter
        self.seed = seed

    def fit(self, X, y, hier):
        """
        Estimate the topic distributions per document (theta), term
        distributions per topic (phi), and regression coefficients (eta).

        Parameters
        ----------
        X : array-like, shape = (n_docs, n_terms)
            The document-term matrix.

        y : array-like, shape = (n_docs, n_labels)
            Response values for each document for each labels.

        hier : 1D array-like, size = n_labels
            The index of the list corresponds to the current label
            and the value of the indexed position is the parent of the label.
                Set -1 as the root.
        """

        self.doc_term_matrix = X
        self.n_docs, self.n_terms = X.shape
        self.n_tokens = X.sum()
        doc_lookup, term_lookup = self._create_lookups(X)

        # iterate
        self.theta, self.phi, self.eta, self.loglikelihoods = gibbs_sampler_blhslda(
            self.n_iter, self.n_report_iter,
            self.n_topics, self.n_docs, self.n_terms, self.n_tokens,
            self.alpha, self.beta, self.mu, self.nu2, self.b, doc_lookup,
            term_lookup, np.ascontiguousarray(y, dtype=np.intc),
            np.ascontiguousarray(hier, dtype=np.intc), self.seed)
