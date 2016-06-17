import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import entropy as KL_divergence
from sklearn.cross_validation import StratifiedKFold
from ..topic_models import LDA, SLDA, BLSLDA, GRTM


def gen_topics(rows):
    topics = []
    topic_base = np.concatenate((np.ones((1, rows)) * (1/rows),
                                 np.zeros((rows-1, rows))), axis=0).ravel()
    for i in range(rows):
        topics.append(np.roll(topic_base, i * rows))
    topic_base = np.concatenate((np.ones((rows, 1)) * (1/rows),
                                 np.zeros((rows, rows-1))), axis=1).ravel()
    for i in range(rows):
        topics.append(np.roll(topic_base, i))
    return np.array(topics)


def gen_thetas(alpha, D):
    return np.random.dirichlet(alpha, size=D)


def gen_doc(seed, K, N, thetas, V, topics, D):
    topic_assignments = np.array([np.random.choice(range(K), size=N, p=theta)
                                  for theta in thetas])
    word_assignments = \
        np.array([[np.random.choice(range(V), size=1,
                                    p=topics[topic_assignments[d, n]])[0]
                   for n in range(N)] for d in range(D)])
    return np.array([np.histogram(word_assignments[d], bins=V,
                                  range=(0, V - 1))[0] for d in range(D)])


def language(document_size):
    # Generate topics
    # We assume a vocabulary of 'rows'^2 terms, and create 'rows'*2 "topics",
    # where each topic assigns exactly 'rows' consecutive terms equal
    # probability.
    rows = 3
    V = rows * rows
    K = rows * 2
    N = K * K
    D = document_size
    seed = 42
    topics = gen_topics(rows)

    # Generate documents from topics
    # We generate D documents from these V topics by sampling D topic
    # distributions, one for each document, from a Dirichlet distribution with
    # parameter α=(1,…,1)
    alpha = np.ones(K)
    np.random.seed(seed)
    thetas = gen_thetas(alpha, D)
    doc_term_matrix = gen_doc(seed, K, N, thetas, V, topics, D)
    return {'V': V, 'K': K, 'D': D, 'seed': seed, 'alpha': alpha,
            'topics': topics, 'thetas': thetas,
            'doc_term_matrix': doc_term_matrix, 'n_report_iters': 100}


def assert_probablity_distribution(results):
    assert (results >= 0).all()
    assert results.sum(axis=1).all()


def check_KL_divergence(topics, results, thresh):
    for res in results:
        minimized_KL = 1
        for topic in topics:
            KL = KL_divergence(topic, res)
            if KL < minimized_KL:
                minimized_KL = KL
        print(minimized_KL)
        assert minimized_KL < thresh


def test_lda():
    l = language(10000)
    n_iter = 2000
    KL_thresh = 0.001

    np.random.seed(l['seed'])
    _beta = np.repeat(0.01, l['V'])
    lda = LDA(l['K'], l['alpha'], _beta, n_iter, seed=l['seed'],
              n_report_iter=l['n_report_iters'])
    lda.fit(l['doc_term_matrix'])

    assert_probablity_distribution(lda.phi)
    check_KL_divergence(l['topics'], lda.phi, KL_thresh)


def test_slda():
    l = language(10000)
    n_iter = 2000
    KL_thresh = 0.001

    nu2 = l['K']
    sigma2 = 1
    np.random.seed(l['seed'])
    eta = np.random.normal(scale=nu2, size=l['K'])
    y = [np.dot(eta, l['thetas'][i]) for i in range(l['D'])] + \
        np.random.normal(scale=sigma2, size=l['D'])
    _beta = np.repeat(0.01, l['V'])
    _mu = 0
    slda = SLDA(l['K'], l['alpha'], _beta, _mu, nu2, sigma2, n_iter,
                seed=l['seed'], n_report_iter=l['n_report_iters'])
    slda.fit(l['doc_term_matrix'], y)

    assert_probablity_distribution(slda.phi)
    check_KL_divergence(l['topics'], slda.phi, KL_thresh)


def test_blslda():
    l = language(10000)
    n_iter = 1500
    KL_thresh = 0.03

    mu = 0.
    nu2 = 1.
    np.random.seed(l['seed'])
    eta = np.random.normal(loc=mu, scale=nu2, size=l['K'])
    zeta = np.array([np.dot(eta, l['thetas'][i]) for i in range(l['D'])])
    y = (zeta >= 0).astype(int)
    _beta = np.repeat(0.01, l['V'])
    _b = 7.25
    blslda = BLSLDA(l['K'], l['alpha'], _beta, mu, nu2, _b, n_iter,
                    seed=l['seed'],
                    n_report_iter=l['n_report_iters'])
    blslda.fit(l['doc_term_matrix'], y)

    assert_probablity_distribution(blslda.phi)
    check_KL_divergence(l['topics'], blslda.phi, KL_thresh)


def test_grtm():
    l = language(1000)
    n_iter = 1000
    KL_thresh = 0.3

    mu = 0.
    nu2 = 1.
    np.random.seed(l['seed'])
    H = np.random.normal(loc=mu, scale=nu2, size=(l['K'], l['K']))
    zeta = pd.DataFrame([(i, j, np.dot(np.dot(l['thetas'][i], H),
                                       l['thetas'][j]))
                         for i, j in product(range(l['D']), repeat=2)],
                        columns=('tail', 'head', 'zeta'))
    zeta['y'] = (zeta.zeta >= 0).astype(int)
    y = zeta[['tail', 'head', 'y']].values
    skf = StratifiedKFold(y[:, 2], n_folds=100)
    _, train_idx = next(iter(skf))
    _K = l['K']
    _alpha = l['alpha'][:_K]
    _beta = np.repeat(0.01, l['V'])
    _b = 1.
    grtm = GRTM(_K, _alpha, _beta, mu, nu2, _b, n_iter, seed=l['seed'],
                n_report_iter=l['n_report_iters'])
    grtm.fit(l['doc_term_matrix'], y[train_idx])

    assert_probablity_distribution(grtm.phi)
    check_KL_divergence(l['topics'], grtm.phi, KL_thresh)
