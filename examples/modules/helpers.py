from math import ceil, sqrt
from itertools import product
from numpy.linalg import norm
from scipy.stats import entropy


def plot_images(mpl_plt, images, image_shape, layout=None, titles=None,
                imshow_kwargs={'cmap': 'gray', 'interpolation': 'nearest'},
                **kwargs):
    """
    Plots sequence of images.

    Parameters
    ----------
    mpl_plt : matplotlib.plot
        A matplotlib plot handler.

    images : array-like
        An array of images, all of the same shape.

    image_shape : 2-tuple
        The 2-dimensional shape of each image.

    layout : 2-tuple (optional)
        Tuple of form (n_rows, n_cols) describing layout of images.
        If None, automatically calculates the most-square layout.

    titles : array-like (optional)
        An array of titles.  Should be same length as images.

    cmap : matplotlib.colormap (optional)
        A colormap for the images. Default is 'gray'.

    kwargs : key-value pairs (optional)
        Keyword arguments to be passed to mpl_plt.

    Returns
    -------
    fig : matplotlib.plot.Figure
        Matplotlib figure.

    Notes
    -----
    This function should show up in zeku at some point, but zeku needs to be
    ported to python 3 first.
    """
    n = len(images)
    if layout:
        n_rows, n_cols = layout
    else:
        n_rows = int(ceil(sqrt(n)))
        if n <= n_rows * (n_rows - 1):
            n_cols = n_rows - 1
        else:
            n_cols = n_rows
    titles = titles or range(n)
    if len(titles) != n:
        raise ValueError('titles should be the same length as images')
    fig, axes = mpl_plt.subplots(n_rows, n_cols, **kwargs)
    if n_rows == 1:
        pairs = range(n_cols)
    elif n_cols == 1:
        pairs = range(n_rows)
    else:
        pairs = product(range(n_rows), range(n_cols))
    pairs = list(pairs)[:n]
    for title, image, pair in zip(titles, images, pairs):
        axes[pair].imshow(image.reshape(*image_shape), **imshow_kwargs)
        axes[pair].set_title('%s' % title)


def JSD(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, M) + entropy(Q, M))
