# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import scipy.stats as stats
import scipy.special as special

prefix = "processed"


def sum_of_squares(a, axis=-1):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    if a.ndim == 0:
        a = np.atleast_1d(a)
    return np.sum(a * a, outaxis)


def pearsonr(x, y, eps=1e-5):
    r"""
    Calculate a Pearson correlation coefficient and the p-value for testing
    non-correlation.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed, and not necessarily zero-mean.
    Like other correlation coefficients, this one varies between -1 and +1
    with 0 implying no correlation. Correlations of -1 or +1 imply an exact
    linear relationship. Positive correlations imply that as x increases, so
    does y. Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------
    r : float
        Pearson's correlation coefficient
    p-value : float
        2-tailed p-value

    Notes
    -----

    The correlation coefficient is calculated as follows:

    .. math::

        r_{pb} = \frac{\sum (x - m_x) (y - m_y)}
                      {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

    where :math:`m_x` is the mean of the vector :math:`x` and :math:`m_y` is
    the mean of the vector :math:`y`.


    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([0, 0, 0, 1, 1, 1, 1])
    >>> b = np.arange(7)
    >>> stats.pearsonr(a, b)
    (0.8660254037844386, 0.011724811003954654)

    >>> stats.pearsonr([1,2,3,4,5], [5,6,7,8,7])
    (0.83205029433784372, 0.080509573298498519)
    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.shape[-1]
    mx = x.mean(axis=-1, keepdims=True)
    my = y.mean(axis=-1, keepdims=True)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym, axis=-1)
    # r_den = np.sqrt(sum_of_squares(xm) * sum_of_squares(ym))
    xm = np.sum(xm**2, axis=-1)
    ym = np.sum(ym**2, axis=-1)
    r_den = np.sqrt(xm * ym)
    idx = np.where(r_den==0)[0]
    r_den[idx] = eps
    r = r_num / r_den
    r[idx] = 0.0
    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = np.clip(r, -1.0, 1.0)
    df = n - 2
    idx = np.where(abs(r) == 1.0)[0]
    r[idx] += eps
    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    prob = special.betainc(
        0.5*df, 0.5, np.fmin(np.asarray(df / (df + t_squared)), 1.0)
    )
    prob[idx] = 0.0
    r[idx] -= eps

    return r, prob


def get_adj(input_x, type='fo'):
    # input_x: (batch, window, x_dim)
    if type == 'fo':
        input_x = input_x.transpose(0, 2, 1)
    adj = np.empty((input_x.shape[0], input_x.shape[1], input_x.shape[1]), dtype=np.float32)
    for j in range(input_x.shape[1]):
        for k in range(j, input_x.shape[1]):
            adj[:, k, j] = adj[:, j, k] = pearsonr(input_x[:, j], input_x[:, k])[0]
    # print('-----adj---------')
    return np.nan_to_num(adj)


def save_z(z, filename='z'):
    """
    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + '_' + str(i) + '.txt', 'w') as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write('%f ' % (z[j][i][k]))
                file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset '+str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        # train_data = data_clean(train_data)
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)

def data_clean(df):
    """
    Filter extremely abnormal data in train data and fill it with nearby values
    """
    df = pd.DataFrame(df)
    for c in df.columns:
        x = df[c]
        d1 = np.quantile(x,.25)
        d3 = np.quantile(x,.75)
        gap = d3 - d1
        min_ = d1 - 3 * gap
        max_ = d3 + 3 * gap
        index = ((x < min_ ) | (x > max_))
        x[index] = np.nan
        x.interpolate(method='nearest', inplace=True)
    return df.values
        
    

def preprocess(df):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df


def minibatch_slices_iterator(length, batch_size,
                              ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.

    Args:
        length (int): Total length of data in an epoch.
        batch_size (int): Size of each mini-batch.
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.

    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.

    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        # check the parameters
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as '
                             '`window_size`')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('The shape of `excludes` is expected to be '
                                 '{}, but got {}'.
                                 format(expected_shape, excludes.shape))

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.

        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.

        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.

        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
                length=len(self._indices),
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)
