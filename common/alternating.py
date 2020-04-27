import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, lp, options
import seaborn as sns
from scipy.linalg import sqrtm
import math
from sklearn import linear_model

#import read_data
import common.missing as missing
import common.kmeans_greedy as kmeans_greedy


options['show_progress'] = False


def get_index_matrix(cluster_num, ind, normalize=True):
    n = np.size(ind)
    ind_mat = np.zeros((n, cluster_num))

    for i in range(n):
        ind_mat[i, ind[i]] = 1.

    if normalize:
        for j in range(cluster_num):
            x = ind_mat[:, j]
            norm = np.linalg.norm(x)
            if norm > 0.000000001:
                ind_mat[:, j] = x / norm
            else:
                # print("WARNING!!!")
                pass

    return ind_mat


def index_dist(cluster_num, int1, int2):
    n = np.size(int1)
    assert (np.size(int2 == n))

    b = np.matmul(get_index_matrix(cluster_num, int1, normalize=False).T,
                  get_index_matrix(cluster_num, int2, normalize=False))
    b = np.reshape(b, (cluster_num * cluster_num,))

    I = spmatrix(1.0, range(cluster_num ** 2), range(cluster_num ** 2))
    row_ones = spmatrix(
        1.0,
        sum([[j] * cluster_num for j in range(cluster_num)], []),
        sum([[cluster_num * j + k for k in range(cluster_num)] for j in range(cluster_num)], [])
    )
    col_ones = spmatrix(
        1.0,
        sum([[j] * cluster_num for j in range(cluster_num)], []),
        sum([[cluster_num * k + j for k in range(cluster_num)] for j in range(cluster_num)], []),
    )

    res = lp(-matrix(b),
             -I, matrix(0.0, size=(cluster_num ** 2, 1)),
             sparse([row_ones, col_ones])[:-1, :], matrix(1.0, size=(2 * cluster_num - 1, 1)))

    return n + round(res['primal objective'])


def gauss_var1_process(theta, sigma, T):
    # variance of innovations is sigma * I
    #
    n, m = np.shape(theta)
    assert(n == m)
    assert(np.linalg.norm(theta, 2) < 0.9999)

    LIMIT = 2000
    x0 = np.zeros(n)
    for _ in range(LIMIT):
        eps = np.random.randn(n) * sigma
        x0 = np.dot(theta, x0) + eps

    x = np.empty((n, T))
    x[:, 0] = x0
    for i in range(1, T):
        eps = np.random.randn(n) * sigma
        x[:, i] = np.dot(theta, x[:, i-1]) + eps

    return x


def lasso_from_covariance(D, c, alpha):
    # minimizes 1/2 x^{T}Dx - c^{T} x + \alpha \| v \|_1
    #
    n = np.size(c)
    assert(np.shape(D) == (n, n))

    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = D

    d = np.empty(2 * n)
    d[:n] = -c
    d[n:] = alpha

    G = np.concatenate((np.concatenate((-np.identity(n), -np.identity(n)), axis=1),
                        np.concatenate((np.identity(n), -np.identity(n)), axis=1)), axis=0)
    h = np.zeros(2 * n)

    res = qp(matrix(Q), matrix(d), G=matrix(G), h=matrix(h))
    x = np.reshape(np.array(res['x'].T), (2 * n,))[:n]

    return x, res['primal objective']


def random_basis(n, k):
    assert (n >= k)
    x = np.random.randn(n, k)
    return np.matmul(x, sqrtm(np.linalg.inv(np.matmul(x.T, x))))  # orthogonal normalization of columns


def u_random(n, k, cluster_num, index=None):
    assert (cluster_num >= k)

    x = random_basis(cluster_num, k)
    if index is None:
        ind = np.random.randint(cluster_num, size=n)
    else:
        ind = index
    ind_mat = get_index_matrix(cluster_num, ind)

    return np.matmul(ind_mat, x), ind


def v_step(D0, D1, alpha_v, z):
    # make sure u^{\T}u = I!
    #
    n, k = np.shape(z)
    assert (np.shape(D0) == (n, n) and np.shape(D1) == (n, n))
    assert (k <= n)

    v = np.empty((k, n))
    loss = 0.

    for j in range(k):
        x, _loss_j = lasso_from_covariance(D0, np.dot(D1, z[:, j]), alpha_v)
        v[j, :] = x
        loss += _loss_j

    return v, loss


def z_step(cluster_num, D1, v, ind_old=None):
    k, n = v.shape
    assert (np.shape(D1) == (n, n))
    assert (k <= n)

    mat = np.matmul(v, D1)

    #check this stupid function!!! <- apparently works...
    res = kmeans_greedy.kmeans_greedy(lambda ind: np.matrix.trace(np.matmul(mat, get_index_matrix(cluster_num, ind))),
                                      cluster_num, n, init_index=ind_old)
    ind_mat = get_index_matrix(cluster_num, res.index)

    return ind_mat, res.index,  .0


class _ResultInstance:
    def __init__(self, theta, u, v, index, loss):
        self.theta = theta
        self.u = u
        self.v = v
        self.loss = loss
        self.index = index


# to implement: index choice; initial index + initial u
# matrix competition (index competition?)
#


def alternating(cluster_num, D0, D1, alpha_v, epochs=10, index_init=None):
    n, _ = np.shape(D0)
    assert (np.shape(D0) == (n, n))
    assert (np.shape(D1) == (n, n))

    if index_init is None:
        mat = np.matmul(random_basis(n, cluster_num).T, D1)
        res = kmeans_greedy.kmeans_greedy(lambda ind: np.linalg.norm(
                                                np.matmul(mat, get_index_matrix(cluster_num, ind)), ord='nuc'
                                            ), cluster_num, n)
        ind_start = res.index
    else:
        ind_start = index_init

    for e in range(epochs):
        if e == 0:
            ind_est = ind_start
            z_est = get_index_matrix(cluster_num, ind_start)
        else:
            z_est, ind_est, _ = z_step(cluster_num, D1, v_est, ind_old=ind_est)
        v_est, loss = v_step(D0, D1, alpha_v, z_est)

    theta_est = np.matmul(z_est, v_est)
    #print("loss : {}".format(loss))

    return _ResultInstance(theta_est, z_est, v_est, ind_est, loss)


def v_step_from_data(x_train, y_train, alpha_v, z):
    # make sure u^{\T}u = I!
    #
    n, tmax = np.shape(x_train)
    _, k = np.shape(z)

    assert (np.shape(y_train) == (n, tmax))
    assert (np.shape(z)[0] == n)

    clf = linear_model.Lasso(alpha=alpha_v)
    v = np.empty((k, n))
    loss = 0.

    for j in range(k):
        _y = np.dot(y_train.T, z[:, j])
        clf.fit(x_train.T, _y)
        v[j, :] = clf.coef_
        loss += np.linalg.norm(_y - np.dot(x_train.T, clf.coef_)) ** 2 / (2 * tmax) + alpha_v * np.linalg.norm(clf.coef_, ord=1)

    return v, loss


def direct(cluster_num, D0, D1, alpha_v, index_init=None):
    n, _ = np.shape(D0)
    assert (np.shape(D0) == (n, n))
    assert (np.shape(D1) == (n, n))

    func = lambda ind: -v_step(D0, D1, alpha_v, get_index_matrix(cluster_num, ind))[1] #loss
    res = kmeans_greedy.kmeans_greedy(func, cluster_num, n, init_index=index_init)

    z_est = get_index_matrix(cluster_num, res.index)
    v_est, loss = v_step(D0, D1, alpha_v, z_est)

    theta_est = np.matmul(z_est, v_est)
    #print("loss : {}".format(loss))

    return _ResultInstance(theta_est, z_est, v_est, res.index, loss)


def direct_from_data(cluster_num, x_train, y_train, alpha_v, index_init=None):
    n, tmax = np.shape(x_train)
    assert (np.shape(x_train) == np.shape(y_train))

    func = lambda ind: -v_step_from_data(x_train, y_train, alpha_v, get_index_matrix(cluster_num, ind))[1]  # loss
    res = kmeans_greedy.kmeans_greedy(func, cluster_num, n, init_index=index_init)

    z_est = get_index_matrix(cluster_num, res.index)
    v_est, loss = v_step(x_train, y_train, alpha_v, z_est)

    theta_est = np.matmul(z_est, v_est)
    # print("loss : {}".format(loss))

    return _ResultInstance(theta_est, z_est, v_est, res.index, loss)


def alternating_from_data(cluster_num, x_train, y_train, alpha_v, epochs=100, index_init=None):
    n, tmax = np.shape(x_train)
    assert (np.shape(x_train) == np.shape(y_train))

    D0 = np.matmul(x_train, x_train.T) / tmax
    D1 = np.matmul(x_train, y_train.T) / tmax

    def _func1(v, ind):
        result = 0
        _z = get_index_matrix(cluster_num, ind)
        for j in range(cluster_num):
            result += np.dot(v[j, :], np.dot(D1, _z[:, j]))
        return result

    def _func0(ind):
        result = 0
        _z = get_index_matrix(cluster_num, ind)
        for j in range(cluster_num):
            result += np.dot(_z[:, j], np.dot(D0, _z[:, j]))
        return result

    if index_init is None:
        res = kmeans_greedy.kmeans_greedy(_func0, cluster_num, n, iter_limit=100)
        ind_start = res.index
    else:
        ind_start = index_init

    for e in range(epochs):
        if e == 0:
            ind_est = ind_start
            nochange = False
        else:
            #print("K={}: epochs {}/{}".format(cluster_num, e, epochs))
            res = kmeans_greedy.kmeans_greedy(lambda ind: _func1(v_est, ind), cluster_num, n, iter_limit=50)
            ind_est = res.index
            nochange = res.nochange
        z_est = get_index_matrix(cluster_num, ind_est)
        v_est, loss = v_step_from_data(x_train, y_train, alpha_v, z_est)
        if nochange:
            break

    theta_est = np.matmul(z_est, v_est)

    return _ResultInstance(theta_est, z_est, v_est, ind_est, loss)


def matrix_competition(type, repeat_num,  *args, **kwargs):
    assert (type in ['ALTER', 'ALTER_FROM_DATA', 'DIRECT', 'DIRECT_FROM_DATA'])
    ress = []
    for _ in range(repeat_num):
        if type == 'ALTER':
            ress.append(alternating(*args, **kwargs))
        elif type == 'ALTER_FROM_DATA':
            ress.append(alternating_from_data(*args, **kwargs))
        elif type == 'DIRECT':
            ress.append(direct(*args, **kwargs))
        else:
            ress.append(direct_from_data(*args, **kwargs))

    _, idx = min([(res.loss, i) for (i, res) in enumerate(ress)])
    return ress[idx]


def simu(n, c_num, s, T, pmin=1.0):
    # define true index
    c_size = int(n // c_num)
    r = n - c_num * c_size
    ind_star = np.array([int(i // (c_size + 1)) if i < r * (c_size + 1)
                         else int((i - r * (c_size + 1)) // c_size) + r for i in range(n)])
    z_star = get_index_matrix(c_num, ind_star)

    #define true theta
    assert (s <= 5)
    v_star = np.zeros((c_num, n))
    active_vals = [0.6, -0.4, 0.1, -0.8, 0.2]
    for j in range(c_num):
        v_star[j, j: j + s] = np.array(active_vals[:s]) * ((-1) ** j)

    theta_star = np.matmul(z_star, v_star)
    coef = 0.5 / np.linalg.norm(theta_star, 2)

    v_star = v_star * coef
    theta_star = theta_star * coef

    #generate the time series
    x = gauss_var1_process(theta_star, 1., T)

    #include missing observations
    mask = np.random.binomial(1, pmin, size=(n, T)).astype(np.float64)
    x_missing = x * mask

    x_train = x_missing[:, :-1]
    y_train = x_missing[:, 1:]

    deltas = np.mean(mask, axis=1)

    D0 = missing.missing_var(x_train, deltas)
    D1 = missing.missing_covar(x_train, y_train, deltas, deltas)

    alphas = np.logspace(-3, 0, num=10, base=2) * 3 * math.sqrt(math.log(n) / (T * (pmin ** 2)))

    node_infls = []
    cl_diffs = []
    theta_diffs = []
    ind_prev = ind_star
    for i, alpha_v in enumerate(alphas):
        res = matrix_competition('ALTER', 1, c_num, D0, D1, alpha_v, index_init=ind_star, epochs=20)
        theta_est, u_est, v_est, loss = res.theta, res.u, res.v, res.loss

        cl_diffs.append(index_dist(c_num, res.index, ind_star))
        theta_diffs.append(np.linalg.norm(theta_est - theta_star, ord='fro'))
        node_infls.append(np.linalg.norm(theta_est, ord=2, axis=0))

        ind_prev = res.index

        print("K={}: {}/10".format(c_num, i))

    if False:
        sns.set()
        ax = sns.heatmap(theta_est, center=0)
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=8)
        #ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

        sns.set()
        ax = sns.heatmap(theta_star, center=0)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=8)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

    return alphas, np.array(theta_diffs), np.array(cl_diffs), np.array(node_infls)

#simu()





