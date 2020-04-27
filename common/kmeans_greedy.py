import numpy as np


# func takes ind!
def find_new_home(func, k, ind, i):
    vals = []
    for label in range(k):
        ind_new = np.array(ind, copy=True)
        ind_new[i] = label

        vals.append(func(ind_new))

    (val_best, label_best) = max([(val, label) for (label, val) in enumerate(vals)])
    if val_best == vals[ind[i]]:
        return ind[i], val_best, False

    return label_best, val_best, True


def greedy_update(func, k, ind):
    n = np.size(ind)

    ind_new = np.array(ind, copy=True)
    ans = False
    for i in range(n):
        label, val, ans_i = find_new_home(func, k, ind_new, i)
        ind_new[i] = label
        if ans_i:
            ans = True

    return ind_new, val, ans


class ReturnInstance:

    def __init__(self, index, value, message, nochange):
        self.index = index
        self.value = value
        self.message = message
        self.nochange = nochange


def kmeans_greedy(func, k, n, iter_limit=10000, init_index=None):
    assert (k <= n)
    if init_index is not None:
        assert (n == np.size(init_index))

    if init_index is None:
        ind = np.random.randint(k, size=n)
    else:
        ind = init_index

    message = "did not converge"
    nochange = False
    for it in range(iter_limit):
        ind, val, ans = greedy_update(func, k, ind)
        if not ans:
            message = "success in {} iterations".format(it)
            if it == 0:
                nochange = True
            break

    return ReturnInstance(ind, val, message, nochange)
