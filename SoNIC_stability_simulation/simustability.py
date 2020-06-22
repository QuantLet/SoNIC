import numpy as np
import sys
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from math import ceil, floor

import common.alternating as alternating
import common.missing as missing


def do(n, T, c_num, s, pmin):
    # set up theta_star
    c_size = int(n // c_num)
    r = n - c_num * c_size
    ind_star = np.array([int(i // (c_size + 1)) if i < r * (c_size + 1)
                         else int((i - r * (c_size + 1)) // c_size) + r for i in range(n)])
    z_star = alternating.get_index_matrix(c_num, ind_star)

    # define true theta
    s = 1
    assert (s <= 5)
    v_star = np.zeros((c_num, n))
    active_vals = [0.6, -0.4, 0.1, -0.8, 0.2]
    for j in range(c_num):
        v_star[j, j: j + s] = np.array(active_vals[:s]) * ((-1) ** j)

    theta_star = np.matmul(z_star, v_star)
    coef = 0.5 / np.linalg.norm(theta_star, 2)
    theta_star = theta_star * coef

    # generate the time series
    x = alternating.gauss_var1_process(theta_star, 1., T)

    # include missing observations
    mask = np.random.binomial(1, pmin, size=(n, T)).astype(np.float64)
    x_missing = x * mask

    # estimate missing probabilities
    deltas = np.mean(mask, axis=1)

    # number of windows and their length
    sim_num = 6
    win_len = (3 * T) // 4

    # candidates cluster numbers
    c_nums = list(range(2, min(20, n // 2) + 1))

    ans_idx = []
    # ans_theta = []
    # ans_loss = []
    for c in c_nums:
        ress = []
        for (a, b) in [(k * ((T - win_len + 1) // sim_num), k * ((T - win_len + 1) // sim_num) + win_len) for k in
                       range(sim_num)]:
            x_train = x_missing[:, a:b - 1]
            y_train = x_missing[:, a + 1:b]

            D0 = missing.missing_var(x_train, deltas)
            D1 = missing.missing_covar(x_train, y_train, deltas, deltas)

            lmbd = (np.linalg.eigvalsh(D0)[-(c+1)]) * np.sqrt(np.log(n)) / np.sqrt(T * np.min(deltas) ** 2)
            c_size = int(n // c)
            r = n - c * c_size
            ind_init = np.array([int(i // (c_size + 1)) if i < r * (c_size + 1)
                                 else int((i - r * (c_size + 1)) // c_size) + r for i in range(n)])
            ress.append(alternating.matrix_competition("ALTER", 5, c, D0, D1, lmbd, epochs=20))

        ans_idx.append([alternating.index_dist(c, ress[0].index, ress[j].index) for j in range(1, sim_num)])
        # ans_theta.append([np.linalg.norm(ress[0].theta - ress[j].theta) for j in range(1, sim_num)])
        # ans_loss.append(ress[0].loss)

    save_path = "results/simu_n{}t{}cnum{}pmin{}".format(n, T, c_num, pmin)

    res = np.zeros((sim_num-1, len(c_nums)))
    for i, _ in enumerate(c_nums):
        res[:, i] = ans_idx[i]
    df = DataFrame(data=res, columns=c_nums)
    df.to_csv(save_path + ".csv")


def get_yticks_my_way(ymin, ymax):
    steps = [1, 2, 5, 10]
    steps.reverse()
    min_ticks = 5

    for step in steps:
        down = floor(ymin // step)
        up = ceil(ymax // step)

        if up - down > min_ticks:
            return range(down * step, up * step + 1, step)

        if step == 1:
            return range(down, step + 1)


def draw(n, T, c_num, s, pmin):
    save_path = "results/simu_n{}t{}cnum{}pmin{}".format(n, T, c_num, pmin)

    sim_num = 6
    df = read_csv(save_path + ".csv", index_col=0)

    plt.figure()
    for c in df.columns:
        plt.plot([int(c)] * (sim_num - 1), df[c].values, 'o', color='b')

    # set integral ticks
    plt.xticks([int(c) for c in df.columns])
    plt.yticks(get_yticks_my_way(np.min(df.values), np.max(df.values)))

    # save png and low quality jpg
    plt.savefig(save_path + ".png", dpi=500)
    plt.savefig(save_path + ".jpg", dpi=100, optimize=True, quality=30)

    # clean before doing next graph
    plt.clf()


def format_func(value, tick_number):
    # find number of multiples of pi/2
    c_show_max = 15

    return int(value) % (c_show_max + 1)


def draw_together(n, Ts, c_num, s, pmin):
    fig, ax = plt.subplots(figsize=(11,3))

    sim_num = 6
    colors = ['b', 'g', 'r', 'c', 'm']
    c_show_max = 15

    major_ticks = []
    minor_ticks = []

    ymax = 0
    for i, T in enumerate(Ts):
        df = read_csv("results/simu_n{}t{}cnum{}pmin{}.csv".format(n, T, c_num, pmin), index_col=0)
        for c in df.columns:
            if int(c) > c_show_max:
                continue
            tick = (c_show_max + 1) * i + int(c)
            ax.plot([tick] * (sim_num - 1), df[c].values, 'o', color=colors[i])
            ymax = max(ymax, np.max(df[c].values))
            if int(c) % 2 == 0:
                major_ticks.append(tick)
            minor_ticks.append(tick)

    save_path = "results/simu_n{}cnum{}pmin{}".format(n, c_num, pmin)

    ax.xaxis.set_major_locator(plt.FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.yaxis.set_major_locator(plt.FixedLocator(get_yticks_my_way(0, int(ymax))))
    ax.yaxis.set_minor_locator(plt.FixedLocator(list(range(0, int(ymax) + 1))))

    # save png and low quality jpg
    fig.savefig(save_path + ".png", dpi=500)
    fig.savefig(save_path + ".jpg", dpi=100, optimize=True, quality=30)

    # clean before doing next graph
    plt.clf()


if __name__ == '__main__':
    # python simustability.py "n" "c_num" "s" "pmin"(optional, =1 by default)
    #

    assert (len(sys.argv) == 4 or len(sys.argv) == 5)
    n = int(sys.argv[1])
    c_num = int(sys.argv[2])
    s = int(sys.argv[3])
    # T = int(sys.argv[4])
    if len(sys.argv) == 4:
        pmin = 1.0
    else:
        pmin = float(sys.argv[4])

    Ts = [100, 200, 500, 1000, 2000]
    #Ts = [100, 400, 500]
    #for T in Ts:
    #    do(n, T, c_num, s, pmin)

    draw_together(n, Ts, c_num, s, pmin)





