import common.alternating as alternating
import numpy as np
import sys
from pandas import DataFrame


SIM_NUM = 20

if __name__ == '__main__':
    assert (len(sys.argv) == 5 or len(sys.argv) == 6)
    n = int(sys.argv[1])
    cl_num = int(sys.argv[2])
    s = int(sys.argv[3])
    T = int(sys.argv[4])
    if len(sys.argv) == 5:
        pmin = 1.0
    else:
        pmin = float(sys.argv[5])

    theta_diffss = []
    cl_diffss = []

    for i in range(SIM_NUM):
        alphas, theta_diffs, cl_diffs, node_infls = alternating.simu(n, cl_num, s, T, pmin=pmin)
        theta_diffss.append(theta_diffs)
        cl_diffss.append(cl_diffs)
        print("N={}: {}/{} done".format(n, i+1, SIM_NUM))
    theta_diffs_mean = np.mean(np.array(theta_diffss), axis=0)
    cl_diffs_mean = np.mean(np.array(cl_diffss), axis=0)

    value = np.empty((np.size(alphas), n + 3))
    value[:, 0] = alphas
    value[:, 1] = theta_diffs_mean
    value[:, 2] = cl_diffs_mean
    value[:, 3:] = node_infls
    columns = ['alpha', 'theta_diff', 'cl_diff'] + [str(i) for i in range(n)]
    df = DataFrame(data=value, columns=columns)

    filename = "simu_n{}k{}s{}T{}.csv".format(n, cl_num, s, T)
    df.to_csv(filename)
