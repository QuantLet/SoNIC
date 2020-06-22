[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SoNIC_AAPL_BTC_benchmark** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet:  SoNIC_AAPL_BTC_benchmark   

Published in: SoNIC

Description: 'Comparison of SoNIC with VAR and l1 penalized VAR on AAPL and BTC datasets. We also make comparison with zero Theta, which corresponds to no causality.'

Keywords:
- social network
- network autoregression
- clustering
- communities
- influencer
- central node
- sentiment analysis
- stocktwits
- adjacency matrix
- lasso
- greedy algorithm
- Apple
- Bitcoin

See also: 
- SoNIC_simulation_study
- SoNIC_AAPL_BTC
- SoNIC_AAPL_BTC_stability

Author: Yegor Klochkov

Submitted: Mon, Jun 17 2019 by Yegor Klochkov
```

### PYTHON Code
```python

import numpy as np
from math import ceil, floor

import common.read_data as read_data
import common.missing as missing
from common.alternating import matrix_competition


def get_sonic_theta(Y_train, deltas, num_clusters, lmbd):
    D0 = missing.missing_var(Y_train[:, :-1], deltas)
    D1 = missing.missing_covar(Y_train[:, :-1], Y_train[:, 1:], deltas, deltas)

    result = matrix_competition('ALTER', 5, num_clusters, D0, D1, lmbd, epochs=50)
    return result.theta


def get_prediction(Y_test, theta, deltas):
    x_test = Y_test[:, :-1]
    y_test = Y_test[:, 1:]

    D0_test_x = missing.missing_var(x_test, deltas, make_positive=False)
    D0_test_y = missing.missing_var(y_test, deltas, make_positive=False)
    D1_test = missing.missing_covar(x_test, y_test, deltas, deltas)

    return (D0_test_y - 2 * np.matmul(theta, D1_test) + np.matmul(theta, np.matmul(D0_test_x, theta.T))).trace()


if __name__ == '__main__':
    tasks = [
        '../data/users_daily_timeseries_AAPL.csv'
        , '../data/users_BTC_timeseries_Daily.csv'
    ]

    for task in tasks:
        data_path = task

        Y, deltas, names = read_data.read_stock_twits_user_sentiment(data_path, min_days=50, min_delta=0.5)
        N, tmax = np.shape(Y)

        t_train = ceil(tmax * 0.7)
        Y_train = Y[:, :t_train]
        Y_test = Y[:, t_train-1:]

        # SET NUMBER OF CLUSTERS
        num_clusters = 2

        # theoretical lambda
        D0 = missing.missing_var(Y_train, deltas)
        lmbd = (np.linalg.eigvalsh(D0)[-(num_clusters + 1)]) * np.sqrt(np.log(N)) / np.sqrt(tmax * np.min(deltas) ** 2)

        ##############################
        # we compare four methods:
        # 1) SONIC
        # 2) VAR + lasso = SONIC with K = N
        # 3) VAR = SONIC with K = N and lambda = 0
        # 4) theta = 0 (no causality)
        #

        methods = ["SONIC", "VAR+LASSO", "VAR", "ZERO"]
        thetas = {
            "SONIC": get_sonic_theta(Y_train, deltas, num_clusters, lmbd),
            "VAR+LASSO": get_sonic_theta(Y_train, deltas, N, lmbd),
            "VAR": get_sonic_theta(Y_train, deltas, N, 0),
            "ZERO": np.zeros((N, N)),
        }

        res = {key: get_prediction(Y_test, thetas[key], deltas) for key in methods}

        print()
        print(data_path)
        for key in methods:
            print("{}: {}".format(key, res[key]))

```

automatically created on 2020-04-27