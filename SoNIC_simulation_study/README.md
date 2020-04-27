[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SoNIC_simulation_study** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet:  SoNIC_simulation_study   

Published in: SoNIC

Description: 'simulation study for SoNIC model with parameters N=T=100, s=1, K=2..30'

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

See also: 
- SoNIC_AAPL_BTC
- SoNIC_AAPL_BTC_stability

Author: Yegor Klochkov

Submitted: Mon, Jun 17 2019 by Yegor Klochkov

```

### PYTHON Code
```python

import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, log

if __name__ == '__main__':
    #cl_nums = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    cl_nums = [5]
    pmin = 0.5
    n = 20
    T = 400
    tasks = [
        ["python", "simusimu.py", str(n), str(cl_num), '1', str(T), str(pmin)] for cl_num in cl_nums
    ]
    if True:
        procs = [subprocess.Popen(task) for task in tasks]
        for i, proc in enumerate(procs):
            out, err = proc.communicate()
            if err is not None:
                print("ERROR for TASK {}".format(" ".join(tasks[i])))
                print(err)

    data = []
    for cl_num in cl_nums:
        data.append(pd.read_csv("simu_n{}k{}s1T{}.csv".format(n, cl_num, T)))

        ############################
        #
        # lambda graphs
        #
        #
    if True:
        for i, cl_num in enumerate(cl_nums):
            plt.plot(data[i]['alpha'] / sqrt(log(n) / (T * (float(pmin) **2)))
                     ,np.minimum(np.array(data[i]['theta_diff']), 5), label=str(cl_num), marker='o')
        plt.savefig('alpha_loss.png', dpi=1000)
        plt.show()
        plt.clf()

    ############################
    #
    #
    # loss with best lambda
    #
    #
    if False:
        theta_min_diffs = []
        for i, cl_num in enumerate(cl_nums):
            theta_min_diffs.append(np.min(data[i]['theta_diff']))
            
        plt.plot(cl_nums, theta_min_diffs)
        plt.savefig('n_loss.png', dpi=1000)
        #plt.show()
        plt.clf()

    ############################
    #
    #
    # cluster distances
    #
    #
    if False:
        plt.plot(data[5]['alpha'], data[5]['cl_diff'])
        plt.savefig('cl_diff.png', dpi=1000)
        #plt.show()
        plt.clf()


```

automatically created on 2020-04-27