import subprocess


if __name__ == '__main__':
    tasks = [
    #    N,c_num, pmin
        (100, 2, 1.0),
        (100, 2, 0.5),
        (100, 5, 1.0),
        (100, 5, 0.5)
    ]

    # s = 1 always

    procs = [subprocess.Popen(
        ["python", "simustability.py", str(task[0]), str(task[1]), "1", str(task[2])]
    ) for task in tasks]
    for i, proc in enumerate(procs):
        out, err = proc.communicate()
        if err is not None:
            print("ERROR for TASK {}".format(" ".join(tasks[i])))
            print(err)









