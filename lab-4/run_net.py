import sys

from rbf_net import *
import matplotlib.pyplot as plt
from configure import *


def main():
    ep_limit = int(sys.argv[1])  # recommended > 30
    norm = float(sys.argv[2])  # 0 < norm <= 1
    name = sys.argv[3]  # any string for table name
    sets = set(map(int, sys.argv[4:]))  # specify the learning sets, all used if empty

    assert 0 < norm <= 1
    print(F'NORM: {norm}')

    assert ep_limit > 0
    print(F'EPS LIMIT: {ep_limit}')

    if not sets:
        sets = set(range(len(INPUTS)))
    print(F'PROCESSING {sorted(sets)} . . .')

    net = RBFNet(f_tf=THRESHOLD_TF, norm=norm, name=F'{name}')
    status, logs = learn(net, INPUTS, sets, epoch_limit=ep_limit)

    display_net(INPUTS, logs, sets, to_file=True, file_name=F'{net.name}')

    plt.title(F'{"SUCCESS" if status else "FAIL"} {name} {norm} {sorted(sets)}')
    plt.xlabel('Epochs, k')
    plt.ylabel('Errors, E')

    plt.plot(list(range(1, len(logs[1:]) + 1)), [log[3] for log in logs[1:]], marker='^')
    plt.gcf().savefig(F'{net.name}.png', dpi=500)

    if not status:
        print(F'NET CANNOT BE LEARNED WITHIN {ep_limit} EPOCHS ON SETS {sorted(sets)}')
        return
    print(F'FINISHED SUCCESSFULLY ({len(logs) - 1} EPS)')


if __name__ == '__main__':
    main()
