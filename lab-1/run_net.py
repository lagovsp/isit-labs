import sys

from src.single_layer_net import *
import matplotlib.pyplot as plt
from configure import *


def main():
    tf_type = sys.argv[1].upper()  # types = {'TH', 'SIG'}
    ep_limit = int(sys.argv[2])  # recommended > 30
    norm = float(sys.argv[3])  # 0 < norm <= 1
    name = sys.argv[4]  # any string for table name
    sets = set(map(int, sys.argv[5:]))  # specify the learning sets, all used if empty

    tf = TF_TYPES.get(tf_type)
    assert tf is not None
    print(F'TF TYPE: {tf_type}')

    assert 0 < norm <= 1
    print(F'NORM: {norm}')

    if not sets:
        sets = set(range(len(INPUTS)))
    print(F'PROCESSING {sets} . . .')

    net = Net(weights_num=ARG_NUM + 1, norm=norm, tf=tf, name=F'{name}')
    status, logs = learn(net, INPUTS, sets, epoch_limit=ep_limit)

    if not status:
        print(F'NET CANNOT BE LEARNED WITHIN {ep_limit} EPOCHS ON SETS {sets}')
        return

    display_net(INPUTS, logs, sets, to_file=True, file_name=F'{net.name}-{tf_type}')

    plt.title(F'{name} {tf_type} {norm} {tuple(sorted(sets))}')
    plt.xlabel('Epochs')
    plt.ylabel('Errors, E')

    plt.plot(list(range(1, len(logs[1:]) + 1)), [log[3] for log in logs[1:]], marker='^')
    plt.gcf().savefig(F'{net.name}-{tf_type}.png', dpi=500)

    print('FINISHED SUCCESSFULLY')


if __name__ == '__main__':
    main()
