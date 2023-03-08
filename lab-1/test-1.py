import os
import re
import sys

from src.single_layer_net import *
import matplotlib.pyplot as plt
from configure import *

ARGS = 2
TF_TYPE = 'TH'
EP_LIMIT = 100
NORM = 1


def main():
    name = sys.argv[1]  # so-called name for bf
    vector = re.split('', name)  # bf vector
    assert len(vector) == 6
    vector.pop(0)
    vector.pop(-1)
    vector = list(map(int, vector))

    tf = TF_TYPES.get(TF_TYPE)
    assert tf is not None
    print(F'TF TYPE: {TF_TYPE}')

    assert 0 < NORM <= 1
    print(F'NORM: {NORM}')

    INPUTS = list(map(append_left_true, create_sets_from_args_num(ARGS)))
    for i in range(len(INPUTS)):
        INPUTS[i].append(vector[i])

    sets = set(range(len(INPUTS)))
    print(F'PROCESSING {sets} . . .')

    net = Net(weights_num=ARGS + 1, norm=NORM, tf=tf, name=F'{name}')
    status, logs = learn(net, INPUTS, sets, epoch_limit=EP_LIMIT)

    if not status:
        print(F'NET CANNOT BE LEARNED WITHIN {EP_LIMIT} EPOCHS ON SETS {sets}')
        return

    folder = '2-arg-functions'
    path = os.path.join(os.path.dirname(__file__), folder)
    if not os.path.exists(path):
        os.mkdir(path)

    display_net(INPUTS, logs, sets, to_file=True, file_name=F'{folder}/{net.name}-{TF_TYPE}')

    plt.title(F'{name} {TF_TYPE} {NORM} {tuple(sorted(sets))}')
    plt.xlabel('Epochs')
    plt.ylabel('Errors, E')

    plt.plot(list(range(1, len(logs[1:]) + 1)), [log[3] for log in logs[1:]], marker='^')
    plt.gcf().savefig(F'{folder}/{net.name}-{TF_TYPE}.png', dpi=500)

    print('FINISHED SUCCESSFULLY')


if __name__ == '__main__':
    main()
