import sys

from single_layer_net import *
import matplotlib.pyplot as plt


def main():
    tf_type = int(sys.argv[1])  # types = {0, 1}
    ep_limit = int(sys.argv[2])  # recommended > 30
    norm = float(sys.argv[3])  # 0 < norm <= 1
    name = sys.argv[4]  # any string for table name
    sets = set(map(int, sys.argv[5:]))  # specify the learning sets, all used if empty

    if tf_type not in TF_TYPES:
        raise Exception('UNKNOWN TF TYPE')
    tf_name = TF_TYPES.get(tf_type)[0]
    print(F'TF TYPE: {tf_name}')

    if not 0 < norm <= 1:
        raise Exception(F'BAD NORM GIVEN ({norm})')
    print(F'NORM: {norm}')

    if not sets:
        sets = set(range(len(INPUTS)))
    print(F'PROCESSING {sets} . . .')

    net = Net(weights_num=ARG_NUM + 1,
              norm=norm,
              tf=TF_TYPES.get(tf_type)[1],
              name=F'{name}')
    status, logs = learn(net,
                         INPUTS,
                         sets,
                         epoch_limit=ep_limit)
    if not status:
        print(F'NET CANNOT BE LEARNED WITHIN {ep_limit} EPOCHS ON SETS {sets}')
        return

    display_net(logs,
                sets,
                to_file=True,
                file_name=f'{net.name}-{tf_name}')

    plt.title(F'{name} {tf_name} {norm} {tuple(sorted(sets))}')
    plt.xlabel('Epochs')
    plt.ylabel('Errors, E')

    plt.plot(list(range(1, len(logs[1:]) + 1)), [log[3] for log in logs[1:]], marker='^')
    plt.gcf().savefig(F'{net.name}-{tf_name}.png', dpi=500)

    print('FINISHED SUCCESSFULLY')


if __name__ == '__main__':
    main()
