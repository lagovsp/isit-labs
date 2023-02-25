import matplotlib.pyplot as plt
from single_layer_net import *


def main():
    with open(f'truthtable-v{VAR}.dat', 'w') as logger:
        logger.write(truthtable(bf, ARG_NUM))

    plt.title('Full learning')
    plt.xlabel('Epochs')
    plt.ylabel('Errors, E')

    learn_indexes = set(range(len(INPUTS)))

    # Full Threshold
    net_threshold_all = Net(weights_num=ARG_NUM + 1,
                            norm=0.3,
                            tf=threshold_tf,
                            name=f'th-fl-v{VAR}')
    status_th, logs_th = learn(net_threshold_all, INPUTS, learn_indexes)
    display_net(logs_th,
                learn_indexes,
                to_file=True,
                file_name=f'{net_threshold_all.name}')
    plt.plot(list(range(len(logs_th[1:]))), [log[3] for log in logs_th[1:]],
             label='Threshold-TF',
             marker='^')

    # Full Sigmoid
    net_sig_all = Net(weights_num=ARG_NUM + 1,
                      norm=0.3,
                      tf=sigmoid_tf,
                      name=f'sig-fl-v{VAR}')
    status_s, logs_s = learn(net_sig_all, INPUTS, learn_indexes)
    display_net(logs_s,
                learn_indexes,
                to_file=True,
                file_name=f'{net_sig_all.name}')
    plt.plot(list(range(len(logs_s[1:]))), [log[3] for log in logs_s[1:]],
             label='Sigmoid-TF',
             marker='o')

    plt.legend()
    plt.gcf().savefig(f'fl-errors-v{VAR}.png', dpi=500)


if __name__ == '__main__':
    main()
