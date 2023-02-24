import matplotlib.pyplot as plt

from single_layer_net import *


def main():
    time = custom_time_str()

    with open(f'tt-v{VAR}.dat', 'w') as logger:
        logger.write(truthtable(bf, ARG_NUM))

    plt.title('Full learning')
    plt.xlabel('Epochs')
    plt.ylabel('Errors, E')

    # Full Threshold
    net_threshold_all = Net(weights_num=ARG_NUM + 1,
                            norm=0.3,
                            tf=threshold_tf_out,
                            tf_der=threshold_tf_der,
                            name=f'th-fl-v{VAR}')
    status_th, logs_th, learn_indexes_th = learn(net_threshold_all,
                                                 INPUTS,
                                                 set(range(len(INPUTS))))
    display_net(logs_th,
                learn_indexes_th,
                to_file=True,
                file_name=f'{net_threshold_all.name}-{time}')
    plt.plot(list(range(len(logs_th[1:]))), [log[3] for log in logs_th[1:]],
             label='Threshold-TF',
             marker='^')

    # Full Logistic
    net_logistic_all = Net(weights_num=ARG_NUM + 1,
                           norm=0.3,
                           tf=logistic_tf_out,
                           tf_der=logistic_tf_der,
                           name=f'l-fl-v{VAR}')
    status_l, logs_l, learn_indexes_l = learn(net_logistic_all,
                                              INPUTS,
                                              set(range(len(INPUTS))))
    display_net(logs_l,
                learn_indexes_l,
                to_file=True,
                file_name=f'{net_logistic_all.name}-{time}')
    plt.plot(list(range(len(logs_l[1:]))), [log[3] for log in logs_l[1:]],
             label='Logistic-TF',
             marker='o')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
