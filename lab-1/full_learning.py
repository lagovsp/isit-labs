from single_layer_net import *


def main():
    time = custom_time_str()

    with open(f'truthtable-v{var}.dat', 'w') as logger:
        logger.write(truthtable(bf, 4))

    # All sets used
    net_threshold_all = Net(weights_num=5,
                            norm=0.3,
                            af=threshold_af_out,
                            af_der=threshold_af_der,
                            name=f'threshold-all-v{var}')
    display_net(learn(net_threshold_all,
                      inputs,
                      set(range(len(inputs))),
                      test_same=True),
                to_file=True,
                file_name=f'{net_threshold_all.name}-{time}')

    net_logistic_all = Net(weights_num=5,
                           norm=0.3,
                           af=logistic_af_out,
                           af_der=logistic_af_der,
                           name=f'logistic-all-v{var}')
    display_net(learn(net_logistic_all,
                      inputs,
                      set(range(len(inputs))),
                      test_same=True),
                to_file=True,
                file_name=f'{net_logistic_all.name}-{time}')


if __name__ == '__main__':
    main()
