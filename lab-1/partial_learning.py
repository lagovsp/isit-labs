from single_layer_net import *


def main():
    time = custom_time_str()

    with open(f'tt-v{var}.dat', 'w') as logger:
        logger.write(truthtable(bf, 4))

    # Part learning
    net_threshold_part = Net(weights_num=5,
                             norm=0.3,
                             af=threshold_af_out,
                             af_der=threshold_af_der,
                             name=f'th-pl-v{var}')
    display_net(learn(net_threshold_part,
                      inputs,
                      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13},
                      test_same=False),
                to_file=True,
                file_name=f'{net_threshold_part.name}-{time}')

    net_logistic_part = Net(weights_num=5,
                            norm=0.3,
                            af=logistic_af_out,
                            af_der=logistic_af_der,
                            name=f'l-pl-v{var}')
    display_net(learn(net_logistic_part,
                      inputs,
                      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13},
                      test_same=False),
                to_file=True,
                file_name=f'{net_logistic_part.name}-{time}')


if __name__ == '__main__':
    main()
