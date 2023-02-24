from single_layer_net import *


def main():
    time = custom_time_str()

    with open(f'tt-v{VAR}.dat', 'w') as logger:
        logger.write(truthtable(bf, ARG_NUM))

    # All Threshold
    net_threshold_all = Net(weights_num=ARG_NUM + 1,
                            norm=0.3,
                            af=threshold_af_out,
                            af_der=threshold_af_der,
                            name=f'th-fl-v{VAR}')
    display_net(*learn(net_threshold_all,
                       INPUTS,
                       set(range(len(INPUTS))))[1:3],
                to_file=True,
                file_name=f'{net_threshold_all.name}-{time}')

    # All Logistic
    net_logistic_all = Net(weights_num=ARG_NUM + 1,
                           norm=0.3,
                           af=logistic_af_out,
                           af_der=logistic_af_der,
                           name=f'l-fl-v{VAR}')
    display_net(*learn(net_logistic_all,
                       INPUTS,
                       set(range(len(INPUTS))))[1:3],
                to_file=True,
                file_name=f'{net_logistic_all.name}-{time}')


if __name__ == '__main__':
    main()
