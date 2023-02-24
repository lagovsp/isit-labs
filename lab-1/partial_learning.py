import matplotlib.pyplot as plt

from single_layer_net import *
import itertools


class NetSetSet:
    def __init__(self, s: list = None):
        self.sets = list() if s is None else s

    def add_set(self, s: set):
        if s not in self.sets:
            self.sets.append(s)

    def pop(self) -> (bool, type):
        if self.sets:
            return True, self.sets[0]
        return False, None

    def __repr__(self) -> str:
        sets = [s.__repr__() for s in self.sets]
        return '\n'.join(sets)


def find_base_sets(epoch_limit: int,
                   norm: float,
                   tf: Callable[[float], int],
                   tf_der: Callable[[float], float],
                   aim_set_len: int,
                   cur_set_len: int,
                   possible_sets: NetSetSet = None) -> (bool, NetSetSet):
    nss = NetSetSet()
    for ps in possible_sets.sets:
        n = Net(weights_num=ARG_NUM + 1, norm=norm, tf=tf, tf_der=tf_der)
        status, *_ = learn(n,
                           INPUTS,
                           set(ps),
                           epoch_limit=epoch_limit)
        if not status:
            continue
        if aim_set_len == cur_set_len:
            nss.add_set(ps)
            return True, nss
        for item in list(itertools.combinations(ps, cur_set_len - 1)):
            nss.add_set(set(item))
    if not nss:
        return False, None
    return find_base_sets(epoch_limit=epoch_limit,
                          norm=norm,
                          tf=tf,
                          tf_der=tf_der,
                          aim_set_len=aim_set_len,
                          cur_set_len=cur_set_len - 1,
                          possible_sets=nss)


def main():
    time = custom_time_str()
    set_len = 5

    plt.title('Partial learning')
    plt.xlabel('Epochs')
    plt.ylabel('Errors, E')

    # Partial Threshold
    nss_th = NetSetSet([set(range(2 ** ARG_NUM))])
    status_th, base_sets_th = find_base_sets(epoch_limit=60,
                                             norm=0.3,
                                             tf=threshold_tf_out,
                                             tf_der=threshold_tf_der,
                                             aim_set_len=set_len,
                                             cur_set_len=2 ** ARG_NUM,
                                             possible_sets=nss_th)
    print(f'These length-{set_len} sets can be used (TH):')
    print(base_sets_th)
    net_threshold = Net(weights_num=ARG_NUM + 1,
                        norm=0.3,
                        tf=threshold_tf_out,
                        tf_der=threshold_tf_der,
                        name=f'th-pl-v{VAR}')
    _, train_set_th = base_sets_th.pop()
    status_th, logs_th, learn_indexes_th = learn(net_threshold,
                                                 INPUTS,
                                                 train_set_th)
    display_net(logs_th,
                learn_indexes_th,
                to_file=True,
                file_name=f'{net_threshold.name}-{time}')
    plt.plot(list(range(len(logs_th[1:]))), [log[3] for log in logs_th[1:]],
             label='Threshold-TF',
             marker='^')

    # Partial Logistic
    nss_l = NetSetSet([set(range(2 ** ARG_NUM))])
    status_l, base_sets_l = find_base_sets(epoch_limit=60,
                                           norm=0.3,
                                           tf=logistic_tf_out,
                                           tf_der=logistic_tf_der,
                                           aim_set_len=set_len,
                                           cur_set_len=2 ** ARG_NUM,
                                           possible_sets=nss_l)
    print(f'These length-{set_len} sets can be used (L):')
    print(base_sets_l)
    net_logistic = Net(weights_num=ARG_NUM + 1,
                       norm=0.3,
                       tf=logistic_tf_out,
                       tf_der=logistic_tf_der,
                       name=f'l-pl-v{VAR}')
    _, train_set_l = base_sets_l.pop()
    status_l, logs_l, learn_indexes_l = learn(net_logistic,
                                              INPUTS,
                                              train_set_l)
    display_net(logs_l,
                learn_indexes_l,
                to_file=True,
                file_name=f'{net_logistic.name}-{time}')
    plt.plot(list(range(len(logs_l[1:]))), [log[3] for log in logs_l[1:]],
             label='Logistic-TF',
             marker='o')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
