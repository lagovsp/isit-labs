import matplotlib.pyplot as plt

from single_layer_net import *
import itertools


class NetSetSet:
    def __init__(self, s: dict[tuple] = None):
        self.dicts = s if s is not None else dict()

    def add_set(self, s: tuple, epochs: int):
        self.dicts.update({s: epochs})

    def pop(self) -> (bool, type):
        if self.dicts:
            k, v = self.dicts.popitem()
            return True, k
        return False, None

    def __repr__(self) -> str:
        sets = [s.__repr__() for s in self.dicts.items()]
        return '\n'.join(sets)


def find_base_sets(epoch_limit: int,
                   norm: float,
                   tf: TF,
                   cur_set_len: int,
                   possible_sets: NetSetSet) -> (bool, NetSetSet):
    nss = NetSetSet()
    while True:
        print(f'NOW CHECK LEN {cur_set_len}')
        for ps in possible_sets.dicts.copy():
            n = Net(weights_num=ARG_NUM + 1, norm=norm, tf=tf)
            status, history = learn(n,
                                    INPUTS,
                                    set(ps),
                                    epoch_limit=epoch_limit)
            if not status:
                possible_sets.dicts.pop(ps)
                continue
            possible_sets.dicts.update({ps: len(history)})
            for item in list(itertools.combinations(ps, cur_set_len - 1)):
                nss.add_set(tuple(sorted(item)), None)
        if not nss.dicts or cur_set_len == 1:
            return True, possible_sets
        if cur_set_len < 6:
            if not nss.dicts:
                print('\n'.join([t.__repr__() for t in possible_sets.dicts]))
            else:
                print(f'{possible_sets.dicts}')
        possible_sets, nss = nss, NetSetSet()
        cur_set_len -= 1


def main():
    plt.title('Partial learning')
    plt.xlabel('Epochs')
    plt.ylabel('Errors, E')

    # Partial Threshold
    nss_th = NetSetSet({tuple(range(2 ** ARG_NUM)): None})
    status_th, base_sets_th = find_base_sets(epoch_limit=200,
                                             norm=0.3,
                                             tf=THRESHOLD_TF,
                                             cur_set_len=2 ** ARG_NUM,
                                             possible_sets=nss_th)
    s, eps = base_sets_th.dicts.popitem()
    s = set(s)
    print(f'These len-{len(s)} sets can be used (TH):')
    base_sets_th.add_set(s, eps)
    print(base_sets_th)
    # mine = {4, 8, 13, 14, 15}
    # mine = {6, 9, 13, 14}
    net_threshold = Net(weights_num=ARG_NUM + 1,
                        norm=0.3,
                        tf=THRESHOLD_TF,
                        name=f'th-pl-v{VAR}')
    status_th, logs_th, learn_indexes_th = learn(net_threshold,
                                                 INPUTS,
                                                 s)
    # {6, 9, 13, 14})
    display_net(logs_th,
                learn_indexes_th,
                to_file=True,
                file_name=f'{net_threshold.name}')
    plt.plot(list(range(len(logs_th[1:]))), [log[3] for log in logs_th[1:]],
             label='Threshold-TF',
             marker='^')

    # Partial Sigmoid
    # nss_l = NetSetSet([set(range(2 ** ARG_NUM))])
    # status_l, base_sets_sig = find_base_sets(epoch_limit=100,
    #                                          norm=0.3,
    #                                          tf=sigmoid_tf,
    #                                          cur_set_len=2 ** ARG_NUM,
    #                                          possible_sets=nss_l)
    # print(f'These length-{len(base_sets_sig[0])} sets can be used (SIG):')
    # print(base_sets_sig)
    sig_mine = {4, 8, 13, 14, 15}
    net_logistic = Net(weights_num=ARG_NUM + 1,
                       norm=0.3,
                       tf=SIGMOID_TF,
                       name=f'sig-pl-v{VAR}')
    # _, train_set_l = base_sets_sig.pop()
    status_l, logs_l = learn(net_logistic,
                             INPUTS,
                             # train_set_l)
                             sig_mine)
    display_net(logs_l,
                sig_mine,
                to_file=True,
                file_name=f'{net_logistic.name}')
    plt.plot(list(range(len(logs_l[1:]))), [log[3] for log in logs_l[1:]],
             label='Sigmoid-TF',
             marker='o')
    plt.legend()
    plt.gcf().savefig(f'pl-errors-v{VAR}.png', dpi=500)


if __name__ == '__main__':
    main()
