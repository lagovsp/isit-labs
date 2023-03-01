import sys

from single_layer_net import *
import itertools
from colorama import Fore


def check_sets_of_length(set_len: int,
                         epoch_limit: int,
                         norm: float,
                         tf: TF) -> dict[tuple, int]:
    found_sets: dict[tuple, int] = dict()
    if not INPUTS:
        return found_sets
    if not 0 < set_len <= len(INPUTS):
        return found_sets
    for s in map(set, itertools.combinations(set(range(len(INPUTS))), set_len)):
        net = Net(weights_num=ARG_NUM + 1, norm=norm, tf=tf)
        status, history = learn(net,
                                INPUTS,
                                set(s),
                                epoch_limit=epoch_limit)
        print(s, end=' ')
        if status:
            print(Fore.GREEN + F'PASSED' + Fore.RESET + F' {len(history)} EPS')
        else:
            print(Fore.RED + F'FAILED' + Fore.RESET)

        if not status:
            continue
        found_sets.update({tuple(sorted(s)): len(history)})
    return found_sets


def main():
    tf_type = int(sys.argv[1])  # types = {0, 1}
    ep_limit = sys.argv[2]  # recommended > 30
    norm = float(sys.argv[3])  # 0 < norm <= 1
    set_len = sys.argv[4]  # 0 < set_len < len(INPUTS)

    if tf_type not in TF_TYPES:
        raise Exception('UNKNOWN TF TYPE')
    print(F'TF TYPE: {TF_TYPES.get(tf_type)[0]}')

    if not 0 < norm <= 1:
        raise Exception(F'BAD NORM GIVEN ({norm})')
    print(F'NORM: {norm}')

    print(F'CHECKING LEN-{set_len} SETS . . .')
    sets = check_sets_of_length(int(set_len),
                                int(ep_limit),
                                float(norm),
                                TF_TYPES.get(tf_type)[1])

    if not sets:
        print('NO SETS FOUND')
        return
    print(f'FOUND SETS: {len(sets)}')
    print('\n'.join(map(str, sets.items())))


if __name__ == '__main__':
    main()
