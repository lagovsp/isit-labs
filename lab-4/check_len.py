import sys

from rbf_net import *
import itertools
from colorama import Fore
from configure import *


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
        rbf_net = RBFNet(f_tf=tf, norm=norm, name='check-net')
        rbf_status, rbf_history = learn(rbf_net, INPUTS, set(s), epoch_limit=epoch_limit)
        print(s, ((Fore.GREEN + F'PASSED') if rbf_status else (Fore.RED + F'FAILED\n')) + Fore.RESET, end='')

        if not rbf_status:
            continue
        print(F' {len(rbf_history)} EPS')

        found_sets.update({tuple(sorted(s)): len(rbf_history)})
    return found_sets


def main():
    tf_type = sys.argv[1].upper()  # types = {'TH', 'SIG'}
    ep_limit = int(sys.argv[2])  # recommended > 30
    norm = float(sys.argv[3])  # 0 < norm <= 1
    set_len = int(sys.argv[4])  # 0 < set_len < len(INPUTS)

    tf = TF_TYPES.get(tf_type)
    assert tf is not None
    print(F'TF TYPE: {tf_type}')

    assert ep_limit > 0
    print(F'EPS LIMIT: {ep_limit}')

    assert 0 < norm <= 1
    print(F'NORM: {norm}')

    print(F'CHECKING LEN-{set_len} SETS . . .')
    sets = check_sets_of_length(set_len, ep_limit, norm, tf)

    if not sets:
        print('NO SETS FOUND')
        return

    print(f'FOUND SETS: {len(sets)}')
    print('\n'.join(map(str, sets.items())))


if __name__ == '__main__':
    main()
