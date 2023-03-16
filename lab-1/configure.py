from typing import Callable
from itertools import product
from texttable import Texttable
from datetime import datetime
from src.single_layer_net import TF
import math

VAR = 6
ARG_NUM = 4

_th_f: Callable[[float], float] = lambda net: net
_th_f_der: Callable[[float], float] = lambda net: 1
THRESHOLD_TF = TF(_th_f, _th_f_der, 0)

_sig_f: Callable[[float], float] = lambda net: (math.tanh(net) + 1) / 2
_sig_f_der: Callable[[float], float] = lambda net: 1 / (2 * (math.cosh(net) ** 2))
SIGMOID_TF = TF(_sig_f, _sig_f_der, 0.5)

TF_TYPES = {
    'TH': THRESHOLD_TF,
    'SIG': SIGMOID_TF,
}


def modify_lists(logs_list: list[list]) -> list[list]:
    for i, log in enumerate(logs_list):
        log[1] = ''.join(map('{: 6.2f}'.format, log[1]))
        log[2] = ''.join(map(str, log[2]))
    return logs_list


def custom_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_append_bf_val_fun(f: Callable[[list[int]], int]) -> Callable[[list[int]], list[int]]:
    def _append_bf_val(args: list[int]) -> list[int]:
        args.append(1 if f(args) else 0)
        return args

    return _append_bf_val


def append_bf_val(f: Callable[[list[int]], bool], args: list[int]) -> list[int]:
    args.append(1 if f(args) else 0)
    return args


def append_left_true(args: list[int]) -> list[int]:
    args.insert(0, 1)
    return args


def bf(args: list[int]) -> bool:
    args = list(map(lambda x: False if x == 0 else True, args))
    # return (not ((args[0]) and (args[1]))) and args[2] and args[3] # Enya
    # return (args[0] or args[1] or args[3]) and args[2]  # Roman
    return (args[2] and args[3]) or (not args[0]) or (not args[1])  # Sergey


def create_sets_from_args_num(args: int) -> list[list[int]]:
    return list(map(list,
                    list(product([0, 1], repeat=args))))


INPUTS = list(map(append_left_true,
                  list(map(get_append_bf_val_fun(bf),
                           list(map(list,
                                    list(product([0, 1], repeat=ARG_NUM))))))))


def display_net(inputs: list[list[int]],
                logs: list[list],
                train_set: set[int],
                to_file: bool = False,
                file_name: str = f'net-{custom_time_str()}') -> None:
    logs = modify_lists(logs)
    logs.insert(0, ['k', 'w', 'y', 'E'])

    t = Texttable()
    t.set_chars(['—', '|', '+', '—'])
    t.set_cols_dtype(['i', 't', 't', 'i'])
    t.set_deco(Texttable.BORDER | Texttable.HEADER | Texttable.VLINES)
    t.add_rows(logs)

    if not to_file:
        print(t.draw())
        return
    with open(f'{file_name}.log', 'w') as logger:
        logger.write(F'NET LEARNED FROM SETS {train_set}\n')
        for i in train_set:
            logger.write(f'{i}\t—> {inputs[i]}\n')
        logger.write(t.draw())


def truthtable(f: Callable[[list[int]], bool], n: int) -> str:
    args = list(map(get_append_bf_val_fun(f),
                    list(map(list,
                             list(product([0, 1], repeat=n))))))
    args.insert(0, [f'x{i}' for i in range(1, n + 1)] + ['f'])
    args[0].insert(0, 'n')
    for i in range(2 ** n):
        args[i + 1].insert(0, i)

    t = Texttable()
    t.set_chars(['—', '|', '+', '—'])
    t.set_deco(Texttable.BORDER | Texttable.HEADER | Texttable.VLINES)
    t.add_rows(args)
    return t.draw()
