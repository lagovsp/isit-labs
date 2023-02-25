import math
from typing import Callable
from datetime import datetime
from itertools import product
from texttable import Texttable


class TF:
    def __init__(self,
                 f: Callable[[float], float],
                 f_der: Callable[[float], float],
                 threshold_val: float):
        self.f = f
        self.f_der = f_der
        self.y_threshold_val = threshold_val

    def y(self, net: float) -> (int, float):
        out = self.f(net)
        return (1 if out >= self.y_threshold_val else 0), out


th_f: Callable[[float], float] = lambda net: net
th_f_der: Callable[[float], float] = lambda net: 1
threshold_tf = TF(th_f, th_f_der, 0)

sig_f: Callable[[float], float] = lambda net: (math.tanh(net) + 1) / 2
sig_d_der: Callable[[float], float] = lambda net: sig_f(net) * (1 - sig_f(net))
sigmoid_tf = TF(sig_f, sig_d_der, 0.5)


class Net:
    def __init__(self,
                 tf: TF,
                 weights_num: int,
                 norm: float,
                 name: str = None):
        self.weights: list[float] = [0] * weights_num
        self.norm = norm
        self.tf = tf
        self.name = name

    def predict(self, args: list[int]) -> (int, float, float):
        net = sum([args[i] * w for i, w in enumerate(self.weights)])
        return *self.tf.y(net), net

    def _correct_weights(self, net: float, delta: int, xs: list[int]):
        for i in range(len(self.weights)):
            self.weights[i] += self.norm * delta * self.tf.f_der(net) * xs[i]

    def learn_epoch(self, xs_sets: list[list[int]]):
        for xs in xs_sets:
            y, out, net = self.predict(xs[:-1])
            self._correct_weights(net, xs[-1] - y, xs)


def test_sets(net: Net, xs_sets: list[list[int]]) -> (list[int], int):
    answers, mistakes = list(), 0
    for xs in xs_sets:
        net_ans, *_ = net.predict(xs[:-1])
        answers.append(net_ans)
        mistakes += 0 if net_ans == xs[-1] else 1
    return answers, mistakes


def learn(net: Net,
          sets: list[list[int]],
          learn_indexes: set[int],
          epoch_limit=None) -> (bool, list[list]):
    epochs = list()
    learn_sets = [sets[i] for i in learn_indexes]

    j = 0
    while True:
        answers, mistakes = test_sets(net, sets)
        epochs.append([j, net.weights.copy(), answers.copy(), mistakes])
        if mistakes == 0:
            return True, epochs
        if epoch_limit is not None and j > epoch_limit:
            return False, None
        j += 1
        net.learn_epoch(learn_sets)


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


def modify_lists(logs_list: list[list]) -> list[list]:
    for i, log in enumerate(logs_list):
        log[1] = ''.join(map('{: 6.2f}'.format, log[1]))
        log[2] = ''.join(map(str, log[2]))
    return logs_list


def custom_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


VAR = 6
ARG_NUM = 4


def bf(args: list[int]) -> bool:
    args = list(map(lambda x: False if x == 0 else True, args))
    return (args[2] and args[3]) or (not args[0]) or (not args[1])


INPUTS = list(map(append_left_true,
                  list(map(get_append_bf_val_fun(bf),
                           list(map(list,
                                    list(product([0, 1], repeat=ARG_NUM))))))))


def display_net(logs: list[list],
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
        logger.write(f'Net learned from sets {train_set}\n')
        for i in train_set:
            logger.write(f'{i}\t—> {INPUTS[i][1:-1]}\n')
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
