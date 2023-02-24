import math
from typing import Callable
from datetime import datetime
from itertools import product
from texttable import Texttable


class Net:
    def __init__(self,
                 weights_num: int = None,
                 norm: float = None,
                 af: Callable[[float], int] = None,
                 af_der: Callable[[float], float] = None,
                 name: str = None):
        self.weights: list[float] = [0] * weights_num
        self.norm = norm
        self.af = af
        self.af_der = af_der
        self.name = name

    def predict(self, args: list[int]) -> (int, float):
        net = sum([args[i] * w for i, w in enumerate(self.weights)])
        return self.af(net), net

    def _correct_weights(self, net: float, delta: int, sample: list[int]):
        for i, w in enumerate(self.weights):
            self.weights[i] += self.norm * delta * self.af_der(net) * sample[i]

    def learn_epoch(self, samples: list[list[int]]):
        for i, sample in enumerate(samples):
            answer, net = self.predict(sample[:-1])
            delta = sample[len(self.weights)] - answer
            self._correct_weights(net, delta, sample)


def test(net: Net, sets: list[list[int]]) -> (list[int], int):
    answers, mistakes = list(), 0
    for s in sets:
        net_ans, _ = net.predict(s[:-1])
        answers.append(net_ans)
        mistakes += 0 if net_ans == s[-1] else 1
    return answers, mistakes


def learn(net: Net,
          sets: list[list[int]],
          learn_indexes: set[int],
          epoch_limit=None) -> (bool, list[list], set[int]):
    success_count, epochs, i, = 0, list(), 0
    learn_sets = [sets[i] for i in learn_indexes]
    test_sets = list()
    for j in range(len(sets)):
        test_sets.append(sets[j])

    while success_count < 1:
        if isinstance(epoch_limit, int) and i > epoch_limit:
            return False, None, learn_indexes
        # print(f'in local learning i({i}) <= ep_lim({epoch_limit})')
        answers, mistakes = test(net, test_sets)
        epochs.append([i, net.weights.copy(), answers.copy(), mistakes])
        success_count += 1 if mistakes == 0 else 0
        net.learn_epoch(learn_sets)
        i += 1
    return True, epochs, learn_indexes


def get_append_bf_val_fun(f: Callable[[list[int]], bool]) -> Callable[[list[int]], list[int]]:
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
PARTIAL_BASE_LEN = 5

bf: Callable[[list[int]], bool] = lambda args: (args[2] and args[3]) or (not args[0]) or (not args[1])

INPUTS = list(map(append_left_true,
                  list(map(get_append_bf_val_fun(bf),
                           list(map(list,
                                    list(product([0, 1], repeat=ARG_NUM))))))))

threshold_af_out: Callable[[float], int] = lambda net: 1 if net >= 0 else 0
threshold_af_der: Callable[[float], float] = lambda net: 1

logistic_af_out: Callable[[float], int] = lambda net: 1 if (math.tanh(net) + 1) / 2 >= 0.5 else 0
logistic_af_der: Callable[[float], float] = lambda net: 1 / (2 * (math.cosh(net) ** 2))


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
            logger.write(f'{i} —> {INPUTS[i][1:-1]}\n')
        logger.write(t.draw())


def truthtable(f: Callable[[list[int]], bool], n: int) -> str:
    args = list(map(get_append_bf_val_fun(f),
                    list(map(list,
                             list(product([0, 1], repeat=n))))))
    args.insert(0, [f'x{i}' for i in range(1, n + 1)] + ['f'])

    t = Texttable()
    t.set_chars(['—', '|', '+', '—'])
    t.set_deco(Texttable.BORDER | Texttable.HEADER | Texttable.VLINES)
    t.add_rows(args)
    return t.draw()
