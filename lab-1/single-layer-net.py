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
        mistakes = 0
        for sample in samples:
            answer, net = self.predict(sample[:-1])
            delta = sample[len(self.weights)] - answer
            mistakes += 0 if delta == 0 else 1
            self._correct_weights(net, delta, sample)


def test_sets(net: Net, sets: list[list[int]]) -> (list[int], int):
    answers, mistakes = list(), 0
    for s in sets:
        net_ans, _ = net.predict(s[:-1])
        answers.append(net_ans)
        mistakes += 0 if net_ans == s[-1] else 1
    return answers, mistakes


def learn(net: Net, sets: list[list[int]]) -> list[list]:
    success_count, epochs, i = 0, list(), 0
    while success_count < 1:
        answers, mistakes = test_sets(net, sets)
        epochs.append([i, net.weights.copy(), answers.copy(), mistakes])
        success_count += 1 if mistakes == 0 else 0
        net.learn_epoch(sets)
        i += 1
    return epochs


def append_bf_val(args: list[int]) -> list[int]:
    args.append(1 if bf(args) else 0)
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


def display_net(logs: list[list],
                to_file: bool = False,
                file_name: str = f'net-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}') -> None:
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
        logger.write(t.draw())


bf: Callable[[list[bool]], bool] = lambda args: (args[2] and args[3]) or (not args[0]) or (not args[1])

threshold_af_out: Callable[[float], int] = lambda net: 1 if net >= 0 else 0
threshold_af_der: Callable[[float], float] = lambda net: 1

logistic_af_out: Callable[[float], int] = lambda net: 1 if (math.tanh(net) + 1) / 2 >= 0.5 else 0
logistic_af_der: Callable[[float], float] = lambda net: 1 / (2 * (math.cosh(net) ** 2))

inputs = list(map(append_left_true,
                  list(map(append_bf_val,
                           list(map(list,
                                    list(product([0, 1], repeat=4))))))))


def main():
    global inputs

    net_threshold = Net(weights_num=len(inputs[0]) - 1,
                        norm=0.3,
                        af=threshold_af_out,
                        af_der=threshold_af_der,
                        name='threshold')
    display_net(learn(net_threshold, inputs),
                to_file=True,
                file_name=f'{net_threshold.name}-{custom_time_str()}')

    net_logistic = Net(weights_num=len(inputs[0]) - 1,
                       norm=0.3,
                       af=logistic_af_out,
                       af_der=logistic_af_der,
                       name='logistic')
    display_net(learn(net_logistic, inputs),
                to_file=True,
                file_name=f'{net_logistic.name}-{custom_time_str()}')


if __name__ == '__main__':
    main()
