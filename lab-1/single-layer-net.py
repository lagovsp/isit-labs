import math
from typing import Callable
from texttable import Texttable


class Net:
    def __init__(self,
                 weights_num: int = None,
                 norm: float = None,
                 af: Callable[[float], float] = None,
                 af_der: Callable[[float], float] = None):
        self.weights = [0] * weights_num
        self.norm = norm
        self.af = af
        self.af_der = af_der

    def launch(self, args: list[int]) -> (int, float):
        net = sum([args[i] * w for i, w in enumerate(self.weights)])
        return self.af(net), net

    def learn_epoch(self, samples: list[list[int]]):
        mistakes = 0
        for sid, sample in enumerate(samples):
            print(f'{sid}. x = {sample}')
            answer, net = self.launch(sample[:-1])
            delta = sample[len(self.weights)] - answer
            mistakes += int(math.fabs(delta))
            print(f'\ta = {answer}, delta = {delta}')
            print(f'\tMISTAKES FO FAR {mistakes}')
            if delta == 0:
                continue
            for i, w in enumerate(self.weights):
                diff = self.norm * float(delta) * self.af_der(net) * float(sample[i])
                # self.weights[i] += self.norm * delta * self.af_der(net) * sample[i]
                self.weights[i] += diff
                print(f'\t\twi = {i}, deltaWi = {diff}, w(i+1) = {self.weights[i]}')


def test_sets(net: Net, sets: list[list[int]]) -> (list[int], int):
    answers, mistakes = list(), 0
    for s in sets:
        net_ans, _ = net.launch(s[:-1])
        answers.append(net_ans)
        mistakes += 0 if net_ans == s[-1] else 1
        print(f'mistakes so far {mistakes}')
    print(f'mistakes {mistakes}')
    return answers, mistakes


def learn(net: Net, sets: list[list[int]]) -> list[list]:
    success_count, epochs, i = 0, list(), 0
    while i < 50:
        print(f'EPOCH {i}')
        answers, mistakes = test_sets(net, sets)
        epochs.append([i, net.weights.copy(), answers.copy(), mistakes])
        success_count += 1 if mistakes == 0 else 0
        net.learn_epoch(sets)
        i += 1
    return epochs


bf: Callable[[list[bool]], bool] = lambda args: (args[2] and args[3]) ^ (not args[0]) ^ (not args[1])
# bf: Callable[[list[bool]], bool] = lambda args: args[0] ^ (not args[1]) ^ (not (args[2] ^ args[3]))

threshold_af_out: Callable[[float], int] = lambda net: 1 if net >= 0 else 0
threshold_af_der: Callable[[float], float] = lambda net: 1

logistic_af_out: Callable[[float], int] = lambda net: 1 if (math.tanh(net) + 1) / 2 >= 0.5 else 0
logistic_af_der: Callable[[float], float] = lambda net: 1 / (2 * (math.cosh(net) ** 2))

sample_af_out: Callable[[float], float] = lambda net: 1 if 1 / (1 + math.exp(-net)) >= 0.5 else 0
sample_af_der: Callable[[float], float] = lambda net: math.exp(net) / ((math.exp(net) + 1) ** 2)

inputs = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]


def main():
    def append_bf_val(args: list[int]) -> list[int]:
        args.append(1 if bf(args) else 0)
        return args

    def append_left_true(args: list[int]) -> list[int]:
        args.insert(0, 1)
        return args

    global inputs
    inputs = list(map(append_bf_val, inputs))
    inputs = list(map(append_left_true, inputs))

    net1 = Net(weights_num=5,
               norm=0.3,
               af=logistic_af_out,
               af_der=logistic_af_der)

    def modify_lists(logs_list: list[list]) -> list[list]:
        for i, log in enumerate(logs_list):
            log[1] = ''.join(map('{: 6.2f}'.format, log[1]))
            log[2] = ''.join(map(str, log[2]))
        return logs_list

    logs = learn(net1, inputs)

    t = Texttable()
    logs = modify_lists(logs)
    logs.insert(0, ['k', 'w', 'y', 'E'])
    t.set_chars(['—', '|', '+', '—'])
    t.set_cols_dtype(['i', 't', 't', 'i'])
    t.add_rows(logs)
    print(logs[0])
    print(t.draw())


if __name__ == '__main__':
    main()
